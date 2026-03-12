use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use he_core::{
    CiphertextVector,
    ClientKey,
    HeError,
    HeKeySet,
    HeResult,
    ServerKey,
    decrypt_vector,
    encrypt_vector,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::model::EncryptedModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    pub session_id: String,
    pub has_input: bool,
    pub has_result: bool,
}

pub struct EncryptedInferenceService {
    model: EncryptedModel,
    client_key: ClientKey,
    public_key: he_core::PublicKey,
    server_key: ServerKey,
    input_store: Arc<RwLock<HashMap<String, CiphertextVector>>>,
    result_store: Arc<RwLock<HashMap<String, CiphertextVector>>>,
    next_session_id: AtomicU64,
}

impl EncryptedInferenceService {
    pub fn new(model: EncryptedModel, keys: HeKeySet) -> Self {
        Self {
            model,
            client_key: keys.client_key,
            public_key: keys.public_key,
            server_key: keys.server_key,
            input_store: Arc::new(RwLock::new(HashMap::new())),
            result_store: Arc::new(RwLock::new(HashMap::new())),
            next_session_id: AtomicU64::new(1),
        }
    }

    pub fn create_session(&self) -> String {
        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        format!("he-session-{id:08}")
    }

    pub fn encrypt_input(&self, input: &[f64]) -> HeResult<CiphertextVector> {
        encrypt_vector(&self.public_key, input)
    }

    pub fn upload_input(&self, session_id: &str, input: CiphertextVector) -> HeResult<()> {
        if input.len() != self.model.input_dim() {
            return Err(HeError::Operation(format!(
                "ciphertext length {} does not match model input dimension {}",
                input.len(),
                self.model.input_dim()
            )));
        }

        self.input_store.write().insert(session_id.to_string(), input);
        Ok(())
    }

    pub fn run_inference(&self, session_id: &str) -> HeResult<CiphertextVector> {
        let input = self
            .input_store
            .read()
            .get(session_id)
            .cloned()
            .ok_or_else(|| HeError::Operation(format!("unknown session id {session_id}")))?;

        let result = self.model.infer_encrypted(&input, &self.server_key)?;
        self.result_store
            .write()
            .insert(session_id.to_string(), result.clone());
        Ok(result)
    }

    pub fn get_result(&self, session_id: &str) -> Option<CiphertextVector> {
        self.result_store.read().get(session_id).cloned()
    }

    pub fn decrypt_result(&self, ciphertext: &CiphertextVector) -> HeResult<Vec<f64>> {
        decrypt_vector(&self.client_key, ciphertext)
    }

    pub fn infer_plaintext_roundtrip(&self, input: &[f64]) -> HeResult<Vec<f64>> {
        let ciphertext = self.encrypt_input(input)?;
        let session_id = self.create_session();
        self.upload_input(&session_id, ciphertext)?;
        let result = self.run_inference(&session_id)?;
        self.decrypt_result(&result)
    }

    pub fn session_status(&self, session_id: &str) -> SessionStatus {
        SessionStatus {
            session_id: session_id.to_string(),
            has_input: self.input_store.read().contains_key(session_id),
            has_result: self.result_store.read().contains_key(session_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use he_core::{Activation, HeParameters, generate_keys};
    use crate::model::DenseLayer;

    #[test]
    fn test_service_runs_three_party_flow() {
        let model = EncryptedModel::new(
            vec![
                DenseLayer::new(
                    vec![vec![0.5, 0.25], vec![0.1, 0.9]],
                    vec![0.0, 0.1],
                    Activation::ReluApprox,
                )
                .unwrap(),
                DenseLayer::new(
                    vec![vec![0.4, -0.2]],
                    vec![0.3],
                    Activation::SigmoidApprox,
                )
                .unwrap(),
            ],
            vec!["abnormal".to_string()],
        )
        .unwrap();
        let service = EncryptedInferenceService::new(model.clone(), generate_keys(HeParameters::default()).unwrap());
        let session_id = service.create_session();

        let input = vec![1.0, -0.5];
        let ciphertext = service.encrypt_input(&input).unwrap();
        service.upload_input(&session_id, ciphertext).unwrap();
        let result = service.run_inference(&session_id).unwrap();
        let decrypted = service.decrypt_result(&result).unwrap();
        let expected = model.infer_plaintext(&input).unwrap();

        assert_eq!(service.session_status(&session_id).has_result, true);
        assert!((decrypted[0] - expected[0]).abs() < 1e-9);
    }

    #[test]
    fn test_unknown_session_returns_error() {
        let model = EncryptedModel::new(
            vec![DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).unwrap()],
            vec!["ok".to_string()],
        )
        .unwrap();
        let service = EncryptedInferenceService::new(model, generate_keys(HeParameters::default()).unwrap());

        let err = service.run_inference("missing-session").unwrap_err();
        assert!(matches!(err, HeError::Operation(_)));
    }
}
