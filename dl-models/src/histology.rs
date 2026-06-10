use fedlearn_core::error::FedResult;
use fedlearn_core::model::{FederatedModel, LocalTrainConfig, ModelUpdate, ModelWeights};
use he_inference::EncryptedModel;
use serde::{Deserialize, Serialize};

use crate::common::{DenseClassifier, OutputKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistologyConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_classes: usize,
    pub seed: u64,
}

impl Default for HistologyConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 24,
            num_classes: 3,
            seed: 99,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistologyModel {
    client_id: String,
    config: HistologyConfig,
    network: DenseClassifier,
}

impl HistologyModel {
    pub fn new(client_id: impl Into<String>, config: HistologyConfig) -> Self {
        let network = DenseClassifier::new(
            config.input_dim,
            config.hidden_dim,
            config.num_classes,
            OutputKind::Multiclass,
            config.seed,
        );

        Self {
            client_id: client_id.into(),
            config,
            network,
        }
    }

    pub fn to_encrypted_model(&self) -> FedResult<EncryptedModel> {
        let mut labels: Vec<String> = ["epithelial", "stromal", "immune"]
            .iter()
            .take(self.config.num_classes)
            .map(|label| (*label).to_string())
            .collect();
        while labels.len() < self.config.num_classes {
            labels.push(format!("class-{}", labels.len()));
        }
        self.network.to_encrypted_model(labels)
    }
}

impl Default for HistologyModel {
    fn default() -> Self {
        Self::new("histology-client", HistologyConfig::default())
    }
}

impl FederatedModel for HistologyModel {
    fn get_weights(&self) -> FedResult<ModelWeights> {
        Ok(self.network.get_weights())
    }

    fn set_weights(&mut self, weights: &ModelWeights) -> FedResult<()> {
        self.network.set_weights(weights)
    }

    fn train_local(
        &mut self,
        data: &[Vec<f32>],
        labels: &[Vec<f32>],
        config: &LocalTrainConfig,
    ) -> FedResult<ModelUpdate> {
        self.network
            .train_local(&self.client_id, data, labels, config)
    }

    fn evaluate(&self, data: &[Vec<f32>], labels: &[Vec<f32>]) -> FedResult<(f64, f64)> {
        self.network.evaluate(data, labels)
    }

    fn num_params(&self) -> usize {
        self.network.num_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fedlearn_core::model::FederatedModel;
    use he_core::{decrypt_vector, encrypt_vector, generate_keys, HeParameters};

    fn synthetic_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let data = vec![
            vec![1.0, 0.9, 0.8, 0.1],
            vec![0.9, 1.0, 0.7, 0.0],
            vec![0.1, 0.2, 1.0, 0.9],
            vec![0.0, 0.1, 0.9, 1.0],
            vec![0.8, 0.1, 0.2, 1.0],
            vec![0.7, 0.0, 0.1, 0.9],
        ];
        let labels = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
        ];
        (data, labels)
    }

    #[test]
    fn test_histology_training_handles_multiclass_data() {
        let (data, labels) = synthetic_dataset();
        let mut model = HistologyModel::new(
            "lab-a",
            HistologyConfig {
                input_dim: 4,
                hidden_dim: 8,
                num_classes: 3,
                seed: 5,
            },
        );

        model
            .train_local(
                &data,
                &labels,
                &LocalTrainConfig {
                    epochs: 100,
                    batch_size: 3,
                    learning_rate: 0.15,
                    round: 1,
                },
            )
            .unwrap();
        let (_, accuracy) = model.evaluate(&data, &labels).unwrap();

        assert!(accuracy >= 0.66);
    }

    #[test]
    fn test_histology_encrypted_conversion_preserves_logits() {
        let model = HistologyModel::new(
            "lab-a",
            HistologyConfig {
                input_dim: 4,
                hidden_dim: 6,
                num_classes: 3,
                seed: 13,
            },
        );
        let encrypted_model = model.to_encrypted_model().unwrap();
        let input: Vec<f64> = vec![0.9, 0.3, 0.2, 0.8];
        let keys = generate_keys(HeParameters::default()).unwrap();
        let ciphertext = encrypt_vector(&keys.public_key, &input).unwrap();

        let encrypted_output = encrypted_model
            .infer_encrypted(&ciphertext, &keys.server_key)
            .unwrap();
        let decrypted_output = decrypt_vector(&keys.client_key, &encrypted_output).unwrap();
        let plaintext_output = encrypted_model.infer_plaintext(&input).unwrap();

        assert_eq!(decrypted_output.len(), 3);
        for (decrypted, plaintext) in decrypted_output.iter().zip(plaintext_output.iter()) {
            assert!((decrypted - plaintext).abs() < 1e-9);
        }
    }
}
