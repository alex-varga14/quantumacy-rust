use fedlearn_core::error::FedResult;
use fedlearn_core::model::{FederatedModel, LocalTrainConfig, ModelUpdate, ModelWeights};
use he_inference::EncryptedModel;
use serde::{Deserialize, Serialize};

use crate::common::{DenseClassifier, OutputKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChestScanConfig {
    pub image_width: usize,
    pub image_height: usize,
    pub hidden_dim: usize,
    pub seed: u64,
}

impl Default for ChestScanConfig {
    fn default() -> Self {
        Self {
            image_width: 32,
            image_height: 32,
            hidden_dim: 32,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChestScanModel {
    client_id: String,
    config: ChestScanConfig,
    network: DenseClassifier,
}

impl ChestScanModel {
    pub fn new(client_id: impl Into<String>, config: ChestScanConfig) -> Self {
        let input_dim = config.image_width * config.image_height;
        let network = DenseClassifier::new(
            input_dim,
            config.hidden_dim,
            1,
            OutputKind::Binary,
            config.seed,
        );

        Self {
            client_id: client_id.into(),
            config,
            network,
        }
    }

    pub fn image_shape(&self) -> (usize, usize) {
        (self.config.image_width, self.config.image_height)
    }

    pub fn to_encrypted_model(&self) -> FedResult<EncryptedModel> {
        self.network
            .to_encrypted_model(vec!["abnormal".to_string()])
    }
}

impl Default for ChestScanModel {
    fn default() -> Self {
        Self::new("chestscan-client", ChestScanConfig::default())
    }
}

impl FederatedModel for ChestScanModel {
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
        let samples = vec![
            vec![1.0, 1.0, 0.8, 0.9],
            vec![0.9, 0.8, 1.0, 1.0],
            vec![0.1, 0.0, 0.2, 0.1],
            vec![0.0, 0.2, 0.1, 0.0],
        ];
        let labels = vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0]];
        (samples, labels)
    }

    #[test]
    fn test_chestscan_training_improves_accuracy_on_simple_data() {
        let (data, labels) = synthetic_dataset();
        let mut model = ChestScanModel::new(
            "site-a",
            ChestScanConfig {
                image_width: 2,
                image_height: 2,
                hidden_dim: 6,
                seed: 7,
            },
        );

        let (_, baseline_accuracy) = model.evaluate(&data, &labels).unwrap();
        let update = model
            .train_local(
                &data,
                &labels,
                &LocalTrainConfig {
                    epochs: 80,
                    batch_size: 2,
                    learning_rate: 0.2,
                    round: 3,
                },
            )
            .unwrap();
        let (_, trained_accuracy) = model.evaluate(&data, &labels).unwrap();

        assert_eq!(update.metadata.round, 3);
        assert!(trained_accuracy >= baseline_accuracy);
        assert!(trained_accuracy >= 0.75);
    }

    #[test]
    fn test_chestscan_encrypted_model_matches_plaintext_output() {
        let model = ChestScanModel::new(
            "site-a",
            ChestScanConfig {
                image_width: 2,
                image_height: 2,
                hidden_dim: 4,
                seed: 11,
            },
        );
        let encrypted_model = model.to_encrypted_model().unwrap();
        let input: Vec<f64> = vec![0.9, 0.8, 0.7, 0.9];
        let keys = generate_keys(HeParameters::default()).unwrap();
        let ciphertext = encrypt_vector(&keys.public_key, &input).unwrap();

        let encrypted_output = encrypted_model
            .infer_encrypted(&ciphertext, &keys.server_key)
            .unwrap();
        let decrypted_output = decrypt_vector(&keys.client_key, &encrypted_output).unwrap();
        let plaintext_output = encrypted_model.infer_plaintext(&input).unwrap();

        assert_eq!(decrypted_output.len(), 1);
        assert!((decrypted_output[0] - plaintext_output[0]).abs() < 1e-9);
    }
}
