//! Model abstraction for federated learning.
//!
//! Provides framework-agnostic types for model weights and updates
//! that can be serialized, aggregated, and transmitted.

use crate::error::FedResult;
use serde::{Deserialize, Serialize};

/// Flattened model weights as a vector of f32 parameters.
/// Each named layer maps to a contiguous slice of parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Layer name → parameter values
    pub layers: Vec<LayerWeights>,
    /// Total parameter count
    pub num_params: usize,
}

/// Weights for a single named layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A client's model update after local training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Client identifier
    pub client_id: String,
    /// Updated weights (or weight deltas)
    pub weights: ModelWeights,
    /// Number of local training samples
    pub num_samples: usize,
    /// Local training loss (final)
    pub loss: f64,
    /// Training metadata
    pub metadata: UpdateMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMetadata {
    pub local_epochs: u32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub round: u32,
    pub training_time_ms: u64,
}

/// Trait for a model that can participate in federated learning
pub trait FederatedModel: Send + Sync {
    /// Get current model weights
    fn get_weights(&self) -> FedResult<ModelWeights>;

    /// Set model weights (e.g., after receiving aggregated update)
    fn set_weights(&mut self, weights: &ModelWeights) -> FedResult<()>;

    /// Train on local data for one or more epochs and return the update
    fn train_local(
        &mut self,
        data: &[Vec<f32>],
        labels: &[Vec<f32>],
        config: &LocalTrainConfig,
    ) -> FedResult<ModelUpdate>;

    /// Evaluate model on given data, return (loss, accuracy)
    fn evaluate(&self, data: &[Vec<f32>], labels: &[Vec<f32>]) -> FedResult<(f64, f64)>;

    /// Number of trainable parameters
    fn num_params(&self) -> usize;
}

/// Configuration for local training on a client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTrainConfig {
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub round: u32,
}

impl Default for LocalTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 32,
            learning_rate: 0.01,
            round: 0,
        }
    }
}

impl ModelWeights {
    /// Create an empty weights container
    pub fn empty() -> Self {
        Self {
            layers: Vec::new(),
            num_params: 0,
        }
    }

    /// Get all parameters as a flat vector
    pub fn flatten(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|l| l.data.iter().copied())
            .collect()
    }

    /// Create weights from flat vector, preserving layer structure from a template
    pub fn from_flat(flat: &[f32], template: &ModelWeights) -> FedResult<Self> {
        let mut offset = 0;
        let mut layers = Vec::new();

        for layer in &template.layers {
            let len = layer.data.len();
            if offset + len > flat.len() {
                return Err(crate::error::FedError::DimensionMismatch {
                    expected: template.num_params,
                    got: flat.len(),
                });
            }
            layers.push(LayerWeights {
                name: layer.name.clone(),
                shape: layer.shape.clone(),
                data: flat[offset..offset + len].to_vec(),
            });
            offset += len;
        }

        Ok(Self {
            num_params: flat.len(),
            layers,
        })
    }

    /// Element-wise addition
    pub fn add(&self, other: &ModelWeights) -> FedResult<ModelWeights> {
        if self.num_params != other.num_params {
            return Err(crate::error::FedError::DimensionMismatch {
                expected: self.num_params,
                got: other.num_params,
            });
        }
        let flat: Vec<f32> = self
            .flatten()
            .iter()
            .zip(other.flatten().iter())
            .map(|(a, b)| a + b)
            .collect();
        ModelWeights::from_flat(&flat, self)
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> ModelWeights {
        let flat: Vec<f32> = self.flatten().iter().map(|&x| x * factor).collect();
        // Safe to unwrap since dimensions match
        ModelWeights::from_flat(&flat, self).unwrap()
    }

    /// L2 norm of all parameters
    pub fn l2_norm(&self) -> f32 {
        self.flatten().iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_weights(data: Vec<f32>) -> ModelWeights {
        let n = data.len();
        ModelWeights {
            layers: vec![LayerWeights {
                name: "dense".to_string(),
                shape: vec![n],
                data,
            }],
            num_params: n,
        }
    }

    #[test]
    fn test_flatten_roundtrip() {
        let w = make_weights(vec![1.0, 2.0, 3.0]);
        let flat = w.flatten();
        let restored = ModelWeights::from_flat(&flat, &w).unwrap();
        assert_eq!(restored.flatten(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add() {
        let a = make_weights(vec![1.0, 2.0, 3.0]);
        let b = make_weights(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.flatten(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scale() {
        let w = make_weights(vec![2.0, 4.0, 6.0]);
        let scaled = w.scale(0.5);
        assert_eq!(scaled.flatten(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_l2_norm() {
        let w = make_weights(vec![3.0, 4.0]);
        assert!((w.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = make_weights(vec![1.0, 2.0]);
        let b = make_weights(vec![1.0, 2.0, 3.0]);
        assert!(a.add(&b).is_err());
    }
}
