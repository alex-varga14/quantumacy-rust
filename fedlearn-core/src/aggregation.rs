//! Federated Averaging (FedAvg) aggregation algorithm.
//!
//! Implements the McMahan et al. (2017) algorithm:
//! 1. Server distributes global model to selected clients
//! 2. Each client trains locally for E epochs
//! 3. Clients send weight updates back to server
//! 4. Server computes weighted average: w_global = Σ(n_k/n * w_k)
//!    where n_k is client k's sample count and n = Σn_k
//!
//! Supports:
//! - Sample-weighted averaging (default)
//! - Uniform averaging
//! - Momentum-based aggregation
//! - Minimum client participation threshold

use crate::error::{FedError, FedResult};
use crate::model::{ModelUpdate, ModelWeights};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// FedAvg aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedAvgConfig {
    /// Minimum number of client updates to proceed with aggregation
    pub min_clients: usize,
    /// Weighting strategy
    pub weighting: WeightingStrategy,
    /// Momentum factor for exponential moving average (0 = no momentum)
    pub momentum: f32,
    /// Maximum allowed L2 norm for any single update (outlier rejection)
    pub max_update_norm: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightingStrategy {
    /// Weight by number of training samples (default FedAvg)
    SampleWeighted,
    /// Equal weight for all clients
    Uniform,
}

impl Default for FedAvgConfig {
    fn default() -> Self {
        Self {
            min_clients: 2,
            weighting: WeightingStrategy::SampleWeighted,
            momentum: 0.0,
            max_update_norm: None,
        }
    }
}

/// FedAvg aggregator
pub struct FedAvg {
    config: FedAvgConfig,
    /// Previous global model for momentum
    previous_global: Option<ModelWeights>,
    /// Current round number
    round: u32,
}

impl FedAvg {
    pub fn new(config: FedAvgConfig) -> Self {
        Self {
            config,
            previous_global: None,
            round: 0,
        }
    }

    /// Aggregate client updates into a new global model.
    ///
    /// # Arguments
    /// * `updates` - Client model updates from the current round
    /// * `global_weights` - Current global model weights (for delta computation)
    ///
    /// # Returns
    /// New aggregated global weights
    pub fn aggregate(
        &mut self,
        updates: &[ModelUpdate],
        global_weights: &ModelWeights,
    ) -> FedResult<ModelWeights> {
        if updates.is_empty() {
            return Err(FedError::NoUpdates);
        }

        if updates.len() < self.config.min_clients {
            warn!(
                received = updates.len(),
                required = self.config.min_clients,
                "Insufficient client participation"
            );
            return Err(FedError::Aggregation(format!(
                "Need {} clients, got {}",
                self.config.min_clients,
                updates.len()
            )));
        }

        // Validate dimensions
        let expected_params = global_weights.num_params;
        for update in updates {
            if update.weights.num_params != expected_params {
                return Err(FedError::DimensionMismatch {
                    expected: expected_params,
                    got: update.weights.num_params,
                });
            }
        }

        // Filter outliers if max_update_norm is set
        let valid_updates: Vec<&ModelUpdate> = if let Some(max_norm) = self.config.max_update_norm {
            updates
                .iter()
                .filter(|u| {
                    let norm = u.weights.l2_norm();
                    if norm > max_norm {
                        warn!(
                            client = %u.client_id,
                            norm = format!("{norm:.4}"),
                            max = format!("{max_norm:.4}"),
                            "Rejecting outlier update"
                        );
                        false
                    } else {
                        true
                    }
                })
                .collect()
        } else {
            updates.iter().collect()
        };

        if valid_updates.is_empty() {
            return Err(FedError::Aggregation(
                "All updates rejected as outliers".into(),
            ));
        }

        info!(
            round = self.round,
            clients = valid_updates.len(),
            "Aggregating updates"
        );

        // Compute weights for each client
        let total_samples: usize = valid_updates.iter().map(|u| u.num_samples).sum();
        let client_weights: Vec<f32> = match self.config.weighting {
            WeightingStrategy::SampleWeighted => valid_updates
                .iter()
                .map(|u| u.num_samples as f32 / total_samples as f32)
                .collect(),
            WeightingStrategy::Uniform => {
                let w = 1.0 / valid_updates.len() as f32;
                vec![w; valid_updates.len()]
            }
        };

        // Weighted average using rayon for parallelism
        let num_params = expected_params;
        let aggregated_flat: Vec<f32> = (0..num_params)
            .into_par_iter()
            .map(|i| {
                valid_updates
                    .iter()
                    .zip(client_weights.iter())
                    .map(|(update, &weight)| {
                        let flat = update.weights.flatten();
                        flat[i] * weight
                    })
                    .sum::<f32>()
            })
            .collect();

        let mut aggregated = ModelWeights::from_flat(&aggregated_flat, global_weights)?;

        // Apply momentum if configured
        if self.config.momentum > 0.0 {
            if let Some(ref prev) = self.previous_global {
                let m = self.config.momentum;
                let momentum_flat: Vec<f32> = aggregated
                    .flatten()
                    .iter()
                    .zip(prev.flatten().iter())
                    .map(|(&curr, &prev)| (1.0 - m) * curr + m * prev)
                    .collect();
                aggregated = ModelWeights::from_flat(&momentum_flat, global_weights)?;
            }
        }

        // Store for next round's momentum
        self.previous_global = Some(aggregated.clone());
        self.round += 1;

        let avg_loss: f64 =
            valid_updates.iter().map(|u| u.loss).sum::<f64>() / valid_updates.len() as f64;

        info!(
            round = self.round,
            avg_loss = format!("{avg_loss:.6}"),
            total_samples,
            "Aggregation complete"
        );

        Ok(aggregated)
    }

    pub fn round(&self) -> u32 {
        self.round
    }

    pub fn reset(&mut self) {
        self.previous_global = None;
        self.round = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerWeights, UpdateMetadata};

    fn make_update(client_id: &str, data: Vec<f32>, num_samples: usize) -> ModelUpdate {
        let n = data.len();
        ModelUpdate {
            client_id: client_id.to_string(),
            weights: ModelWeights {
                layers: vec![LayerWeights {
                    name: "dense".to_string(),
                    shape: vec![n],
                    data,
                }],
                num_params: n,
            },
            num_samples,
            loss: 0.5,
            metadata: UpdateMetadata {
                local_epochs: 1,
                learning_rate: 0.01,
                batch_size: 32,
                round: 0,
                training_time_ms: 100,
            },
        }
    }

    fn make_global(data: Vec<f32>) -> ModelWeights {
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
    fn test_uniform_average() {
        let mut fedavg = FedAvg::new(FedAvgConfig {
            min_clients: 2,
            weighting: WeightingStrategy::Uniform,
            momentum: 0.0,
            max_update_norm: None,
        });

        let updates = vec![
            make_update("c1", vec![2.0, 4.0], 100),
            make_update("c2", vec![4.0, 6.0], 100),
        ];
        let global = make_global(vec![1.0, 1.0]);

        let result = fedavg.aggregate(&updates, &global).unwrap();
        let flat = result.flatten();
        assert!((flat[0] - 3.0).abs() < 1e-6);
        assert!((flat[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_weighted_average() {
        let mut fedavg = FedAvg::new(FedAvgConfig {
            min_clients: 2,
            weighting: WeightingStrategy::SampleWeighted,
            momentum: 0.0,
            max_update_norm: None,
        });

        let updates = vec![
            make_update("c1", vec![10.0], 300), // 75% weight
            make_update("c2", vec![2.0], 100),  // 25% weight
        ];
        let global = make_global(vec![0.0]);

        let result = fedavg.aggregate(&updates, &global).unwrap();
        let expected = 10.0 * 0.75 + 2.0 * 0.25; // 8.0
        assert!((result.flatten()[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_insufficient_clients() {
        let mut fedavg = FedAvg::new(FedAvgConfig {
            min_clients: 3,
            ..Default::default()
        });

        let updates = vec![
            make_update("c1", vec![1.0], 100),
            make_update("c2", vec![2.0], 100),
        ];
        let global = make_global(vec![0.0]);

        assert!(matches!(
            fedavg.aggregate(&updates, &global),
            Err(FedError::Aggregation(_))
        ));
    }

    #[test]
    fn test_no_updates() {
        let mut fedavg = FedAvg::new(FedAvgConfig::default());
        let global = make_global(vec![0.0]);
        assert!(matches!(
            fedavg.aggregate(&[], &global),
            Err(FedError::NoUpdates)
        ));
    }

    #[test]
    fn test_outlier_rejection() {
        let mut fedavg = FedAvg::new(FedAvgConfig {
            min_clients: 1,
            max_update_norm: Some(10.0),
            weighting: WeightingStrategy::Uniform,
            momentum: 0.0,
        });

        let updates = vec![
            make_update("normal", vec![1.0, 1.0], 100),
            make_update("outlier", vec![100.0, 100.0], 100), // norm >> 10
        ];
        let global = make_global(vec![0.0, 0.0]);

        let result = fedavg.aggregate(&updates, &global).unwrap();
        // Only the normal update should be included
        assert!((result.flatten()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut fedavg = FedAvg::new(FedAvgConfig {
            min_clients: 1,
            ..Default::default()
        });

        let updates = vec![make_update("c1", vec![1.0, 2.0, 3.0], 100)];
        let global = make_global(vec![0.0, 0.0]); // Mismatch: 2 vs 3

        assert!(matches!(
            fedavg.aggregate(&updates, &global),
            Err(FedError::DimensionMismatch { .. })
        ));
    }
}
