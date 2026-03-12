//! Training round orchestration.
//!
//! Manages the lifecycle of a federated learning round:
//! client selection → distribute model → local training → collect updates → aggregate

use crate::aggregation::{FedAvg, FedAvgConfig};
use crate::error::{FedError, FedResult};
use crate::model::{ModelUpdate, ModelWeights};
use crate::privacy::{DifferentialPrivacy, DpConfig, DpMode};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Configuration for a federated training session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Total number of federated rounds
    pub num_rounds: u32,
    /// Fraction of clients to select each round (0.0–1.0)
    pub client_fraction: f64,
    /// FedAvg aggregation config
    pub fedavg: FedAvgConfig,
    /// Differential privacy config (None = no DP)
    pub dp: Option<DpConfig>,
    /// Target accuracy to stop early
    pub target_accuracy: Option<f64>,
    /// Minimum improvement per round to continue (early stopping)
    pub min_delta: f64,
    /// Patience: number of rounds without improvement before stopping
    pub patience: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            client_fraction: 1.0,
            fedavg: FedAvgConfig::default(),
            dp: None,
            target_accuracy: None,
            min_delta: 1e-4,
            patience: 10,
        }
    }
}

/// Tracks training progress across rounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub round: u32,
    pub total_rounds: u32,
    pub global_loss: f64,
    pub global_accuracy: Option<f64>,
    pub num_participating_clients: usize,
    pub total_samples: usize,
    pub dp_epsilon_spent: Option<f64>,
    pub history: Vec<RoundResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    pub round: u32,
    pub avg_loss: f64,
    pub accuracy: Option<f64>,
    pub num_clients: usize,
    pub total_samples: usize,
}

/// Orchestrates a single federated round.
///
/// This is a pure function — network communication is handled externally.
pub fn execute_round(
    round: u32,
    global_weights: &ModelWeights,
    client_updates: Vec<ModelUpdate>,
    aggregator: &mut FedAvg,
    dp: Option<&mut DifferentialPrivacy>,
) -> FedResult<(ModelWeights, RoundResult)> {
    if client_updates.is_empty() {
        return Err(FedError::NoUpdates);
    }

    let num_clients = client_updates.len();
    let total_samples: usize = client_updates.iter().map(|u| u.num_samples).sum();
    let avg_loss: f64 = client_updates.iter().map(|u| u.loss).sum::<f64>() / num_clients as f64;

    info!(
        round,
        num_clients,
        total_samples,
        avg_loss = format!("{avg_loss:.6}"),
        "Executing federated round"
    );

    // Apply DP clipping to each update if using Central DP
    let processed_updates: Vec<ModelUpdate> = if let Some(ref dp_instance) = dp {
        if matches!(dp_instance.mode(), DpMode::Central) {
            client_updates
                .into_iter()
                .map(|mut u| {
                    u.weights = dp_instance.clip(&u.weights);
                    u
                })
                .collect()
        } else {
            client_updates
        }
    } else {
        client_updates
    };

    // Aggregate
    let mut aggregated = aggregator.aggregate(&processed_updates, global_weights)?;

    // Add DP noise if configured
    if let Some(dp_instance) = dp {
        if matches!(dp_instance.mode(), DpMode::Central) {
            aggregated = dp_instance.add_noise(&aggregated, num_clients)?;
        }
    }

    let result = RoundResult {
        round,
        avg_loss,
        accuracy: None, // Set externally after evaluation
        num_clients,
        total_samples,
    };

    Ok((aggregated, result))
}

/// Check early stopping criteria
pub fn should_stop_early(
    history: &[RoundResult],
    config: &TrainingConfig,
) -> bool {
    if history.len() < config.patience as usize + 1 {
        return false;
    }

    // Check target accuracy
    if let Some(target) = config.target_accuracy {
        if let Some(latest) = history.last() {
            if let Some(acc) = latest.accuracy {
                if acc >= target {
                    info!(accuracy = format!("{acc:.4}"), target = format!("{target:.4}"),
                          "Target accuracy reached");
                    return true;
                }
            }
        }
    }

    // Check for stagnation
    let recent = &history[history.len() - config.patience as usize..];
    let best_recent_loss = recent.iter().map(|r| r.avg_loss).fold(f64::MAX, f64::min);
    let earlier = &history[..history.len() - config.patience as usize];
    let best_earlier_loss = earlier.iter().map(|r| r.avg_loss).fold(f64::MAX, f64::min);

    if best_earlier_loss - best_recent_loss < config.min_delta {
        warn!(
            patience = config.patience,
            best_earlier = format!("{best_earlier_loss:.6}"),
            best_recent = format!("{best_recent_loss:.6}"),
            "Early stopping: no improvement"
        );
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerWeights, UpdateMetadata};

    fn make_update(id: &str, vals: Vec<f32>, samples: usize) -> ModelUpdate {
        let n = vals.len();
        ModelUpdate {
            client_id: id.to_string(),
            weights: ModelWeights {
                layers: vec![LayerWeights {
                    name: "dense".to_string(),
                    shape: vec![n],
                    data: vals,
                }],
                num_params: n,
            },
            num_samples: samples,
            loss: 0.5,
            metadata: UpdateMetadata {
                local_epochs: 1,
                learning_rate: 0.01,
                batch_size: 32,
                round: 0,
                training_time_ms: 50,
            },
        }
    }

    fn make_global(vals: Vec<f32>) -> ModelWeights {
        let n = vals.len();
        ModelWeights {
            layers: vec![LayerWeights {
                name: "dense".to_string(),
                shape: vec![n],
                data: vals,
            }],
            num_params: n,
        }
    }

    #[test]
    fn test_execute_round() {
        let global = make_global(vec![0.0, 0.0]);
        let updates = vec![
            make_update("c1", vec![1.0, 2.0], 100),
            make_update("c2", vec![3.0, 4.0], 100),
        ];
        let mut agg = FedAvg::new(FedAvgConfig::default());

        let (new_weights, result) = execute_round(1, &global, updates, &mut agg, None).unwrap();
        assert_eq!(result.num_clients, 2);
        assert_eq!(result.total_samples, 200);
        assert!(!new_weights.flatten().is_empty());
    }

    #[test]
    fn test_early_stopping() {
        let config = TrainingConfig {
            patience: 3,
            min_delta: 0.01,
            ..Default::default()
        };

        // Stagnating losses
        let history: Vec<RoundResult> = (0..10)
            .map(|i| RoundResult {
                round: i,
                avg_loss: 0.5, // No improvement
                accuracy: None,
                num_clients: 5,
                total_samples: 500,
            })
            .collect();

        assert!(should_stop_early(&history, &config));
    }

    #[test]
    fn test_no_early_stop_improving() {
        let config = TrainingConfig {
            patience: 3,
            min_delta: 0.01,
            ..Default::default()
        };

        let history: Vec<RoundResult> = (0..10)
            .map(|i| RoundResult {
                round: i,
                avg_loss: 1.0 - (i as f64 * 0.05), // Improving
                accuracy: None,
                num_clients: 5,
                total_samples: 500,
            })
            .collect();

        assert!(!should_stop_early(&history, &config));
    }
}
