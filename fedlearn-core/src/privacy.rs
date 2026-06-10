//! Differential privacy for federated learning.
//!
//! Implements the DP-SGD approach (Abadi et al., 2016):
//! 1. **Gradient clipping**: Bound per-sample gradient L2 norm to limit sensitivity
//! 2. **Gaussian noise**: Add calibrated noise to aggregated gradients
//! 3. **Privacy accounting**: Track cumulative privacy budget (ε, δ)
//!
//! Also supports local differential privacy (LDP) where each client
//! adds noise before sending updates.

use crate::error::{FedError, FedResult};
use crate::model::ModelWeights;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpConfig {
    /// Maximum L2 norm for gradient clipping (sensitivity bound)
    pub max_grad_norm: f32,
    /// Noise multiplier σ (noise_std = σ * sensitivity / batch_size)
    pub noise_multiplier: f64,
    /// Target δ for (ε,δ)-DP
    pub delta: f64,
    /// Total privacy budget ε — training stops when exhausted
    pub epsilon_budget: f64,
    /// Whether to apply DP at client level (LDP) or server level (CDP)
    pub mode: DpMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DpMode {
    /// Central DP: server adds noise after aggregation
    Central,
    /// Local DP: each client adds noise before sending
    Local,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: 1.0,
            noise_multiplier: 1.0,
            delta: 1e-5,
            epsilon_budget: 10.0,
            mode: DpMode::Central,
        }
    }
}

/// Differential privacy mechanism
pub struct DifferentialPrivacy {
    config: DpConfig,
    /// Cumulative ε spent so far
    epsilon_spent: f64,
    /// Number of DP operations applied
    num_compositions: u32,
    /// RNG for noise generation
    rng: ChaCha20Rng,
}

impl DifferentialPrivacy {
    pub fn new(config: DpConfig) -> Self {
        Self {
            config,
            epsilon_spent: 0.0,
            num_compositions: 0,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(config: DpConfig, seed: u64) -> Self {
        Self {
            config,
            epsilon_spent: 0.0,
            num_compositions: 0,
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Clip gradients/weights to have L2 norm ≤ max_grad_norm
    pub fn clip(&self, weights: &ModelWeights) -> ModelWeights {
        let norm = weights.l2_norm();
        if norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / norm;
            debug!(
                original_norm = format!("{norm:.4}"),
                clipped_to = format!("{:.4}", self.config.max_grad_norm),
                "Clipping gradient"
            );
            weights.scale(scale)
        } else {
            weights.clone()
        }
    }

    /// Add calibrated Gaussian noise to weights.
    /// noise_std = noise_multiplier * max_grad_norm / num_clients
    pub fn add_noise(
        &mut self,
        weights: &ModelWeights,
        num_clients: usize,
    ) -> FedResult<ModelWeights> {
        // Check privacy budget
        let round_epsilon = self.compute_round_epsilon(num_clients);
        if self.epsilon_spent + round_epsilon > self.config.epsilon_budget {
            return Err(FedError::PrivacyBudgetExhausted {
                epsilon: self.epsilon_spent,
            });
        }

        let noise_std =
            (self.config.noise_multiplier * self.config.max_grad_norm as f64) / num_clients as f64;

        let flat = weights.flatten();
        let noisy: Vec<f32> = flat
            .iter()
            .map(|&x| {
                let noise = self.gaussian_noise(noise_std);
                x + noise as f32
            })
            .collect();

        self.epsilon_spent += round_epsilon;
        self.num_compositions += 1;

        info!(
            noise_std = format!("{noise_std:.6}"),
            epsilon_spent = format!("{:.4}", self.epsilon_spent),
            epsilon_budget = format!("{:.4}", self.config.epsilon_budget),
            round = self.num_compositions,
            "Applied DP noise"
        );

        ModelWeights::from_flat(&noisy, weights)
    }

    /// Generate Gaussian noise using Box-Muller transform
    fn gaussian_noise(&mut self, std_dev: f64) -> f64 {
        let u1: f64 = self.rng.gen::<f64>().max(1e-10); // Avoid log(0)
        let u2: f64 = self.rng.gen::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * std_dev
    }

    /// Compute per-round ε using simple composition theorem.
    /// For tighter bounds, use Rényi DP accounting (future improvement).
    fn compute_round_epsilon(&self, num_clients: usize) -> f64 {
        let sigma = self.config.noise_multiplier;
        let sensitivity = self.config.max_grad_norm as f64 / num_clients as f64;

        if sigma <= 0.0 {
            return f64::INFINITY;
        }

        // Simple Gaussian mechanism: ε = sensitivity * sqrt(2 * ln(1.25/δ)) / σ
        let eps =
            sensitivity * (2.0 * (1.25 / self.config.delta).ln()).sqrt() / (sigma * sensitivity);

        eps.max(0.0)
    }

    /// Remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        (self.config.epsilon_budget - self.epsilon_spent).max(0.0)
    }

    /// Total ε spent so far
    pub fn epsilon_spent(&self) -> f64 {
        self.epsilon_spent
    }

    /// Number of DP rounds applied
    pub fn num_compositions(&self) -> u32 {
        self.num_compositions
    }

    /// Check if budget is exhausted
    pub fn budget_exhausted(&self) -> bool {
        self.epsilon_spent >= self.config.epsilon_budget
    }

    /// Reset accounting (e.g., for a new training session)
    pub fn reset(&mut self) {
        self.epsilon_spent = 0.0;
        self.num_compositions = 0;
    }

    /// Get the DP mode
    pub fn mode(&self) -> DpMode {
        self.config.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LayerWeights;

    fn make_weights(data: Vec<f32>) -> ModelWeights {
        let n = data.len();
        ModelWeights {
            layers: vec![LayerWeights {
                name: "test".to_string(),
                shape: vec![n],
                data,
            }],
            num_params: n,
        }
    }

    #[test]
    fn test_clip_within_bound() {
        let dp = DifferentialPrivacy::new(DpConfig {
            max_grad_norm: 10.0,
            ..Default::default()
        });

        let w = make_weights(vec![1.0, 2.0, 3.0]); // norm ≈ 3.74
        let clipped = dp.clip(&w);
        assert_eq!(clipped.flatten(), w.flatten()); // No clipping needed
    }

    #[test]
    fn test_clip_exceeds_bound() {
        let dp = DifferentialPrivacy::new(DpConfig {
            max_grad_norm: 1.0,
            ..Default::default()
        });

        let w = make_weights(vec![3.0, 4.0]); // norm = 5.0
        let clipped = dp.clip(&w);
        let norm = clipped.l2_norm();
        assert!((norm - 1.0).abs() < 1e-5, "Clipped norm: {norm}");
    }

    #[test]
    fn test_noise_changes_weights() {
        let mut dp = DifferentialPrivacy::with_seed(
            DpConfig {
                noise_multiplier: 1.0,
                max_grad_norm: 1.0,
                epsilon_budget: 100.0,
                ..Default::default()
            },
            42,
        );

        let w = make_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let noisy = dp.add_noise(&w, 10).unwrap();

        // Noise should change the values
        assert_ne!(noisy.flatten(), w.flatten());
        // But not by too much with reasonable noise multiplier
        let diff: f32 = noisy
            .flatten()
            .iter()
            .zip(w.flatten().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut dp = DifferentialPrivacy::with_seed(
            DpConfig {
                noise_multiplier: 0.01, // Very low noise → high ε per round
                max_grad_norm: 1.0,
                epsilon_budget: 1.0, // Very tight budget
                ..Default::default()
            },
            42,
        );

        let w = make_weights(vec![1.0]);

        // Should eventually exhaust budget
        let mut exhausted = false;
        for _ in 0..1000 {
            match dp.add_noise(&w, 10) {
                Ok(_) => continue,
                Err(FedError::PrivacyBudgetExhausted { .. }) => {
                    exhausted = true;
                    break;
                }
                Err(e) => panic!("Unexpected error: {e}"),
            }
        }
        assert!(exhausted, "Budget should have been exhausted");
    }

    #[test]
    fn test_gaussian_noise_distribution() {
        let mut dp = DifferentialPrivacy::with_seed(DpConfig::default(), 123);

        let samples: Vec<f64> = (0..10_000).map(|_| dp.gaussian_noise(1.0)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Mean should be ~0, variance ~1
        assert!(mean.abs() < 0.05, "Mean: {mean}");
        assert!((variance - 1.0).abs() < 0.1, "Variance: {variance}");
    }
}
