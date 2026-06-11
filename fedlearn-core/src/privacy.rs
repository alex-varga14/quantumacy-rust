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
use crate::rdp::RdpAccountant;
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
    /// Cumulative ε spent so far (RDP-accounted, kept for API compatibility)
    epsilon_spent: f64,
    /// Number of DP operations applied
    num_compositions: u32,
    /// Rényi DP accountant for tight multi-round composition
    rdp: RdpAccountant,
    /// RNG for noise generation
    rng: ChaCha20Rng,
}

impl DifferentialPrivacy {
    pub fn new(config: DpConfig) -> Self {
        let rdp = RdpAccountant::new(config.noise_multiplier);
        Self {
            config,
            epsilon_spent: 0.0,
            num_compositions: 0,
            rdp,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(config: DpConfig, seed: u64) -> Self {
        let rdp = RdpAccountant::new(config.noise_multiplier);
        Self {
            config,
            epsilon_spent: 0.0,
            num_compositions: 0,
            rdp,
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
        // Check privacy budget using RDP composition: refuse the release if
        // the ε after this step would exceed the budget (fail-closed, before
        // any noise is applied or weights are released).
        let mut probe = self.rdp.clone();
        probe.step();
        if probe.epsilon(self.config.delta) > self.config.epsilon_budget {
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

        self.rdp.step();
        self.epsilon_spent = self.rdp.epsilon(self.config.delta);
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

    /// Classical single-round ε of the Gaussian mechanism (diagnostic).
    ///
    /// The noise std is `σ · sensitivity`, so the sensitivity cancels and the
    /// classical single-round Gaussian bound depends only on σ and δ:
    ///
    /// ε = sqrt(2 · ln(1.25/δ)) / σ
    ///
    /// A non-positive σ provides no privacy, so ε is infinite (refusal).
    ///
    /// Budget enforcement uses the tighter [`RdpAccountant`] composition;
    /// this bound is retained as a reference point for audits.
    pub fn compute_round_epsilon(&self) -> f64 {
        let sigma = self.config.noise_multiplier;
        if sigma <= 0.0 {
            return f64::INFINITY;
        }
        (2.0 * (1.25 / self.config.delta).ln()).sqrt() / sigma
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
        self.rdp = RdpAccountant::new(self.config.noise_multiplier);
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

    fn dp_with(noise_multiplier: f64, delta: f64) -> DifferentialPrivacy {
        DifferentialPrivacy::new(DpConfig {
            noise_multiplier,
            delta,
            ..Default::default()
        })
    }

    proptest::proptest! {
        /// (a) The per-round ε must equal the closed-form Gaussian bound
        /// sqrt(2·ln(1.25/δ)) / σ.
        #[test]
        fn prop_round_epsilon_matches_closed_form(
            sigma in 0.5f64..=10.0,
            delta in 1e-7f64..=1e-3,
        ) {
            let dp = dp_with(sigma, delta);
            let expected = (2.0 * (1.25 / delta).ln()).sqrt() / sigma;
            let actual = dp.compute_round_epsilon();
            proptest::prop_assert!(
                (actual - expected).abs() < 1e-12,
                "actual={actual}, expected={expected}"
            );
        }

        /// (b1) ε is monotonically decreasing in σ: more noise → more privacy.
        #[test]
        fn prop_epsilon_decreasing_in_sigma(
            sigma in 0.5f64..=10.0,
            delta in 1e-7f64..=1e-3,
            factor in 1.01f64..=4.0,
        ) {
            let eps_lo_sigma = dp_with(sigma, delta).compute_round_epsilon();
            let eps_hi_sigma = dp_with(sigma * factor, delta).compute_round_epsilon();
            proptest::prop_assert!(
                eps_hi_sigma < eps_lo_sigma,
                "ε(σ={}) = {eps_hi_sigma} should be < ε(σ={sigma}) = {eps_lo_sigma}",
                sigma * factor
            );
        }

        /// (b2) ε is monotonically decreasing in δ: a looser failure
        /// probability requires less ε for the same noise.
        #[test]
        fn prop_epsilon_decreasing_in_delta(
            sigma in 0.5f64..=10.0,
            delta in 1e-7f64..=1e-3,
            factor in 1.01f64..=10.0,
        ) {
            let eps_lo_delta = dp_with(sigma, delta).compute_round_epsilon();
            let eps_hi_delta = dp_with(sigma, delta * factor).compute_round_epsilon();
            proptest::prop_assert!(
                eps_hi_delta < eps_lo_delta,
                "ε(δ={}) = {eps_hi_delta} should be < ε(δ={delta}) = {eps_lo_delta}",
                delta * factor
            );
        }

        /// (c) Non-positive σ provides no privacy: ε must be infinite (refusal).
        #[test]
        fn prop_nonpositive_sigma_yields_infinite_epsilon(
            sigma in -10.0f64..=0.0,
            delta in 1e-7f64..=1e-3,
        ) {
            let eps = dp_with(sigma, delta).compute_round_epsilon();
            proptest::prop_assert!(eps.is_infinite() && eps > 0.0, "ε = {eps}");
        }
    }

    #[test]
    fn test_rdp_budget_allows_more_rounds_than_naive() {
        // σ=1, δ=1e-5, budget=10: naive linear summation of the classical
        // per-round bound (ε ≈ 4.84) only permits 2 rounds, but the RDP
        // accountant permits exactly 3 (ε₃ ≈ 9.84, ε₄ ≈ 11.76).
        let mut dp = DifferentialPrivacy::with_seed(
            DpConfig {
                noise_multiplier: 1.0,
                max_grad_norm: 1.0,
                delta: 1e-5,
                epsilon_budget: 10.0,
                ..Default::default()
            },
            42,
        );

        let w = make_weights(vec![1.0, 2.0, 3.0]);
        for round in 1..=3 {
            dp.add_noise(&w, 10)
                .unwrap_or_else(|e| panic!("round {round} should fit in budget: {e}"));
        }
        match dp.add_noise(&w, 10) {
            Err(FedError::PrivacyBudgetExhausted { .. }) => {}
            other => panic!("4th round must exhaust budget, got {other:?}"),
        }
        assert_eq!(dp.num_compositions(), 3);
    }

    #[test]
    fn test_epsilon_spent_tracks_rdp_accountant() {
        let mut dp = DifferentialPrivacy::with_seed(
            DpConfig {
                noise_multiplier: 1.0,
                max_grad_norm: 1.0,
                delta: 1e-5,
                epsilon_budget: 100.0,
                ..Default::default()
            },
            7,
        );

        let w = make_weights(vec![1.0, 2.0]);
        let mut reference = crate::rdp::RdpAccountant::new(1.0);
        for _ in 0..5 {
            dp.add_noise(&w, 10).unwrap();
            reference.step();
        }

        let expected = reference.epsilon(1e-5);
        assert!(
            (dp.epsilon_spent() - expected).abs() < 1e-9,
            "epsilon_spent = {}, RDP accountant = {expected}",
            dp.epsilon_spent()
        );
        assert!(
            (dp.remaining_budget() - (100.0 - expected)).abs() < 1e-9,
            "remaining_budget inconsistent"
        );
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
