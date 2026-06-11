//! Rényi differential privacy (RDP) accountant for the Gaussian mechanism.
//!
//! Tracks privacy loss across composed Gaussian-mechanism releases using
//! Rényi DP (Mironov, 2017). For the Gaussian mechanism with sensitivity Δ
//! and noise standard deviation σ·Δ (σ = noise multiplier), the RDP at
//! order α is:
//!
//!   ε_RDP(α) = α·Δ² / (2·(σ·Δ)²) = α / (2σ²)
//!
//! RDP composes additively per order, so after `k` steps the accumulated
//! RDP at order α is `k·α / (2σ²)`. The accountant converts back to
//! (ε, δ)-DP with the standard conversion (Mironov, 2017, Prop. 3):
//!
//!   ε(δ) = min over α of [ ε_RDP(α) + ln(1/δ) / (α − 1) ]
//!
//! minimized over a fixed grid of orders α > 1. This yields strictly
//! tighter multi-round bounds than naive linear ε-summation of the
//! classical per-round Gaussian bound.

/// Fixed grid of Rényi orders over which the (ε, δ) conversion is minimized.
const RDP_ORDERS: [f64; 18] = [
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0,
    64.0,
];

/// RDP accountant for repeated (non-subsampled) Gaussian-mechanism releases
/// with a fixed noise multiplier σ.
#[derive(Debug, Clone)]
pub struct RdpAccountant {
    noise_multiplier: f64,
    steps: u32,
}

impl RdpAccountant {
    /// Create an accountant for the Gaussian mechanism with the given noise
    /// multiplier σ (noise std = σ · sensitivity).
    pub fn new(noise_multiplier: f64) -> Self {
        Self {
            noise_multiplier,
            steps: 0,
        }
    }

    /// Record one composition (one noisy release).
    pub fn step(&mut self) {
        self.steps += 1;
    }

    /// Number of compositions recorded so far.
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Convert the accumulated RDP to an (ε, δ)-DP guarantee:
    /// ε(δ) = min over the order grid of [ k·α/(2σ²) + ln(1/δ)/(α−1) ].
    ///
    /// Returns 0 when no steps have been taken. Returns infinity (refusal)
    /// for a non-positive noise multiplier or a δ outside (0, 1).
    pub fn epsilon(&self, delta: f64) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        if self.noise_multiplier <= 0.0 || delta <= 0.0 || delta >= 1.0 {
            return f64::INFINITY;
        }

        let sigma_sq = self.noise_multiplier * self.noise_multiplier;
        let log_inv_delta = (1.0 / delta).ln();
        let steps = f64::from(self.steps);

        RDP_ORDERS
            .iter()
            .map(|&alpha| steps * alpha / (2.0 * sigma_sq) + log_inv_delta / (alpha - 1.0))
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 1.0;
    const DELTA: f64 = 1e-5;

    /// Grid minimum of the RDP-to-(ε, δ) conversion, derived independently
    /// in the test from the formulas in the module docs.
    fn expected_epsilon(sigma: f64, steps: u32, delta: f64) -> f64 {
        RDP_ORDERS
            .iter()
            .map(|&alpha| {
                steps as f64 * alpha / (2.0 * sigma * sigma) + (1.0 / delta).ln() / (alpha - 1.0)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Classical single-round Gaussian bound: sqrt(2·ln(1.25/δ))/σ.
    fn classical_round_epsilon(sigma: f64, delta: f64) -> f64 {
        (2.0 * (1.25 / delta).ln()).sqrt() / sigma
    }

    #[test]
    fn test_single_step_matches_analytic_grid_minimum() {
        let mut acc = RdpAccountant::new(SIGMA);
        acc.step();
        assert_eq!(acc.steps(), 1);

        let eps = acc.epsilon(DELTA);
        let expected = expected_epsilon(SIGMA, 1, DELTA);
        assert!(
            (eps - expected).abs() < 1e-9,
            "ε = {eps}, expected grid minimum = {expected}"
        );
        // Plausible band for σ=1, δ=1e-5 (analytic minimum is ≈ 5.30 at α=6).
        assert!(
            (4.0..6.0).contains(&eps),
            "ε = {eps} outside plausible band"
        );
    }

    #[test]
    fn test_composition_beats_naive_linear_summation() {
        let mut acc = RdpAccountant::new(SIGMA);
        for _ in 0..100 {
            acc.step();
        }
        assert_eq!(acc.steps(), 100);

        let rdp_eps = acc.epsilon(DELTA);
        let naive_eps = 100.0 * classical_round_epsilon(SIGMA, DELTA);
        assert!(
            rdp_eps < naive_eps,
            "RDP ε over 100 steps ({rdp_eps}) must be strictly smaller than \
             100× the classical single-round bound ({naive_eps})"
        );
    }

    #[test]
    fn test_epsilon_monotonically_increases_with_steps() {
        let mut acc = RdpAccountant::new(SIGMA);
        let mut prev = acc.epsilon(DELTA); // 0 at zero steps
        for step in 1..=200 {
            acc.step();
            let eps = acc.epsilon(DELTA);
            assert!(
                eps > prev,
                "ε must strictly increase with steps: step {step}: {eps} <= {prev}"
            );
            prev = eps;
        }
    }

    #[test]
    fn test_smaller_delta_gives_larger_epsilon() {
        let mut acc = RdpAccountant::new(SIGMA);
        for _ in 0..10 {
            acc.step();
        }
        let eps_tight = acc.epsilon(1e-9);
        let eps_loose = acc.epsilon(1e-3);
        assert!(
            eps_tight > eps_loose,
            "smaller δ must cost more ε: ε(1e-9) = {eps_tight}, ε(1e-3) = {eps_loose}"
        );
    }

    #[test]
    fn test_zero_steps_spends_no_epsilon() {
        let acc = RdpAccountant::new(SIGMA);
        assert_eq!(acc.steps(), 0);
        assert_eq!(acc.epsilon(DELTA), 0.0);
    }

    #[test]
    fn test_nonpositive_noise_multiplier_refused() {
        for sigma in [0.0, -1.0] {
            let mut acc = RdpAccountant::new(sigma);
            acc.step();
            assert!(
                acc.epsilon(DELTA).is_infinite(),
                "σ = {sigma} must yield infinite ε"
            );
        }
    }

    #[test]
    fn test_invalid_delta_refused() {
        let mut acc = RdpAccountant::new(SIGMA);
        acc.step();
        for delta in [0.0, -1e-5, 1.0, 2.0] {
            assert!(
                acc.epsilon(delta).is_infinite(),
                "δ = {delta} must yield infinite ε"
            );
        }
    }
}
