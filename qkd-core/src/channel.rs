//! Quantum channel simulation with configurable noise and eavesdropping models.
//!
//! Models realistic imperfections including:
//! - Depolarizing noise (uniform random errors)
//! - Detector dark counts
//! - Channel loss / photon absorption
//! - Intercept-resend eavesdropping (Eve)

use crate::types::{Basis, Qubit, QubitValue};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for quantum channel behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Intrinsic error rate from noise (typical: 0.01–0.03)
    pub noise_rate: f64,
    /// Probability of photon loss per qubit (typical: 0.1–0.3)
    pub loss_rate: f64,
    /// Dark count probability per detector (typical: 1e-5)
    pub dark_count_rate: f64,
    /// Whether an eavesdropper is present
    pub eavesdropper: Option<EavesdropperConfig>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            noise_rate: 0.02,
            loss_rate: 0.1,
            dark_count_rate: 1e-5,
            eavesdropper: None,
        }
    }
}

/// Eavesdropper (Eve) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EavesdropperConfig {
    /// Fraction of qubits Eve intercepts (0.0–1.0)
    pub intercept_rate: f64,
    /// Strategy used by Eve
    pub strategy: EveStrategy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EveStrategy {
    /// Measure in random basis and resend — introduces ~25% QBER on intercepted qubits
    InterceptResend,
    /// Optimal individual attack using Breidbart basis
    Breidbart,
}

/// Simulates a quantum channel between Alice and Bob
pub struct QuantumChannel {
    config: ChannelConfig,
}

impl QuantumChannel {
    pub fn new(config: ChannelConfig) -> Self {
        Self { config }
    }

    /// Transmit a qubit through the channel. Returns None if lost.
    pub fn transmit<R: Rng>(&self, qubit: &Qubit, rng: &mut R) -> Option<Qubit> {
        // Photon loss
        if rng.gen::<f64>() < self.config.loss_rate {
            return None;
        }

        let mut transmitted = *qubit;

        // Eavesdropper intercept-resend
        if let Some(ref eve) = self.config.eavesdropper {
            if rng.gen::<f64>() < eve.intercept_rate {
                transmitted = self.eavesdrop(&transmitted, eve, rng);
            }
        }

        // Channel noise (depolarizing)
        if rng.gen::<f64>() < self.config.noise_rate {
            transmitted.value = if rng.gen::<bool>() {
                QubitValue::One
            } else {
                QubitValue::Zero
            };
        }

        // Dark count (flip with very low probability)
        if rng.gen::<f64>() < self.config.dark_count_rate {
            transmitted.value = match transmitted.value {
                QubitValue::Zero => QubitValue::One,
                QubitValue::One => QubitValue::Zero,
            };
        }

        Some(transmitted)
    }

    /// Simulate Eve's intercept-resend attack
    fn eavesdrop<R: Rng>(
        &self,
        qubit: &Qubit,
        eve: &EavesdropperConfig,
        rng: &mut R,
    ) -> Qubit {
        match eve.strategy {
            EveStrategy::InterceptResend => {
                // Eve picks a random basis to measure
                let eve_basis = if rng.gen::<bool>() {
                    Basis::Rectilinear
                } else {
                    Basis::Diagonal
                };

                if eve_basis == qubit.basis {
                    // Correct basis → no disturbance
                    *qubit
                } else {
                    // Wrong basis → random outcome, 50% error
                    Qubit {
                        basis: qubit.basis,
                        value: if rng.gen::<bool>() {
                            QubitValue::Zero
                        } else {
                            QubitValue::One
                        },
                    }
                }
            }
            EveStrategy::Breidbart => {
                // Breidbart basis: measure at 22.5° — optimal individual attack
                // Introduces ~14.6% QBER (vs ~25% for intercept-resend)
                let error_prob = 0.146;
                if rng.gen::<f64>() < error_prob {
                    Qubit {
                        basis: qubit.basis,
                        value: match qubit.value {
                            QubitValue::Zero => QubitValue::One,
                            QubitValue::One => QubitValue::Zero,
                        },
                    }
                } else {
                    *qubit
                }
            }
        }
    }

    /// Calculate theoretical QBER for the current configuration
    pub fn theoretical_qber(&self) -> f64 {
        let mut qber = self.config.noise_rate;

        if let Some(ref eve) = self.config.eavesdropper {
            let eve_error = match eve.strategy {
                EveStrategy::InterceptResend => 0.25 * eve.intercept_rate,
                EveStrategy::Breidbart => 0.146 * eve.intercept_rate,
            };
            // Combine noise and eavesdropping (approximate for low rates)
            qber = qber + eve_error - qber * eve_error;
        }

        qber
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_lossless_noiseless_channel() {
        let config = ChannelConfig {
            noise_rate: 0.0,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };
        let channel = QuantumChannel::new(config);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let qubit = Qubit {
            basis: Basis::Rectilinear,
            value: QubitValue::One,
        };

        let result = channel.transmit(&qubit, &mut rng).unwrap();
        assert_eq!(result.value, QubitValue::One);
        assert_eq!(result.basis, Basis::Rectilinear);
    }

    #[test]
    fn test_full_loss_channel() {
        let config = ChannelConfig {
            noise_rate: 0.0,
            loss_rate: 1.0,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };
        let channel = QuantumChannel::new(config);
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let qubit = Qubit {
            basis: Basis::Diagonal,
            value: QubitValue::Zero,
        };

        assert!(channel.transmit(&qubit, &mut rng).is_none());
    }

    #[test]
    fn test_eavesdropper_increases_qber() {
        let config_no_eve = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };

        let config_with_eve = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: Some(EavesdropperConfig {
                intercept_rate: 1.0,
                strategy: EveStrategy::InterceptResend,
            }),
        };

        let ch_clean = QuantumChannel::new(config_no_eve);
        let ch_eve = QuantumChannel::new(config_with_eve);

        assert!(ch_eve.theoretical_qber() > ch_clean.theoretical_qber());
        // Intercept-resend on all qubits → ~25% QBER
        assert!((ch_eve.theoretical_qber() - 0.2575).abs() < 0.01);
    }

    #[test]
    fn test_statistical_noise_rate() {
        let config = ChannelConfig {
            noise_rate: 0.1,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };
        let channel = QuantumChannel::new(config);
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let qubit = Qubit {
            basis: Basis::Rectilinear,
            value: QubitValue::Zero,
        };

        let n = 10_000;
        let errors: usize = (0..n)
            .filter_map(|_| channel.transmit(&qubit, &mut rng))
            .filter(|q| q.value != QubitValue::Zero)
            .count();

        let measured_rate = errors as f64 / n as f64;
        // Should be roughly 5% (noise flips to random, so ~50% of noise_rate)
        assert!(measured_rate > 0.03 && measured_rate < 0.08,
            "measured noise flip rate: {measured_rate}");
    }
}
