//! Six-State QKD protocol.
//!
//! Extension of BB84 using three mutually unbiased bases (rectilinear, diagonal, circular).
//! Provides better eavesdropping detection (lower QBER threshold ~12.6% vs BB84's ~11%)
//! and higher noise tolerance.

use crate::channel::{ChannelConfig, QuantumChannel};
use crate::error::{QkdError, QkdResult};
use crate::error_correction::cascade;
use crate::privacy_amplification;
use crate::protocols::QkdProtocol;
use crate::types::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tracing::{debug, info, warn};

/// Six-State protocol configuration
#[derive(Debug, Clone)]
pub struct SixState {
    pub sample_fraction: f64,
    /// Theoretical QBER threshold for Six-State: ~12.6%
    pub qber_threshold: f64,
    pub min_key_bits: usize,
    pub seed: Option<u64>,
}

impl Default for SixState {
    fn default() -> Self {
        Self {
            sample_fraction: 0.1,
            qber_threshold: 0.126,
            min_key_bits: 256,
            seed: None,
        }
    }
}

impl SixState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn random_basis<R: Rng>(rng: &mut R) -> Basis {
        match rng.gen_range(0u8..3) {
            0 => Basis::Rectilinear,
            1 => Basis::Diagonal,
            _ => Basis::Circular,
        }
    }

    fn alice_prepare<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<Qubit> {
        (0..n)
            .map(|_| Qubit {
                basis: Self::random_basis(rng),
                value: if rng.gen::<bool>() {
                    QubitValue::One
                } else {
                    QubitValue::Zero
                },
            })
            .collect()
    }

    fn bob_measure<R: Rng>(
        &self,
        received: &[Option<Qubit>],
        rng: &mut R,
    ) -> Vec<Option<(Basis, QubitValue)>> {
        received
            .iter()
            .map(|q| {
                q.map(|qubit| {
                    let bob_basis = Self::random_basis(rng);
                    let value = if bob_basis == qubit.basis {
                        qubit.value
                    } else {
                        if rng.gen::<bool>() {
                            QubitValue::One
                        } else {
                            QubitValue::Zero
                        }
                    };
                    (bob_basis, value)
                })
            })
            .collect()
    }

    fn sift(
        &self,
        alice_qubits: &[Qubit],
        bob_measurements: &[Option<(Basis, QubitValue)>],
    ) -> (Vec<u8>, Vec<u8>) {
        let mut alice_bits = Vec::new();
        let mut bob_bits = Vec::new();

        for (aq, bm) in alice_qubits.iter().zip(bob_measurements.iter()) {
            if let Some((bob_basis, bob_value)) = bm {
                if aq.basis == *bob_basis {
                    alice_bits.push(aq.value.as_bit());
                    bob_bits.push(bob_value.as_bit());
                }
            }
        }

        (alice_bits, bob_bits)
    }

    fn estimate_qber<R: Rng>(
        &self,
        alice_bits: &[u8],
        bob_bits: &[u8],
        rng: &mut R,
    ) -> (f64, Vec<u8>, Vec<u8>) {
        let n = alice_bits.len();
        let sample_size = (n as f64 * self.sample_fraction).ceil() as usize;

        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..sample_size.min(n) {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
        }

        let sample_indices: std::collections::HashSet<usize> =
            indices[..sample_size].iter().copied().collect();

        let errors: usize = sample_indices
            .iter()
            .filter(|&&i| alice_bits[i] != bob_bits[i])
            .count();
        let qber = if sample_size > 0 {
            errors as f64 / sample_size as f64
        } else {
            0.0
        };

        let remaining_alice: Vec<u8> = alice_bits
            .iter()
            .enumerate()
            .filter(|(i, _)| !sample_indices.contains(i))
            .map(|(_, &b)| b)
            .collect();
        let remaining_bob: Vec<u8> = bob_bits
            .iter()
            .enumerate()
            .filter(|(i, _)| !sample_indices.contains(i))
            .map(|(_, &b)| b)
            .collect();

        (qber, remaining_alice, remaining_bob)
    }
}

impl QkdProtocol for SixState {
    fn execute(
        &self,
        num_qubits: usize,
        channel_config: &ChannelConfig,
    ) -> QkdResult<(SecureKey, QkdStats)> {
        let mut rng = match self.seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };

        let channel = QuantumChannel::new(channel_config.clone());
        info!(protocol = "Six-State", qubits = num_qubits, "Starting key exchange");

        let alice_qubits = self.alice_prepare(num_qubits, &mut rng);
        let received: Vec<Option<Qubit>> = alice_qubits
            .iter()
            .map(|q| channel.transmit(q, &mut rng))
            .collect();
        let received_count = received.iter().filter(|q| q.is_some()).count();

        let bob_measurements = self.bob_measure(&received, &mut rng);
        let (alice_sifted, bob_sifted) = self.sift(&alice_qubits, &bob_measurements);
        let sifted_len = alice_sifted.len();
        debug!(sifted_bits = sifted_len, "Sifting complete");

        // Six-State sifting rate is ~1/3 (vs ~1/2 for BB84)
        let (qber, alice_remaining, bob_remaining) =
            self.estimate_qber(&alice_sifted, &bob_sifted, &mut rng);

        info!(qber = format!("{qber:.4}"), "QBER estimated");

        let eavesdropping_detected = qber > self.qber_threshold;
        if eavesdropping_detected {
            warn!(qber = format!("{qber:.4}"), "Eavesdropping detected");
            return Err(QkdError::EavesdroppingDetected {
                qber,
                threshold: self.qber_threshold,
            });
        }

        let corrected = cascade::correct(&alice_remaining, &bob_remaining, qber)?;
        let final_key_bits = privacy_amplification::amplify(&corrected, qber)?;

        if final_key_bits.len() < self.min_key_bits / 8 {
            return Err(QkdError::InsufficientKeyBits {
                got: final_key_bits.len() * 8,
                need: self.min_key_bits,
            });
        }

        let key = SecureKey {
            key_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            material: final_key_bits,
            length_bits: self.min_key_bits,
        };

        let stats = QkdStats {
            protocol: ProtocolType::SixState,
            qubits_sent: num_qubits,
            qubits_received: received_count,
            sifting_rate: sifted_len as f64 / num_qubits as f64,
            qber,
            raw_key_bits: alice_remaining.len(),
            final_key_bits: key.material.len() * 8,
            key_rate: (key.material.len() * 8) as f64 / num_qubits as f64,
            eavesdropping_detected,
        };

        info!(final_key_bits = stats.final_key_bits, "Six-State key exchange successful");
        Ok((key, stats))
    }

    fn qber_threshold(&self) -> f64 {
        self.qber_threshold
    }

    fn name(&self) -> &'static str {
        "Six-State"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_six_state_clean_channel() {
        let proto = SixState::new().with_seed(42);
        let channel = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.05,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };

        let (key, stats) = proto.execute(15_000, &channel).unwrap();
        assert!(!key.material.is_empty());
        assert!(stats.qber < 0.05);
        // Six-State sifting rate should be ~33%
        assert!(stats.sifting_rate > 0.28 && stats.sifting_rate < 0.40,
            "sifting rate: {}", stats.sifting_rate);
    }
}
