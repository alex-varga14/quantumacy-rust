//! B92 QKD protocol.
//!
//! Uses only two non-orthogonal states (one from each basis).
//! Simpler than BB84 but lower key rate and more susceptible to loss.
//! Alice sends |0⟩ for bit 0 and |+⟩ for bit 1.
//! Bob randomly measures in rectilinear or diagonal basis.
//! A conclusive result occurs only when Bob's measurement is
//! incompatible with the "wrong" state.

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

#[derive(Debug, Clone)]
pub struct B92 {
    pub sample_fraction: f64,
    pub qber_threshold: f64,
    pub min_key_bits: usize,
    pub seed: Option<u64>,
}

impl Default for B92 {
    fn default() -> Self {
        Self {
            sample_fraction: 0.15,
            qber_threshold: 0.11,
            min_key_bits: 256,
            seed: None,
        }
    }
}

impl B92 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Alice encodes: bit 0 → |0⟩ (rectilinear), bit 1 → |+⟩ (diagonal)
    fn alice_prepare<R: Rng>(&self, n: usize, rng: &mut R) -> (Vec<u8>, Vec<Qubit>) {
        let bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0u8..2)).collect();
        let qubits: Vec<Qubit> = bits
            .iter()
            .map(|&b| {
                if b == 0 {
                    Qubit {
                        basis: Basis::Rectilinear,
                        value: QubitValue::Zero,
                    }
                } else {
                    Qubit {
                        basis: Basis::Diagonal,
                        value: QubitValue::Zero, // |+⟩ state
                    }
                }
            })
            .collect();
        (bits, qubits)
    }

    /// Bob measures and reports conclusive/inconclusive results.
    /// Conclusive: Bob measures |1⟩ in rectilinear (means Alice sent |+⟩, bit=1)
    ///             or |−⟩ in diagonal (means Alice sent |0⟩, bit=0)
    fn bob_measure<R: Rng>(&self, received: &[Option<Qubit>], rng: &mut R) -> Vec<Option<u8>> {
        received
            .iter()
            .map(|q| {
                match q {
                    None => None, // Lost photon
                    Some(qubit) => {
                        // Bob randomly picks a basis
                        let bob_basis = if rng.gen::<bool>() {
                            Basis::Rectilinear
                        } else {
                            Basis::Diagonal
                        };

                        // Simulate measurement
                        if bob_basis == qubit.basis {
                            // Same basis → result consistent with Alice's state → inconclusive
                            None
                        } else {
                            // Different basis → 50% chance of conclusive result
                            if rng.gen::<bool>() {
                                // Conclusive detection
                                if bob_basis == Basis::Rectilinear {
                                    Some(1) // Detected |1⟩ → Alice sent |+⟩ → bit 1
                                } else {
                                    Some(0) // Detected |−⟩ → Alice sent |0⟩ → bit 0
                                }
                            } else {
                                None // No detection
                            }
                        }
                    }
                }
            })
            .collect()
    }

    /// Sift: keep only conclusive detections
    fn sift(&self, alice_bits: &[u8], bob_results: &[Option<u8>]) -> (Vec<u8>, Vec<u8>) {
        let mut a_sifted = Vec::new();
        let mut b_sifted = Vec::new();

        for (&a_bit, b_result) in alice_bits.iter().zip(bob_results.iter()) {
            if let Some(b_bit) = b_result {
                a_sifted.push(a_bit);
                b_sifted.push(*b_bit);
            }
        }

        (a_sifted, b_sifted)
    }
}

impl QkdProtocol for B92 {
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
        info!(
            protocol = "B92",
            qubits = num_qubits,
            "Starting key exchange"
        );

        let (alice_bits, alice_qubits) = self.alice_prepare(num_qubits, &mut rng);

        let received: Vec<Option<Qubit>> = alice_qubits
            .iter()
            .map(|q| channel.transmit(q, &mut rng))
            .collect();
        let received_count = received.iter().filter(|q| q.is_some()).count();

        let bob_results = self.bob_measure(&received, &mut rng);
        let (alice_sifted, bob_sifted) = self.sift(&alice_bits, &bob_results);
        let sifted_len = alice_sifted.len();
        debug!(sifted_bits = sifted_len, "B92 sifting complete");

        // QBER estimation
        let sample_size = (sifted_len as f64 * self.sample_fraction).ceil() as usize;
        let mut indices: Vec<usize> = (0..sifted_len).collect();
        for i in 0..sample_size.min(sifted_len) {
            let j = rng.gen_range(i..sifted_len);
            indices.swap(i, j);
        }
        let sample: std::collections::HashSet<usize> =
            indices[..sample_size].iter().copied().collect();

        let errors: usize = sample
            .iter()
            .filter(|&&i| alice_sifted[i] != bob_sifted[i])
            .count();
        let qber = if sample_size > 0 {
            errors as f64 / sample_size as f64
        } else {
            0.0
        };

        info!(qber = format!("{qber:.4}"), "B92 QBER estimated");

        if qber > self.qber_threshold {
            warn!(qber = format!("{qber:.4}"), "Eavesdropping detected");
            return Err(QkdError::EavesdroppingDetected {
                qber,
                threshold: self.qber_threshold,
            });
        }

        let alice_remaining: Vec<u8> = alice_sifted
            .iter()
            .enumerate()
            .filter(|(i, _)| !sample.contains(i))
            .map(|(_, &b)| b)
            .collect();
        let bob_remaining: Vec<u8> = bob_sifted
            .iter()
            .enumerate()
            .filter(|(i, _)| !sample.contains(i))
            .map(|(_, &b)| b)
            .collect();

        let reconciled = cascade::reconcile(&alice_remaining, &bob_remaining, qber)?;
        let pa_seed: [u8; 32] = rng.gen();
        let final_key_bits = privacy_amplification::toeplitz_amplify(
            &reconciled.corrected,
            qber,
            reconciled.leaked_bits,
            &pa_seed,
        )?;

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
            protocol: ProtocolType::B92,
            qubits_sent: num_qubits,
            qubits_received: received_count,
            sifting_rate: sifted_len as f64 / num_qubits as f64,
            qber,
            raw_key_bits: alice_remaining.len(),
            final_key_bits: key.material.len() * 8,
            key_rate: (key.material.len() * 8) as f64 / num_qubits as f64,
            eavesdropping_detected: false,
        };

        info!(
            final_key_bits = stats.final_key_bits,
            "B92 key exchange successful"
        );
        Ok((key, stats))
    }

    fn qber_threshold(&self) -> f64 {
        self.qber_threshold
    }

    fn name(&self) -> &'static str {
        "B92"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b92_clean_channel() {
        let proto = B92::new().with_seed(42);
        let channel = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.05,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };

        // B92 has lower sifting rate, need more qubits
        let (key, stats) = proto.execute(20_000, &channel).unwrap();
        assert!(!key.material.is_empty());
        assert!(stats.qber < 0.05);
        // B92 sifting rate ~25%
        assert!(
            stats.sifting_rate > 0.15 && stats.sifting_rate < 0.35,
            "sifting rate: {}",
            stats.sifting_rate
        );
    }
}
