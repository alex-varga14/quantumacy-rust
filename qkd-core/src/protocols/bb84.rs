//! BB84 Quantum Key Distribution protocol.
//!
//! Implements the complete BB84 pipeline:
//! 1. Alice prepares qubits in random bases with random values
//! 2. Qubits transmitted through quantum channel (may have noise/Eve)
//! 3. Bob measures in random bases
//! 4. Basis reconciliation via classical channel (sifting)
//! 5. QBER estimation on a sample
//! 6. Error correction (CASCADE)
//! 7. Privacy amplification (universal hashing)

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

/// BB84 protocol configuration
#[derive(Debug, Clone)]
pub struct Bb84 {
    /// Fraction of sifted bits used for QBER estimation (sacrificed)
    pub sample_fraction: f64,
    /// QBER threshold — abort if exceeded (theoretical max for BB84: 11%)
    pub qber_threshold: f64,
    /// Minimum final key length in bits
    pub min_key_bits: usize,
    /// RNG seed (None = random)
    pub seed: Option<u64>,
}

impl Default for Bb84 {
    fn default() -> Self {
        Self {
            sample_fraction: 0.1,
            qber_threshold: 0.11,
            min_key_bits: 256,
            seed: None,
        }
    }
}

impl Bb84 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.qber_threshold = threshold;
        self
    }

    /// Step 1: Alice prepares qubits
    fn alice_prepare<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<Qubit> {
        (0..n)
            .map(|_| Qubit {
                basis: if rng.gen::<bool>() {
                    Basis::Rectilinear
                } else {
                    Basis::Diagonal
                },
                value: if rng.gen::<bool>() {
                    QubitValue::One
                } else {
                    QubitValue::Zero
                },
            })
            .collect()
    }

    /// Step 2-3: Bob measures received qubits in random bases
    fn bob_measure<R: Rng>(
        &self,
        received: &[Option<Qubit>],
        rng: &mut R,
    ) -> Vec<Option<(Basis, QubitValue)>> {
        received
            .iter()
            .map(|q| {
                q.map(|qubit| {
                    let bob_basis = if rng.gen::<bool>() {
                        Basis::Rectilinear
                    } else {
                        Basis::Diagonal
                    };

                    let value = if bob_basis == qubit.basis {
                        // Correct basis → deterministic outcome
                        qubit.value
                    } else {
                        // Wrong basis → random outcome
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

    /// Step 4: Basis sifting — keep only positions where both used same basis
    fn sift(
        &self,
        alice_qubits: &[Qubit],
        bob_measurements: &[Option<(Basis, QubitValue)>],
    ) -> (Vec<u8>, Vec<u8>) {
        let mut alice_bits = Vec::new();
        let mut bob_bits = Vec::new();

        for (alice_q, bob_m) in alice_qubits.iter().zip(bob_measurements.iter()) {
            if let Some((bob_basis, bob_value)) = bob_m {
                if alice_q.basis == *bob_basis {
                    alice_bits.push(alice_q.value.as_bit());
                    bob_bits.push(bob_value.as_bit());
                }
            }
        }

        (alice_bits, bob_bits)
    }

    /// Step 5: Estimate QBER from a random sample of sifted bits
    fn estimate_qber<R: Rng>(
        &self,
        alice_bits: &[u8],
        bob_bits: &[u8],
        rng: &mut R,
    ) -> (f64, Vec<u8>, Vec<u8>) {
        let n = alice_bits.len();
        let sample_size = (n as f64 * self.sample_fraction).ceil() as usize;

        // Choose random sample indices
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle for sample selection
        for i in 0..sample_size.min(n) {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
        }

        let sample_indices: std::collections::HashSet<usize> =
            indices[..sample_size].iter().copied().collect();

        // Count errors in sample
        let errors: usize = sample_indices
            .iter()
            .filter(|&&i| alice_bits[i] != bob_bits[i])
            .count();
        let qber = if sample_size > 0 {
            errors as f64 / sample_size as f64
        } else {
            0.0
        };

        // Remaining bits (non-sampled) become the key
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

impl QkdProtocol for Bb84 {
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
            protocol = "BB84",
            qubits = num_qubits,
            "Starting key exchange"
        );

        // Step 1: Alice prepares
        let alice_qubits = self.alice_prepare(num_qubits, &mut rng);
        debug!(count = alice_qubits.len(), "Alice prepared qubits");

        // Step 2: Transmit through quantum channel
        let received: Vec<Option<Qubit>> = alice_qubits
            .iter()
            .map(|q| channel.transmit(q, &mut rng))
            .collect();

        let received_count = received.iter().filter(|q| q.is_some()).count();
        debug!(
            sent = num_qubits,
            received = received_count,
            "Channel transmission complete"
        );

        // Step 3: Bob measures
        let bob_measurements = self.bob_measure(&received, &mut rng);

        // Step 4: Sifting
        let (alice_sifted, bob_sifted) = self.sift(&alice_qubits, &bob_measurements);
        let sifted_len = alice_sifted.len();
        debug!(sifted_bits = sifted_len, "Basis sifting complete");

        // Step 5: QBER estimation
        let (qber, alice_remaining, bob_remaining) =
            self.estimate_qber(&alice_sifted, &bob_sifted, &mut rng);

        info!(
            qber = format!("{qber:.4}"),
            threshold = format!("{:.4}", self.qber_threshold),
            "QBER estimation complete"
        );

        let eavesdropping_detected = qber > self.qber_threshold;
        if eavesdropping_detected {
            warn!(
                qber = format!("{qber:.4}"),
                "Eavesdropping detected — aborting"
            );
            return Err(QkdError::EavesdroppingDetected {
                qber,
                threshold: self.qber_threshold,
            });
        }

        // Step 6: Error correction (CASCADE) with parity-leakage accounting
        let reconciled = cascade::reconcile(&alice_remaining, &bob_remaining, qber)?;
        debug!(
            bits = reconciled.corrected.len(),
            leaked_bits = reconciled.leaked_bits,
            "Error correction complete"
        );

        // Step 7: Privacy amplification (Toeplitz hash, shared public seed),
        // subtracting the parity bits revealed during CASCADE.
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

        let sifting_rate = sifted_len as f64 / num_qubits as f64;
        let stats = QkdStats {
            protocol: ProtocolType::BB84,
            qubits_sent: num_qubits,
            qubits_received: received_count,
            sifting_rate,
            qber,
            raw_key_bits: alice_remaining.len(),
            final_key_bits: key.material.len() * 8,
            key_rate: (key.material.len() * 8) as f64 / num_qubits as f64,
            eavesdropping_detected,
        };

        info!(
            final_key_bits = stats.final_key_bits,
            key_rate = format!("{:.4}", stats.key_rate),
            "BB84 key exchange successful"
        );

        Ok((key, stats))
    }

    fn qber_threshold(&self) -> f64 {
        self.qber_threshold
    }

    fn name(&self) -> &'static str {
        "BB84"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bb84_clean_channel() {
        let bb84 = Bb84::new().with_seed(42);
        let channel = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.05,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };

        let (key, stats) = bb84.execute(10_000, &channel).unwrap();
        assert!(!key.material.is_empty());
        assert!(stats.qber < 0.05);
        assert!(!stats.eavesdropping_detected);
        assert!(stats.sifting_rate > 0.4 && stats.sifting_rate < 0.6);
    }

    #[test]
    fn test_bb84_detects_eavesdropper() {
        use crate::channel::{EavesdropperConfig, EveStrategy};

        let bb84 = Bb84::new().with_seed(99);
        let channel = ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: Some(EavesdropperConfig {
                intercept_rate: 1.0,
                strategy: EveStrategy::InterceptResend,
            }),
        };

        let result = bb84.execute(10_000, &channel);
        assert!(matches!(
            result,
            Err(QkdError::EavesdroppingDetected { .. })
        ));
    }

    #[test]
    fn test_bb84_sifting_rate() {
        let bb84 = Bb84::new().with_seed(7);
        let channel = ChannelConfig {
            noise_rate: 0.0,
            loss_rate: 0.0,
            dark_count_rate: 0.0,
            eavesdropper: None,
        };

        let (_, stats) = bb84.execute(50_000, &channel).unwrap();
        // BB84 sifting rate should be ~50%
        assert!(
            (stats.sifting_rate - 0.5).abs() < 0.02,
            "sifting rate: {}",
            stats.sifting_rate
        );
    }
}
