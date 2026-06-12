//! Privacy amplification via universal hashing.
//!
//! Reduces Eve's information to negligible levels by hashing the
//! error-corrected key to a shorter length. The compression ratio
//! depends on the estimated QBER (higher QBER → more compression needed)
//! and on the number of parity bits leaked during error correction.
//!
//! The protocol pipeline uses [`toeplitz_amplify`]: a random (m × n)
//! Toeplitz matrix over GF(2), derived from a shared public seed, compresses
//! the n-bit reconciled key to m output bits, where m subtracts the CASCADE
//! leakage and a safety margin. Toeplitz matrices form a universal₂ hash
//! family, which is what the leftover hash lemma requires; the seed may be
//! exchanged publicly as long as it is random and chosen independently of
//! Eve's information.
//!
//! The legacy SHA-256 counter construction ([`amplify`]) is retained for
//! backward compatibility.

use crate::error::{QkdError, QkdResult};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use tracing::debug;

/// Extra bits removed beyond the rate bound and measured leakage.
pub const SAFETY_MARGIN_BITS: usize = 32;

/// Privacy amplification with a seeded Toeplitz universal hash.
///
/// `corrected_bits` is the n-bit reconciled key (one bit per byte, values
/// 0/1). The output length is
/// `m = floor(n * secret_key_rate) - leaked_bits - SAFETY_MARGIN_BITS`
/// (rounded down to whole bytes), where `secret_key_rate` is the same
/// heuristic used by [`amplify`] and `leaked_bits` is the number of parity
/// bits revealed during CASCADE. Errors if nothing remains after leakage.
///
/// Both parties must call this with identical bits and the same `seed`
/// (exchanged over the authenticated classical channel) to derive identical
/// keys. The (m × n) Toeplitz matrix is defined by n+m-1 bits drawn from a
/// ChaCha20 stream seeded with `seed`; output bit i is the GF(2) inner
/// product of matrix row i with the key.
pub fn toeplitz_amplify(
    corrected_bits: &[u8],
    qber: f64,
    leaked_bits: usize,
    seed: &[u8; 32],
) -> QkdResult<Vec<u8>> {
    let n = corrected_bits.len();
    if n == 0 {
        return Err(QkdError::PrivacyAmplification("Empty input".into()));
    }

    let secure_fraction = secret_key_rate(qber)?;
    let rate_bits = (n as f64 * secure_fraction).floor() as usize;
    let m = rate_bits
        .saturating_sub(leaked_bits)
        .saturating_sub(SAFETY_MARGIN_BITS);
    // Whole bytes only.
    let output_bytes = m / 8;
    if output_bytes == 0 {
        return Err(QkdError::PrivacyAmplification(format!(
            "No secure bits remain: rate yields {rate_bits} bits, \
             error correction leaked {leaked_bits}, margin {SAFETY_MARGIN_BITS}"
        )));
    }
    let m = output_bytes * 8;

    debug!(
        input_bits = n,
        output_bits = m,
        leaked_bits,
        secure_fraction = format!("{secure_fraction:.4}"),
        "Toeplitz privacy amplification"
    );

    // Diagonal-constant matrix entries: T[i][j] = t[i + (n - 1) - j],
    // with t holding n + m - 1 random bits.
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let diag: Vec<u8> = (0..n + m - 1).map(|_| rng.gen::<bool>() as u8).collect();

    let mut output = vec![0u8; output_bytes];
    for (i, out_byte) in output.iter_mut().enumerate() {
        for bit_in_byte in 0..8 {
            let row = i * 8 + bit_in_byte;
            let mut acc = 0u8;
            for (j, &key_bit) in corrected_bits.iter().enumerate() {
                acc ^= diag[row + (n - 1) - j] & key_bit & 1;
            }
            *out_byte |= acc << (7 - bit_in_byte);
        }
    }

    Ok(output)
}

/// Secret-key-rate heuristic shared by both amplification constructions:
/// r = 1 - h(QBER) - f·h(QBER), with f ≈ 1.16 (CASCADE efficiency).
fn secret_key_rate(qber: f64) -> QkdResult<f64> {
    let h_qber = binary_entropy(qber);
    let f_ec = 1.16;
    let secure_fraction = 1.0 - h_qber - f_ec * h_qber;
    if secure_fraction <= 0.0 {
        return Err(QkdError::PrivacyAmplification(format!(
            "QBER {qber:.4} too high for secure key extraction (secure fraction ≤ 0)"
        )));
    }
    Ok(secure_fraction)
}

/// Compress a corrected key using privacy amplification.
///
/// Returns the final secure key bytes. The output length is determined
/// by the secret key rate formula: r = 1 - h(QBER) - f*h(QBER)
/// where h is the binary entropy function and f is the error correction
/// efficiency (typically ~1.16 for CASCADE).
pub fn amplify(corrected_bits: &[u8], qber: f64) -> QkdResult<Vec<u8>> {
    let n = corrected_bits.len();
    if n == 0 {
        return Err(QkdError::PrivacyAmplification("Empty input".into()));
    }

    // Calculate secure key fraction using secret key rate bound
    let h_qber = binary_entropy(qber);
    let f_ec = 1.16; // CASCADE efficiency factor
    let secure_fraction = (1.0 - h_qber - f_ec * h_qber).max(0.0);

    if secure_fraction <= 0.0 {
        return Err(QkdError::PrivacyAmplification(format!(
            "QBER {qber:.4} too high for secure key extraction (secure fraction ≤ 0)"
        )));
    }

    let output_bits = (n as f64 * secure_fraction).floor() as usize;
    let output_bytes = output_bits / 8;

    if output_bytes == 0 {
        return Err(QkdError::PrivacyAmplification(
            "Insufficient bits for secure key after amplification".into(),
        ));
    }

    debug!(
        input_bits = n,
        output_bits,
        secure_fraction = format!("{secure_fraction:.4}"),
        "Privacy amplification"
    );

    // Universal hash using SHA-256 in a counter mode construction
    // This is a simplified version; production would use Toeplitz matrix hashing
    let mut output = Vec::with_capacity(output_bytes);
    let mut counter: u64 = 0;

    while output.len() < output_bytes {
        let mut hasher = Sha256::new();
        hasher.update(corrected_bits);
        hasher.update(counter.to_le_bytes());
        let hash = hasher.finalize();

        let remaining = output_bytes - output.len();
        let take = remaining.min(32); // SHA-256 = 32 bytes
        output.extend_from_slice(&hash[..take]);
        counter += 1;
    }

    output.truncate(output_bytes);
    Ok(output)
}

/// Binary entropy function: h(p) = -p*log2(p) - (1-p)*log2(1-p)
fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Toeplitz universal hashing (Workstream 5) ----

    fn random_bits(n: usize, seed: u64) -> Vec<u8> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(0u8..2)).collect()
    }

    #[test]
    fn test_toeplitz_deterministic_given_seed() {
        let bits = random_bits(1024, 1);
        let seed = [7u8; 32];
        let a = toeplitz_amplify(&bits, 0.03, 100, &seed).unwrap();
        let b = toeplitz_amplify(&bits, 0.03, 100, &seed).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn test_toeplitz_same_seed_both_parties_identical() {
        // Alice and Bob hold identical reconciled bits and the shared seed;
        // independent invocations must produce the same final key.
        let alice_bits = random_bits(2048, 2);
        let bob_bits = alice_bits.clone();
        let seed = [42u8; 32];
        let alice_key = toeplitz_amplify(&alice_bits, 0.02, 250, &seed).unwrap();
        let bob_key = toeplitz_amplify(&bob_bits, 0.02, 250, &seed).unwrap();
        assert_eq!(alice_key, bob_key);
    }

    #[test]
    fn test_toeplitz_different_seed_differs() {
        let bits = random_bits(1024, 3);
        let a = toeplitz_amplify(&bits, 0.03, 100, &[1u8; 32]).unwrap();
        let b = toeplitz_amplify(&bits, 0.03, 100, &[2u8; 32]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_toeplitz_output_shrinks_with_leakage() {
        let bits = random_bits(2048, 4);
        let seed = [9u8; 32];
        let none = toeplitz_amplify(&bits, 0.03, 0, &seed).unwrap();
        let some = toeplitz_amplify(&bits, 0.03, 400, &seed).unwrap();
        let more = toeplitz_amplify(&bits, 0.03, 800, &seed).unwrap();
        assert!(none.len() > some.len());
        assert!(some.len() > more.len());
    }

    #[test]
    fn test_toeplitz_errors_when_leakage_consumes_key() {
        let bits = random_bits(256, 5);
        // Leakage exceeds anything the key could yield.
        let result = toeplitz_amplify(&bits, 0.03, 10_000, &[0u8; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_toeplitz_single_bit_flip_diffuses() {
        // Flipping one input bit must flip ~half the output bits on average
        // (each output bit depends on input bit j via a fresh random matrix
        // entry). Note Toeplitz·0 = 0, so this is the right diffusion test.
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99);

        let n = 512;
        let trials = 40;
        let mut flip_fractions = Vec::new();
        for _ in 0..trials {
            let bits = random_bits(n, rng.gen());
            let seed: [u8; 32] = rng.gen();
            let base = toeplitz_amplify(&bits, 0.02, 0, &seed).unwrap();

            let mut flipped = bits.clone();
            let pos = rng.gen_range(0..n);
            flipped[pos] ^= 1;
            let other = toeplitz_amplify(&flipped, 0.02, 0, &seed).unwrap();

            assert_eq!(base.len(), other.len());
            let total_bits = base.len() * 8;
            let differing: u32 = base
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            flip_fractions.push(differing as f64 / total_bits as f64);
        }
        let mean = flip_fractions.iter().sum::<f64>() / trials as f64;
        assert!(
            (mean - 0.5).abs() < 0.1,
            "mean output-bit flip fraction {mean:.3} not ~0.5"
        );
    }

    #[test]
    fn test_binary_entropy() {
        assert!((binary_entropy(0.0) - 0.0).abs() < 1e-10);
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-10);
        assert!((binary_entropy(1.0) - 0.0).abs() < 1e-10);
        assert!((binary_entropy(0.11) - 0.5) < 0.1);
    }

    #[test]
    fn test_amplify_low_qber() {
        let bits: Vec<u8> = (0..1000).map(|i| (i % 2) as u8).collect();
        let result = amplify(&bits, 0.02).unwrap();
        assert!(!result.is_empty());
        // With low QBER, output should be a substantial fraction of input
        assert!(result.len() * 8 > 500);
    }

    #[test]
    fn test_amplify_high_qber_fails() {
        let bits: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();
        // QBER of 0.5 → no secure bits possible
        let result = amplify(&bits, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_amplify_deterministic() {
        let bits: Vec<u8> = vec![
            0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
            1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
        ];
        let r1 = amplify(&bits, 0.03).unwrap();
        let r2 = amplify(&bits, 0.03).unwrap();
        assert_eq!(r1, r2);
    }
}
