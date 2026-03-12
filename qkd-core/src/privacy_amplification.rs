//! Privacy amplification via universal hashing.
//!
//! Reduces Eve's information to negligible levels by hashing the
//! error-corrected key to a shorter length. The compression ratio
//! depends on the estimated QBER (higher QBER → more compression needed).

use crate::error::{QkdError, QkdResult};
use sha2::{Digest, Sha256};
use tracing::debug;

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
        let bits: Vec<u8> = vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                                  0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                                  1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0];
        let r1 = amplify(&bits, 0.03).unwrap();
        let r2 = amplify(&bits, 0.03).unwrap();
        assert_eq!(r1, r2);
    }
}
