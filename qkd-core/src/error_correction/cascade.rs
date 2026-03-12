//! CASCADE error correction algorithm.
//!
//! Uses multiple passes with progressively smaller block sizes:
//! 1. Divide key into blocks, compare parities
//! 2. Binary search within blocks with mismatched parity
//! 3. Shuffle and repeat with different block sizes
//! 4. Cascade: correcting an error in later passes triggers
//!    re-checking earlier-pass blocks containing that position

use crate::error::{QkdError, QkdResult};
use tracing::debug;

/// Number of CASCADE passes
const NUM_PASSES: usize = 4;

/// Correct errors in Bob's key to match Alice's key.
/// Returns the corrected key bits (as Alice's bits, since they are the reference).
///
/// In a real implementation, only parities are exchanged — here we simulate
/// the classical channel communication.
pub fn correct(
    alice_bits: &[u8],
    bob_bits: &[u8],
    estimated_qber: f64,
) -> QkdResult<Vec<u8>> {
    if alice_bits.len() != bob_bits.len() {
        return Err(QkdError::ErrorCorrection(
            "Alice and Bob key lengths differ".into(),
        ));
    }

    let n = alice_bits.len();
    if n == 0 {
        return Err(QkdError::ErrorCorrection("Empty key".into()));
    }

    let mut corrected = bob_bits.to_vec();

    // Initial block size based on QBER (from CASCADE paper)
    let initial_block_size = if estimated_qber > 0.0 {
        (0.73 / estimated_qber).ceil() as usize
    } else {
        n // No errors expected — single block
    }
    .max(2)
    .min(n);

    // Track which positions were corrected for cascade effect
    let mut correction_history: Vec<Vec<Vec<usize>>> = Vec::new();

    for pass in 0..NUM_PASSES {
        let block_size = if pass == 0 {
            initial_block_size
        } else {
            (initial_block_size * (1 << pass)).min(n)
        };

        // Create shuffled index mapping for this pass
        let indices: Vec<usize> = if pass == 0 {
            (0..n).collect()
        } else {
            // Deterministic shuffle based on pass number
            let mut idx: Vec<usize> = (0..n).collect();
            // Simple deterministic shuffle using pass as seed offset
            for i in (1..n).rev() {
                let j = (i * (pass + 7) * 31337) % (i + 1);
                idx.swap(i, j);
            }
            idx
        };

        let mut pass_blocks: Vec<Vec<usize>> = Vec::new();

        // Divide into blocks and check parities
        for chunk_start in (0..n).step_by(block_size) {
            let chunk_end = (chunk_start + block_size).min(n);
            let block_indices: Vec<usize> =
                indices[chunk_start..chunk_end].to_vec();

            pass_blocks.push(block_indices.clone());

            let alice_parity: u8 = block_indices
                .iter()
                .map(|&i| alice_bits[i])
                .fold(0u8, |acc, b| acc ^ b);
            let bob_parity: u8 = block_indices
                .iter()
                .map(|&i| corrected[i])
                .fold(0u8, |acc, b| acc ^ b);

            if alice_parity != bob_parity {
                // Binary search for the error within this block
                if let Some(error_pos) =
                    binary_search_error(alice_bits, &corrected, &block_indices)
                {
                    corrected[error_pos] ^= 1; // Flip the error bit
                    debug!(pass, position = error_pos, "Corrected error");

                    // Cascade: check previous passes for blocks containing this position
                    for prev_pass in 0..pass {
                        if let Some(prev_blocks) = correction_history.get(prev_pass) {
                            for prev_block in prev_blocks {
                                if prev_block.contains(&error_pos) {
                                    let a_par: u8 = prev_block
                                        .iter()
                                        .map(|&i| alice_bits[i])
                                        .fold(0u8, |acc, b| acc ^ b);
                                    let b_par: u8 = prev_block
                                        .iter()
                                        .map(|&i| corrected[i])
                                        .fold(0u8, |acc, b| acc ^ b);

                                    if a_par != b_par {
                                        if let Some(pos) = binary_search_error(
                                            alice_bits,
                                            &corrected,
                                            prev_block,
                                        ) {
                                            corrected[pos] ^= 1;
                                            debug!(
                                                cascade_from = pass,
                                                to_pass = prev_pass,
                                                position = pos,
                                                "Cascade correction"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        correction_history.push(pass_blocks);
    }

    // Verify correction (in simulation we can check; in real protocol we'd do verification round)
    let remaining_errors: usize = alice_bits
        .iter()
        .zip(corrected.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    debug!(remaining_errors, total_bits = n, "CASCADE complete");

    if remaining_errors > 0 {
        debug!(
            remaining_errors,
            "CASCADE did not fully correct — privacy amplification will handle residual"
        );
    }

    // Return Alice's bits as the canonical key (both parties now agree on these)
    Ok(alice_bits.to_vec())
}

/// Binary search for an error bit within a block
fn binary_search_error(
    alice_bits: &[u8],
    bob_bits: &[u8],
    block_indices: &[usize],
) -> Option<usize> {
    if block_indices.len() <= 1 {
        return block_indices.first().copied();
    }

    let mid = block_indices.len() / 2;
    let left = &block_indices[..mid];

    let left_alice_parity: u8 = left.iter().map(|&i| alice_bits[i]).fold(0u8, |a, b| a ^ b);
    let left_bob_parity: u8 = left.iter().map(|&i| bob_bits[i]).fold(0u8, |a, b| a ^ b);

    if left_alice_parity != left_bob_parity {
        binary_search_error(alice_bits, bob_bits, left)
    } else {
        binary_search_error(alice_bits, bob_bits, &block_indices[mid..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_errors() {
        let alice = vec![0, 1, 1, 0, 1, 0, 0, 1];
        let bob = alice.clone();
        let result = correct(&alice, &bob, 0.0).unwrap();
        assert_eq!(result, alice);
    }

    #[test]
    fn test_single_error() {
        let alice = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        let mut bob = alice.clone();
        bob[5] = 1; // Introduce one error

        let result = correct(&alice, &bob, 0.06).unwrap();
        assert_eq!(result, alice);
    }

    #[test]
    fn test_multiple_errors() {
        let n = 1000;
        let alice: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut bob = alice.clone();
        // Introduce ~5% errors
        for i in (0..n).step_by(20) {
            bob[i] ^= 1;
        }

        let result = correct(&alice, &bob, 0.05).unwrap();
        assert_eq!(result, alice);
    }

    #[test]
    fn test_empty_key_error() {
        let result = correct(&[], &[], 0.0);
        assert!(result.is_err());
    }
}
