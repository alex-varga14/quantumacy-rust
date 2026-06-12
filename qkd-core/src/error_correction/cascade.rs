//! CASCADE error correction algorithm — true two-party reconciliation.
//!
//! Uses multiple passes with progressively larger block sizes:
//! 1. Divide key into blocks, compare parities
//! 2. Binary search within blocks with mismatched parity (BINARY protocol)
//! 3. Shuffle and repeat with different block sizes
//! 4. Cascade: correcting an error in later passes triggers
//!    re-checking earlier-pass blocks containing that position
//!
//! The protocol is message-driven: [`BobCascade`] (the party correcting its
//! bits) emits [`CascadeMessage`]s, [`AliceCascade`] (the reference party)
//! answers them. The state machines never see each other's bits, so they can
//! run in separate processes connected by a network transport. Every parity
//! Alice reveals is information leaked to an eavesdropper on the classical
//! channel and is counted (one parity response = one leaked bit); the count
//! must be subtracted from the privacy-amplification output length.
//!
//! [`reconcile`] is a convenience wrapper that drives both state machines in
//! a loop for in-process use (single-process protocol simulations).

use crate::error::{QkdError, QkdResult};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use tracing::debug;

/// Number of CASCADE passes
const NUM_PASSES: usize = 4;

/// Default seed for the deterministic block-shuffle schedule. The shuffle is
/// public information (block assignments leak nothing about the bits), so a
/// fixed default is fine; networked runs may negotiate any value.
pub const CASCADE_SHUFFLE_SEED: u64 = 0x5ca1ab1e;

/// Upper bound on extra correction passes after a failed hash verification.
const MAX_EXTRA_PASSES: usize = 16;

/// Messages exchanged between the two CASCADE parties over the (authenticated)
/// classical channel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CascadeMessage {
    /// Bob → Alice: request the parity of the bits at `indices`.
    ParityRequest { indices: Vec<usize> },
    /// Alice → Bob: parity answer. Each response leaks exactly one bit.
    ParityResponse { parity: u8 },
    /// Bob → Alice: SHA-256 of Bob's current bits, for convergence check.
    KeyHash { hash: [u8; 32] },
    /// Alice → Bob: whether Bob's hash matches Alice's key.
    KeyHashVerdict { matches: bool },
    /// Bob → (driver): reconciliation finished, keys verified equal.
    Done,
}

/// Outcome of a completed reconciliation.
#[derive(Debug, Clone)]
pub struct CascadeOutcome {
    /// Bob's corrected bits (verified equal to Alice's via hash exchange).
    pub corrected: Vec<u8>,
    /// Number of parity bits revealed on the classical channel.
    pub leaked_bits: usize,
}

/// Alice's side of CASCADE: answers parity queries over her reference bits
/// and counts every revealed parity as one leaked bit.
pub struct AliceCascade<'a> {
    bits: &'a [u8],
    leaked_bits: usize,
}

impl<'a> AliceCascade<'a> {
    pub fn new(bits: &'a [u8]) -> Self {
        Self {
            bits,
            leaked_bits: 0,
        }
    }

    /// Handle one message from Bob and produce the response.
    pub fn handle(&mut self, msg: CascadeMessage) -> QkdResult<CascadeMessage> {
        match msg {
            CascadeMessage::ParityRequest { indices } => {
                let parity = parity_of(self.bits, &indices)?;
                // One parity response = one bit of information revealed.
                self.leaked_bits += 1;
                Ok(CascadeMessage::ParityResponse { parity })
            }
            CascadeMessage::KeyHash { hash } => Ok(CascadeMessage::KeyHashVerdict {
                matches: hash == bits_hash(self.bits),
            }),
            other => Err(QkdError::ErrorCorrection(format!(
                "Alice received unexpected CASCADE message: {other:?}"
            ))),
        }
    }

    /// Total parity bits revealed so far.
    pub fn leaked_bits(&self) -> usize {
        self.leaked_bits
    }
}

/// What Bob is waiting for from Alice.
enum BobState {
    /// Ready to emit the next query.
    Idle,
    /// Awaiting parity of the full block `block_id`.
    AwaitBlockParity { block_id: usize },
    /// Awaiting parity of the left half of `segment` (binary search inside
    /// block `block_id`).
    AwaitBinary {
        block_id: usize,
        segment: Vec<usize>,
    },
    /// Awaiting Alice's verdict on Bob's key hash.
    AwaitVerdict,
    /// Reconciliation complete.
    Finished,
}

/// Bob's side of CASCADE: corrects his bits toward Alice's by querying
/// block parities and binary-searching mismatches.
pub struct BobCascade {
    bits: Vec<u8>,
    /// All blocks ever scheduled (index = block id). Used for the cascade
    /// effect: a corrected bit triggers re-checks of every other block
    /// containing that position.
    blocks: Vec<Vec<usize>>,
    queue: VecDeque<usize>,
    queued: HashSet<usize>,
    state: BobState,
    shuffle_seed: u64,
    extra_passes: usize,
    parities_received: usize,
}

impl BobCascade {
    /// `estimated_qber` sizes the first-pass blocks (~0.73 / QBER per the
    /// CASCADE paper); `shuffle_seed` determines the public block shuffles.
    pub fn new(bits: Vec<u8>, estimated_qber: f64, shuffle_seed: u64) -> Self {
        let n = bits.len();
        let initial_block_size = if estimated_qber > 0.0 {
            (0.73 / estimated_qber).ceil() as usize
        } else {
            n.max(1)
        }
        .max(2)
        .min(n.max(1));

        let mut bob = Self {
            bits,
            blocks: Vec::new(),
            queue: VecDeque::new(),
            queued: HashSet::new(),
            state: BobState::Idle,
            shuffle_seed,
            extra_passes: 0,
            parities_received: 0,
        };

        for pass in 0..NUM_PASSES {
            let block_size = if pass == 0 {
                initial_block_size
            } else {
                (initial_block_size << pass).min(n.max(1))
            };
            bob.schedule_pass(
                block_size,
                shuffle_seed.wrapping_add(pass as u64),
                pass != 0,
            );
        }

        bob
    }

    /// Schedule one pass of blocks of `block_size` over a (possibly shuffled)
    /// index permutation.
    fn schedule_pass(&mut self, block_size: usize, seed: u64, shuffle: bool) {
        let n = self.bits.len();
        if n == 0 {
            return;
        }
        let mut indices: Vec<usize> = (0..n).collect();
        if shuffle {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        for chunk in indices.chunks(block_size.max(1)) {
            let id = self.blocks.len();
            self.blocks.push(chunk.to_vec());
            self.queue.push_back(id);
            self.queued.insert(id);
        }
    }

    /// Advance the state machine. `response` must be Alice's answer to the
    /// previously emitted message (None on the first call). Returns the next
    /// message to send to Alice, or [`CascadeMessage::Done`] when Bob's bits
    /// are verified equal to Alice's.
    pub fn step(&mut self, response: Option<CascadeMessage>) -> QkdResult<CascadeMessage> {
        // Consume the pending response.
        match std::mem::replace(&mut self.state, BobState::Idle) {
            BobState::Idle => {
                if response.is_some() {
                    return Err(QkdError::ErrorCorrection(
                        "Unexpected response with no outstanding query".into(),
                    ));
                }
            }
            BobState::Finished => return Ok(CascadeMessage::Done),
            BobState::AwaitBlockParity { block_id } => {
                let alice_parity = expect_parity(response)?;
                self.parities_received += 1;
                let mine = parity_of(&self.bits, &self.blocks[block_id])?;
                if alice_parity != mine {
                    if let Some(msg) = self.begin_binary(block_id)? {
                        return Ok(msg);
                    }
                }
            }
            BobState::AwaitBinary { block_id, segment } => {
                let alice_left_parity = expect_parity(response)?;
                self.parities_received += 1;
                let mid = segment.len() / 2;
                let (left, right) = segment.split_at(mid);
                let my_left = parity_of(&self.bits, left)?;
                let chosen: Vec<usize> = if my_left != alice_left_parity {
                    left.to_vec()
                } else {
                    right.to_vec()
                };
                if let Some(msg) = self.descend(block_id, chosen)? {
                    return Ok(msg);
                }
            }
            BobState::AwaitVerdict => match response {
                Some(CascadeMessage::KeyHashVerdict { matches: true }) => {
                    self.state = BobState::Finished;
                    return Ok(CascadeMessage::Done);
                }
                Some(CascadeMessage::KeyHashVerdict { matches: false }) => {
                    // Residual errors (even error counts in every block of
                    // every pass). Schedule another shuffled pass.
                    self.extra_passes += 1;
                    if self.extra_passes > MAX_EXTRA_PASSES {
                        return Err(QkdError::ErrorCorrection(format!(
                            "CASCADE failed to converge after {MAX_EXTRA_PASSES} extra passes"
                        )));
                    }
                    let n = self.bits.len();
                    let block_size = (n / 16).max(2);
                    let seed = self
                        .shuffle_seed
                        .wrapping_add(1000)
                        .wrapping_add(self.extra_passes as u64);
                    debug!(extra_pass = self.extra_passes, "CASCADE extra pass");
                    self.schedule_pass(block_size, seed, true);
                }
                other => {
                    return Err(QkdError::ErrorCorrection(format!(
                        "Expected KeyHashVerdict, got {other:?}"
                    )));
                }
            },
        }

        // Emit the next query.
        if let Some(block_id) = self.queue.pop_front() {
            self.queued.remove(&block_id);
            let indices = self.blocks[block_id].clone();
            self.state = BobState::AwaitBlockParity { block_id };
            return Ok(CascadeMessage::ParityRequest { indices });
        }

        // No work left: verify convergence via hash exchange.
        self.state = BobState::AwaitVerdict;
        Ok(CascadeMessage::KeyHash {
            hash: bits_hash(&self.bits),
        })
    }

    /// Start a binary search inside `block_id` (its parity mismatched).
    /// Returns the next query, or None if the block has a single bit (which
    /// is then flipped directly).
    fn begin_binary(&mut self, block_id: usize) -> QkdResult<Option<CascadeMessage>> {
        let segment = self.blocks[block_id].clone();
        self.descend(block_id, segment)
    }

    /// Continue binary search on `segment`, which is known to contain an odd
    /// number of errors.
    fn descend(
        &mut self,
        block_id: usize,
        segment: Vec<usize>,
    ) -> QkdResult<Option<CascadeMessage>> {
        if segment.len() == 1 {
            self.flip(segment[0], block_id);
            return Ok(None);
        }
        let mid = segment.len() / 2;
        let left = segment[..mid].to_vec();
        self.state = BobState::AwaitBinary { block_id, segment };
        Ok(Some(CascadeMessage::ParityRequest { indices: left }))
    }

    /// Flip a corrected bit and trigger the cascade effect: every other
    /// already-scheduled block containing this position must be re-checked.
    fn flip(&mut self, pos: usize, current_block: usize) {
        self.bits[pos] ^= 1;
        debug!(position = pos, "Corrected error");
        for (id, block) in self.blocks.iter().enumerate() {
            if id != current_block && !self.queued.contains(&id) && block.contains(&pos) {
                self.queue.push_back(id);
                self.queued.insert(id);
            }
        }
    }

    /// Number of parity responses received (Bob's view of the leakage; equals
    /// Alice's [`AliceCascade::leaked_bits`]).
    pub fn parities_received(&self) -> usize {
        self.parities_received
    }

    /// Consume the state machine, returning the corrected bits.
    pub fn into_bits(self) -> Vec<u8> {
        self.bits
    }
}

fn expect_parity(response: Option<CascadeMessage>) -> QkdResult<u8> {
    match response {
        Some(CascadeMessage::ParityResponse { parity }) => Ok(parity),
        other => Err(QkdError::ErrorCorrection(format!(
            "Expected ParityResponse, got {other:?}"
        ))),
    }
}

fn parity_of(bits: &[u8], indices: &[usize]) -> QkdResult<u8> {
    let mut parity = 0u8;
    for &i in indices {
        let bit = bits.get(i).ok_or_else(|| {
            QkdError::ErrorCorrection(format!("Parity index {i} out of range ({})", bits.len()))
        })?;
        parity ^= bit & 1;
    }
    Ok(parity)
}

fn bits_hash(bits: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bits);
    hasher.finalize().into()
}

/// Two-party CASCADE reconciliation, driven in-process.
///
/// Bob's bits are corrected toward Alice's; the returned
/// [`CascadeOutcome::leaked_bits`] counts every parity revealed on the
/// classical channel and must be subtracted during privacy amplification.
pub fn reconcile(
    alice_bits: &[u8],
    bob_bits: &[u8],
    estimated_qber: f64,
) -> QkdResult<CascadeOutcome> {
    if alice_bits.len() != bob_bits.len() {
        return Err(QkdError::ErrorCorrection(
            "Alice and Bob key lengths differ".into(),
        ));
    }
    if alice_bits.is_empty() {
        return Err(QkdError::ErrorCorrection("Empty key".into()));
    }

    let mut alice = AliceCascade::new(alice_bits);
    let mut bob = BobCascade::new(bob_bits.to_vec(), estimated_qber, CASCADE_SHUFFLE_SEED);

    let mut response: Option<CascadeMessage> = None;
    loop {
        match bob.step(response.take())? {
            CascadeMessage::Done => break,
            outgoing @ (CascadeMessage::ParityRequest { .. } | CascadeMessage::KeyHash { .. }) => {
                response = Some(alice.handle(outgoing)?);
            }
            other => {
                return Err(QkdError::ErrorCorrection(format!(
                    "Unexpected message from Bob: {other:?}"
                )));
            }
        }
    }

    let leaked_bits = alice.leaked_bits();
    debug!(leaked_bits, "CASCADE reconciliation complete");
    Ok(CascadeOutcome {
        corrected: bob.into_bits(),
        leaked_bits,
    })
}

/// Correct errors in Bob's key to match Alice's key.
///
/// Backward-compatible wrapper over [`reconcile`] that discards the leakage
/// count. Prefer [`reconcile`] so the leaked parity bits can be subtracted
/// during privacy amplification.
pub fn correct(alice_bits: &[u8], bob_bits: &[u8], estimated_qber: f64) -> QkdResult<Vec<u8>> {
    reconcile(alice_bits, bob_bits, estimated_qber).map(|outcome| outcome.corrected)
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

    // ---- Two-party message-driven CASCADE (Workstream 5) ----

    fn random_bits(n: usize, seed: u64) -> Vec<u8> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(0u8..2)).collect()
    }

    fn flip_fraction(bits: &[u8], fraction: f64, seed: u64) -> Vec<u8> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        bits.iter()
            .map(|&b| {
                if rng.gen::<f64>() < fraction {
                    b ^ 1
                } else {
                    b
                }
            })
            .collect()
    }

    #[test]
    fn test_reconcile_converges_at_3_percent_qber() {
        let alice = random_bits(2048, 1);
        let bob = flip_fraction(&alice, 0.03, 2);
        let outcome = reconcile(&alice, &bob, 0.03).unwrap();
        assert_eq!(outcome.corrected, alice, "Bob must converge to Alice's key");
        assert!(outcome.leaked_bits > 0);
    }

    #[test]
    fn test_reconcile_converges_at_8_percent_qber() {
        let alice = random_bits(2048, 3);
        let bob = flip_fraction(&alice, 0.08, 4);
        let outcome = reconcile(&alice, &bob, 0.08).unwrap();
        assert_eq!(outcome.corrected, alice, "Bob must converge to Alice's key");
        assert!(outcome.leaked_bits > 0);
    }

    #[test]
    fn test_leakage_is_plausible() {
        let alice = random_bits(2048, 5);
        let bob = flip_fraction(&alice, 0.05, 6);
        let outcome = reconcile(&alice, &bob, 0.05).unwrap();
        // Every parity response is one leaked bit: more than zero, but far
        // fewer than the key itself for moderate QBER.
        assert!(outcome.leaked_bits > 0);
        assert!(
            outcome.leaked_bits < alice.len(),
            "leaked {} bits out of {}",
            outcome.leaked_bits,
            alice.len()
        );
    }

    #[test]
    fn test_zero_error_leaks_only_initial_pass_parities() {
        let alice = random_bits(1024, 7);
        let bob = alice.clone();
        let outcome = reconcile(&alice, &bob, 0.0).unwrap();
        assert_eq!(outcome.corrected, alice);
        // With no errors, the only parities exchanged are the per-block
        // parities of each pass (one query per block, no binary search).
        assert_eq!(outcome.leaked_bits, NUM_PASSES);
    }

    #[test]
    fn test_message_driven_api_matches_wrapper() {
        let alice_bits = random_bits(1024, 8);
        let bob_bits = flip_fraction(&alice_bits, 0.04, 9);

        // Wrapper result
        let outcome = reconcile(&alice_bits, &bob_bits, 0.04).unwrap();

        // Drive the state machines manually via explicit messages
        let mut alice = AliceCascade::new(&alice_bits);
        let mut bob = BobCascade::new(bob_bits.clone(), 0.04, CASCADE_SHUFFLE_SEED);
        let mut response: Option<CascadeMessage> = None;
        let corrected = loop {
            match bob.step(response.take()).unwrap() {
                CascadeMessage::ParityRequest { indices } => {
                    response = Some(
                        alice
                            .handle(CascadeMessage::ParityRequest { indices })
                            .unwrap(),
                    );
                }
                CascadeMessage::KeyHash { hash } => {
                    response = Some(alice.handle(CascadeMessage::KeyHash { hash }).unwrap());
                }
                CascadeMessage::Done => break bob.into_bits(),
                other => panic!("unexpected message from Bob: {other:?}"),
            }
        };

        assert_eq!(corrected, outcome.corrected);
        assert_eq!(alice.leaked_bits(), outcome.leaked_bits);
        assert_eq!(corrected, alice_bits);
    }

    #[test]
    fn test_correct_wrapper_still_works() {
        let alice = random_bits(512, 10);
        let bob = flip_fraction(&alice, 0.03, 11);
        let result = correct(&alice, &bob, 0.03).unwrap();
        assert_eq!(result, alice);
    }
}
