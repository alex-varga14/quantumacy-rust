//! Peer-to-peer QKD key exchange over a real, authenticated TCP transport.
//!
//! Alice and Bob run as independent async tasks (or independent OS
//! processes) connected by TCP. Every message travels over an
//! [`AuthenticatedChannel`]: HMAC-SHA256 tags keyed by a pre-shared
//! authentication key protect the classical traffic from tampering — QKD's
//! actual hard requirement. (Needing a small pre-shared secret is standard:
//! QKD is key *expansion*, not key creation from nothing.)
//!
//! # The simulation boundary
//!
//! There is no quantum hardware. The qubit channel is simulated: Alice
//! prepares qubits and runs them through [`QuantumChannel`] locally
//! (modelling loss, noise, and eavesdropping), then ships the post-channel
//! quantum states to Bob inside a [`P2pMessage::SimulatedQuantum`] message
//! over the same TCP link. That message is the simulation boundary — in a
//! real deployment it would be replaced by photons in fibre, and Bob's
//! measurement would happen in his detector instead of in his process. Bob
//! still chooses his own random measurement bases locally, so everything
//! after the quantum transmission (sifting, QBER estimation, CASCADE,
//! privacy-amplification seed exchange) is a genuine two-party protocol over
//! authenticated classical messages.
//!
//! The old in-process single-task simulation is kept as [`QkdPeer`], a test
//! fixture and convenience wrapper.

use crate::classical_channel::AuthenticatedChannel;
use crate::{NetworkError, NetworkResult};
use qkd_core::channel::{ChannelConfig, QuantumChannel};
use qkd_core::error_correction::cascade::{
    AliceCascade, BobCascade, CascadeMessage, CASCADE_SHUFFLE_SEED,
};
use qkd_core::key_manager::{KeyManager, KeyManagerConfig};
use qkd_core::privacy_amplification::toeplitz_amplify;
use qkd_core::protocols::{b92::B92, bb84::Bb84, six_state::SixState, QkdProtocol};
use qkd_core::types::{Basis, ProtocolType, QkdStats, Qubit, SecureKey};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info, warn};

/// Wire messages for the P2P QKD session. All of them flow over the
/// HMAC-authenticated classical channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum P2pMessage {
    /// Alice → Bob: session parameters.
    Hello {
        protocol: ProtocolType,
        num_qubits: usize,
        qber_threshold: f64,
        min_key_bits: usize,
    },
    /// Bob → Alice: parameters accepted.
    HelloAck,
    /// Alice → Bob: **SIMULATED QUANTUM TRANSMISSION.**
    ///
    /// This is the simulation boundary (see module docs): the post-channel
    /// quantum states that would physically arrive at Bob's detector. `None`
    /// entries are photons lost in the channel. Only possible in simulation —
    /// a classical message can never carry real qubits.
    SimulatedQuantum { received: Vec<Option<Qubit>> },
    /// Bob → Alice: measurement bases per position (BB84 / Six-State
    /// sifting). `None` = no detection.
    SiftBases { bases: Vec<Option<Basis>> },
    /// Bob → Alice: positions with conclusive measurements (B92 sifting).
    SiftConclusive { indices: Vec<usize> },
    /// Alice → Bob: positions (into the announced sequence) both keep.
    SiftKeep { indices: Vec<usize> },
    /// Alice → Bob: sifted-key positions sacrificed for QBER estimation.
    QberSample { indices: Vec<usize> },
    /// Bob → Alice: Bob's bits at the sampled positions.
    QberSampleBits { bits: Vec<u8> },
    /// Alice → Bob: measured QBER and whether to proceed.
    QberVerdict { qber: f64, proceed: bool },
    /// Two-party CASCADE error correction traffic (Item: every
    /// `ParityResponse` Alice sends is one leaked bit, counted by both ends).
    Cascade(CascadeMessage),
    /// Alice → Bob: privacy-amplification parameters. The Toeplitz seed is
    /// public; `leaked_bits` lets Bob cross-check his own leakage count.
    PaParams {
        seed: [u8; 32],
        leaked_bits: usize,
        key_id: String,
    },
    /// Either direction: abort with reason.
    Abort { reason: String },
}

/// Session parameters for Alice (the initiating peer).
#[derive(Debug, Clone)]
pub struct P2pSessionConfig {
    pub protocol: ProtocolType,
    pub num_qubits: usize,
    pub channel: ChannelConfig,
    /// Fraction of sifted bits sacrificed for QBER estimation.
    pub sample_fraction: f64,
    /// Abort threshold for the measured QBER.
    pub qber_threshold: f64,
    /// Minimum acceptable final key length in bits.
    pub min_key_bits: usize,
}

impl Default for P2pSessionConfig {
    fn default() -> Self {
        Self::for_protocol(ProtocolType::BB84)
    }
}

impl P2pSessionConfig {
    /// Defaults matching the qkd-core protocol implementations.
    pub fn for_protocol(protocol: ProtocolType) -> Self {
        let (sample_fraction, qber_threshold) = match protocol {
            ProtocolType::BB84 => (0.1, 0.11),
            ProtocolType::SixState => (0.1, 0.126),
            ProtocolType::B92 => (0.15, 0.11),
        };
        Self {
            protocol,
            num_qubits: 4096,
            channel: ChannelConfig::default(),
            sample_fraction,
            qber_threshold,
            min_key_bits: 256,
        }
    }
}

fn unexpected(msg: P2pMessage, expected: &str) -> NetworkError {
    match msg {
        P2pMessage::Abort { reason } => NetworkError::Protocol(format!("Peer aborted: {reason}")),
        other => NetworkError::Protocol(format!("Expected {expected}, got {other:?}")),
    }
}

/// Run the Alice (initiator) side of a P2P QKD session over `stream`.
///
/// `auth_key` is the pre-shared classical-channel authentication key; both
/// peers must hold the same bytes or the very first frame is rejected.
pub async fn run_alice<S>(
    stream: S,
    auth_key: &[u8],
    cfg: P2pSessionConfig,
) -> NetworkResult<(SecureKey, QkdStats)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut ch = AuthenticatedChannel::new(stream, auth_key);
    let mut rng = StdRng::from_entropy();
    let n = cfg.num_qubits;

    info!(protocol = ?cfg.protocol, qubits = n, "Alice: starting P2P session");
    ch.send(&P2pMessage::Hello {
        protocol: cfg.protocol,
        num_qubits: n,
        qber_threshold: cfg.qber_threshold,
        min_key_bits: cfg.min_key_bits,
    })
    .await?;
    match ch.recv().await? {
        P2pMessage::HelloAck => {}
        other => return Err(unexpected(other, "HelloAck")),
    }

    // --- Simulated quantum phase (see module docs) ---
    let (alice_bits, alice_qubits): (Vec<u8>, Vec<Qubit>) = match cfg.protocol {
        ProtocolType::BB84 => {
            let qubits = Bb84::default().alice_prepare(n, &mut rng);
            (qubits.iter().map(|q| q.value.as_bit()).collect(), qubits)
        }
        ProtocolType::SixState => {
            let qubits = SixState::default().alice_prepare(n, &mut rng);
            (qubits.iter().map(|q| q.value.as_bit()).collect(), qubits)
        }
        ProtocolType::B92 => B92::default().alice_prepare(n, &mut rng),
    };
    let channel = QuantumChannel::new(cfg.channel.clone());
    let received: Vec<Option<Qubit>> = alice_qubits
        .iter()
        .map(|q| channel.transmit(q, &mut rng))
        .collect();
    let received_count = received.iter().filter(|q| q.is_some()).count();
    ch.send(&P2pMessage::SimulatedQuantum { received }).await?;

    // --- Sifting (authenticated classical) ---
    let alice_sifted: Vec<u8> = match cfg.protocol {
        ProtocolType::BB84 | ProtocolType::SixState => {
            let bases = match ch.recv().await? {
                P2pMessage::SiftBases { bases } => bases,
                other => return Err(unexpected(other, "SiftBases")),
            };
            if bases.len() != n {
                return Err(NetworkError::Protocol(format!(
                    "SiftBases length {} != {n}",
                    bases.len()
                )));
            }
            let keep: Vec<usize> = bases
                .iter()
                .enumerate()
                .filter(|(i, b)| **b == Some(alice_qubits[*i].basis))
                .map(|(i, _)| i)
                .collect();
            let sifted = keep.iter().map(|&i| alice_bits[i]).collect();
            ch.send(&P2pMessage::SiftKeep { indices: keep }).await?;
            sifted
        }
        ProtocolType::B92 => {
            let indices = match ch.recv().await? {
                P2pMessage::SiftConclusive { indices } => indices,
                other => return Err(unexpected(other, "SiftConclusive")),
            };
            indices
                .iter()
                .map(|&i| {
                    alice_bits.get(i).copied().ok_or_else(|| {
                        NetworkError::Protocol(format!("Conclusive index {i} out of range"))
                    })
                })
                .collect::<NetworkResult<Vec<u8>>>()?
        }
    };
    let sifted_len = alice_sifted.len();
    debug!(sifted_bits = sifted_len, "Alice: sifting complete");

    // --- QBER estimation ---
    let (qber, alice_remaining) =
        alice_estimate_qber(&mut ch, &mut rng, &alice_sifted, cfg.sample_fraction).await?;
    let proceed = qber <= cfg.qber_threshold;
    ch.send(&P2pMessage::QberVerdict { qber, proceed }).await?;
    if !proceed {
        warn!(
            qber = format!("{qber:.4}"),
            "Alice: aborting — QBER over threshold"
        );
        return Err(NetworkError::Qkd(
            qkd_core::QkdError::EavesdroppingDetected {
                qber,
                threshold: cfg.qber_threshold,
            },
        ));
    }

    // --- CASCADE: Alice answers Bob's parity queries, counting leakage ---
    let leaked_bits = {
        let mut alice_cascade = AliceCascade::new(&alice_remaining);
        loop {
            let msg = match ch.recv().await? {
                P2pMessage::Cascade(m) => m,
                other => return Err(unexpected(other, "Cascade")),
            };
            match msg {
                CascadeMessage::Done => break,
                m @ (CascadeMessage::ParityRequest { .. } | CascadeMessage::KeyHash { .. }) => {
                    let resp = alice_cascade.handle(m).map_err(NetworkError::Qkd)?;
                    ch.send(&P2pMessage::Cascade(resp)).await?;
                }
                other => {
                    return Err(NetworkError::Protocol(format!(
                        "Unexpected CASCADE message from Bob: {other:?}"
                    )))
                }
            }
        }
        alice_cascade.leaked_bits()
    };
    debug!(leaked_bits, "Alice: CASCADE complete");

    // --- Privacy amplification: share seed + leakage, derive key ---
    let seed: [u8; 32] = rng.gen();
    let key_id = uuid::Uuid::new_v4().to_string();
    ch.send(&P2pMessage::PaParams {
        seed,
        leaked_bits,
        key_id: key_id.clone(),
    })
    .await?;
    let material =
        toeplitz_amplify(&alice_remaining, qber, leaked_bits, &seed).map_err(NetworkError::Qkd)?;
    if material.len() * 8 < cfg.min_key_bits {
        return Err(NetworkError::Qkd(qkd_core::QkdError::InsufficientKeyBits {
            got: material.len() * 8,
            need: cfg.min_key_bits,
        }));
    }

    let final_key_bits = material.len() * 8;
    let key = SecureKey {
        key_id,
        timestamp: chrono::Utc::now(),
        material,
        length_bits: final_key_bits,
    };
    let stats = QkdStats {
        protocol: cfg.protocol,
        qubits_sent: n,
        qubits_received: received_count,
        sifting_rate: sifted_len as f64 / n as f64,
        qber,
        raw_key_bits: alice_remaining.len(),
        final_key_bits,
        key_rate: final_key_bits as f64 / n as f64,
        eavesdropping_detected: false,
        leaked_bits,
    };
    info!(
        key_id = %key.key_id,
        final_key_bits,
        leaked_bits,
        "Alice: P2P key exchange complete"
    );
    Ok((key, stats))
}

/// Run the Bob (responder) side of a P2P QKD session over `stream`.
pub async fn run_bob<S>(stream: S, auth_key: &[u8]) -> NetworkResult<(SecureKey, QkdStats)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut ch = AuthenticatedChannel::new(stream, auth_key);
    let mut rng = StdRng::from_entropy();

    let (protocol, num_qubits, qber_threshold, min_key_bits) = match ch.recv().await? {
        P2pMessage::Hello {
            protocol,
            num_qubits,
            qber_threshold,
            min_key_bits,
        } => (protocol, num_qubits, qber_threshold, min_key_bits),
        other => return Err(unexpected(other, "Hello")),
    };
    info!(protocol = ?protocol, qubits = num_qubits, "Bob: joining P2P session");
    ch.send(&P2pMessage::HelloAck).await?;

    // --- Receive the simulated quantum transmission, measure locally ---
    let received = match ch.recv().await? {
        P2pMessage::SimulatedQuantum { received } => received,
        other => return Err(unexpected(other, "SimulatedQuantum")),
    };
    if received.len() != num_qubits {
        return Err(NetworkError::Protocol(format!(
            "SimulatedQuantum length {} != {num_qubits}",
            received.len()
        )));
    }
    let received_count = received.iter().filter(|q| q.is_some()).count();

    // Bob's basis choices and outcomes are his own randomness; only the
    // announcements below travel back to Alice.
    let bob_sifted: Vec<u8> = match protocol {
        ProtocolType::BB84 | ProtocolType::SixState => {
            let measurements = match protocol {
                ProtocolType::BB84 => Bb84::default().bob_measure(&received, &mut rng),
                _ => SixState::default().bob_measure(&received, &mut rng),
            };
            let bases: Vec<Option<Basis>> =
                measurements.iter().map(|m| m.map(|(b, _)| b)).collect();
            let values: Vec<Option<u8>> = measurements
                .iter()
                .map(|m| m.map(|(_, v)| v.as_bit()))
                .collect();
            ch.send(&P2pMessage::SiftBases { bases }).await?;
            let keep = match ch.recv().await? {
                P2pMessage::SiftKeep { indices } => indices,
                other => return Err(unexpected(other, "SiftKeep")),
            };
            keep.iter()
                .map(|&i| {
                    values.get(i).copied().flatten().ok_or_else(|| {
                        NetworkError::Protocol(format!("SiftKeep index {i} invalid"))
                    })
                })
                .collect::<NetworkResult<Vec<u8>>>()?
        }
        ProtocolType::B92 => {
            let results = B92::default().bob_measure(&received, &mut rng);
            let (indices, bits): (Vec<usize>, Vec<u8>) = results
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.map(|bit| (i, bit)))
                .unzip();
            ch.send(&P2pMessage::SiftConclusive { indices }).await?;
            bits
        }
    };
    let sifted_len = bob_sifted.len();
    debug!(sifted_bits = sifted_len, "Bob: sifting complete");

    // --- QBER estimation: reveal the sampled bits Alice asks for ---
    let sample_indices = match ch.recv().await? {
        P2pMessage::QberSample { indices } => indices,
        other => return Err(unexpected(other, "QberSample")),
    };
    let sample_bits = sample_indices
        .iter()
        .map(|&i| {
            bob_sifted
                .get(i)
                .copied()
                .ok_or_else(|| NetworkError::Protocol(format!("QberSample index {i} out of range")))
        })
        .collect::<NetworkResult<Vec<u8>>>()?;
    ch.send(&P2pMessage::QberSampleBits { bits: sample_bits })
        .await?;

    let (qber, proceed) = match ch.recv().await? {
        P2pMessage::QberVerdict { qber, proceed } => (qber, proceed),
        other => return Err(unexpected(other, "QberVerdict")),
    };
    if !proceed {
        warn!(
            qber = format!("{qber:.4}"),
            "Bob: peer aborted — QBER over threshold"
        );
        return Err(NetworkError::Qkd(
            qkd_core::QkdError::EavesdroppingDetected {
                qber,
                threshold: qber_threshold,
            },
        ));
    }
    let sampled: HashSet<usize> = sample_indices.into_iter().collect();
    let bob_remaining: Vec<u8> = bob_sifted
        .iter()
        .enumerate()
        .filter(|(i, _)| !sampled.contains(i))
        .map(|(_, &b)| b)
        .collect();

    // --- CASCADE: Bob drives the reconciliation toward Alice's key ---
    let mut bob_cascade = BobCascade::new(bob_remaining, qber, CASCADE_SHUFFLE_SEED);
    let mut response: Option<CascadeMessage> = None;
    loop {
        let outgoing = bob_cascade
            .step(response.take())
            .map_err(NetworkError::Qkd)?;
        let done = matches!(outgoing, CascadeMessage::Done);
        ch.send(&P2pMessage::Cascade(outgoing)).await?;
        if done {
            break;
        }
        response = Some(match ch.recv().await? {
            P2pMessage::Cascade(m) => m,
            other => return Err(unexpected(other, "Cascade")),
        });
    }
    let parities_seen = bob_cascade.parities_received();
    let corrected = bob_cascade.into_bits();
    debug!(leaked_bits = parities_seen, "Bob: CASCADE complete");

    // --- Privacy amplification ---
    let (seed, leaked_bits, key_id) = match ch.recv().await? {
        P2pMessage::PaParams {
            seed,
            leaked_bits,
            key_id,
        } => (seed, leaked_bits, key_id),
        other => return Err(unexpected(other, "PaParams")),
    };
    // Cross-check the leakage accounting: both ends counted every parity.
    if leaked_bits != parities_seen {
        return Err(NetworkError::Protocol(format!(
            "Leakage accounting mismatch: Alice reports {leaked_bits}, Bob counted {parities_seen}"
        )));
    }
    let material =
        toeplitz_amplify(&corrected, qber, leaked_bits, &seed).map_err(NetworkError::Qkd)?;
    if material.len() * 8 < min_key_bits {
        return Err(NetworkError::Qkd(qkd_core::QkdError::InsufficientKeyBits {
            got: material.len() * 8,
            need: min_key_bits,
        }));
    }

    let final_key_bits = material.len() * 8;
    let key = SecureKey {
        key_id,
        timestamp: chrono::Utc::now(),
        material,
        length_bits: final_key_bits,
    };
    let stats = QkdStats {
        protocol,
        qubits_sent: num_qubits,
        qubits_received: received_count,
        sifting_rate: sifted_len as f64 / num_qubits as f64,
        qber,
        raw_key_bits: corrected.len(),
        final_key_bits,
        key_rate: final_key_bits as f64 / num_qubits as f64,
        eavesdropping_detected: false,
        leaked_bits,
    };
    info!(
        key_id = %key.key_id,
        final_key_bits,
        leaked_bits,
        "Bob: P2P key exchange complete"
    );
    Ok((key, stats))
}

/// Alice's QBER sampling round: pick random sifted positions, ask Bob for
/// his bits there, and return the measured QBER plus the surviving bits.
async fn alice_estimate_qber<S>(
    ch: &mut AuthenticatedChannel<S>,
    rng: &mut StdRng,
    alice_sifted: &[u8],
    sample_fraction: f64,
) -> NetworkResult<(f64, Vec<u8>)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let m = alice_sifted.len();
    if m < 2 {
        return Err(NetworkError::Protocol(
            "Too few sifted bits for QBER estimation".into(),
        ));
    }
    let sample_size = ((m as f64 * sample_fraction).ceil() as usize).clamp(1, m - 1);
    let indices: Vec<usize> = rand::seq::index::sample(rng, m, sample_size).into_vec();

    ch.send(&P2pMessage::QberSample {
        indices: indices.clone(),
    })
    .await?;
    let bob_bits = match ch.recv().await? {
        P2pMessage::QberSampleBits { bits } => bits,
        other => return Err(unexpected(other, "QberSampleBits")),
    };
    if bob_bits.len() != sample_size {
        return Err(NetworkError::Protocol(format!(
            "QberSampleBits length {} != {sample_size}",
            bob_bits.len()
        )));
    }

    let errors = indices
        .iter()
        .zip(bob_bits.iter())
        .filter(|(&i, &b)| alice_sifted[i] != b)
        .count();
    let qber = errors as f64 / sample_size as f64;

    let sampled: HashSet<usize> = indices.into_iter().collect();
    let remaining: Vec<u8> = alice_sifted
        .iter()
        .enumerate()
        .filter(|(i, _)| !sampled.contains(i))
        .map(|(_, &b)| b)
        .collect();
    Ok((qber, remaining))
}

/// In-process P2P peer — **test fixture / legacy convenience only**.
///
/// Simulates both parties inside one task via `QkdProtocol::execute`. Real
/// distributed exchanges should use [`run_alice`] / [`run_bob`] over TCP.
pub struct QkdPeer {
    peer_id: String,
    key_manager: KeyManager,
}

impl QkdPeer {
    pub fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            key_manager: KeyManager::new(KeyManagerConfig::default()),
        }
    }

    /// Execute a key exchange simulated entirely in-process.
    pub async fn exchange_key(
        &self,
        protocol: &dyn QkdProtocol,
        channel_config: &ChannelConfig,
        num_qubits: usize,
    ) -> NetworkResult<SecureKey> {
        let cc = channel_config.clone();
        let nq = num_qubits;
        let proto_name = protocol.name().to_string();

        let (key, stats) = tokio::task::spawn_blocking(move || {
            let proto = Bb84::default();
            proto.execute(nq, &cc)
        })
        .await
        .map_err(|e| NetworkError::Connection(format!("Join error: {e}")))?
        .map_err(NetworkError::Qkd)?;

        info!(
            peer = %self.peer_id,
            protocol = %proto_name,
            key_id = %key.key_id,
            qber = format!("{:.4}", stats.qber),
            "In-process P2P key exchange complete"
        );

        self.key_manager.store(key.clone())?;
        Ok(key)
    }

    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }

    pub fn available_keys(&self) -> usize {
        self.key_manager.key_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qkd_core::channel::{EavesdropperConfig, EveStrategy};
    use tokio::net::{TcpListener, TcpStream};

    const AUTH_KEY: &[u8] = b"test-preshared-classical-auth-key";

    fn quiet_channel() -> ChannelConfig {
        ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.05,
            dark_count_rate: 0.0,
            eavesdropper: None,
        }
    }

    async fn run_session(
        cfg: P2pSessionConfig,
        bob_auth_key: &'static [u8],
    ) -> (
        NetworkResult<(SecureKey, QkdStats)>,
        NetworkResult<(SecureKey, QkdStats)>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let alice = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            run_alice(stream, AUTH_KEY, cfg).await
        });
        let bob = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            run_bob(stream, bob_auth_key).await
        });

        (alice.await.unwrap(), bob.await.unwrap())
    }

    #[tokio::test]
    async fn test_full_bb84_over_tcp_yields_identical_keys() {
        let cfg = P2pSessionConfig {
            num_qubits: 8192,
            channel: quiet_channel(),
            ..P2pSessionConfig::for_protocol(ProtocolType::BB84)
        };
        let (alice, bob) = run_session(cfg, AUTH_KEY).await;
        let (alice_key, alice_stats) = alice.unwrap();
        let (bob_key, bob_stats) = bob.unwrap();

        assert_eq!(alice_key.key_id, bob_key.key_id);
        assert_eq!(alice_key.material, bob_key.material);
        assert!(
            alice_key.material.len() >= 32,
            "key long enough for AES-256"
        );
        assert!((alice_stats.qber - bob_stats.qber).abs() < f64::EPSILON);
        assert_eq!(alice_stats.final_key_bits, bob_stats.final_key_bits);
    }

    #[tokio::test]
    async fn test_six_state_over_tcp_yields_identical_keys() {
        let cfg = P2pSessionConfig {
            num_qubits: 12288,
            channel: quiet_channel(),
            ..P2pSessionConfig::for_protocol(ProtocolType::SixState)
        };
        let (alice, bob) = run_session(cfg, AUTH_KEY).await;
        let (alice_key, _) = alice.unwrap();
        let (bob_key, _) = bob.unwrap();
        assert_eq!(alice_key.material, bob_key.material);
    }

    #[tokio::test]
    async fn test_b92_over_tcp_yields_identical_keys() {
        let cfg = P2pSessionConfig {
            num_qubits: 16384,
            channel: quiet_channel(),
            ..P2pSessionConfig::for_protocol(ProtocolType::B92)
        };
        let (alice, bob) = run_session(cfg, AUTH_KEY).await;
        let (alice_key, _) = alice.unwrap();
        let (bob_key, _) = bob.unwrap();
        assert_eq!(alice_key.material, bob_key.material);
    }

    #[tokio::test]
    async fn test_wrong_auth_key_aborts_protocol() {
        let cfg = P2pSessionConfig {
            num_qubits: 2048,
            channel: quiet_channel(),
            ..P2pSessionConfig::default()
        };
        let (alice, bob) = run_session(cfg, b"attacker-key-not-the-real-one").await;

        let bob_err = bob.unwrap_err();
        assert!(
            matches!(bob_err, NetworkError::Authentication(_)),
            "Bob must hard-fail on the first forged tag, got: {bob_err:?}"
        );
        assert!(alice.is_err(), "Alice must not complete either");
    }

    #[tokio::test]
    async fn test_qber_threshold_abort_over_network() {
        let cfg = P2pSessionConfig {
            num_qubits: 4096,
            channel: ChannelConfig {
                noise_rate: 0.01,
                loss_rate: 0.0,
                dark_count_rate: 0.0,
                // Full intercept-resend → ~25% QBER, way over threshold.
                eavesdropper: Some(EavesdropperConfig {
                    intercept_rate: 1.0,
                    strategy: EveStrategy::InterceptResend,
                }),
            },
            ..P2pSessionConfig::for_protocol(ProtocolType::BB84)
        };
        let (alice, bob) = run_session(cfg, AUTH_KEY).await;

        for (who, result) in [("alice", alice), ("bob", bob)] {
            match result {
                Err(NetworkError::Qkd(qkd_core::QkdError::EavesdroppingDetected {
                    qber, ..
                })) => {
                    assert!(
                        qber > 0.11,
                        "{who}: detected QBER should be high, got {qber}"
                    );
                }
                other => panic!("{who}: expected EavesdroppingDetected, got {other:?}"),
            }
        }
    }
}
