//! Peer-to-peer QKD key exchange demo over real, authenticated TCP.
//!
//! Mirrors the QKDSimkit P2P CLI from the upstream CERN/Quantumacy repo.
//! Alice and Bob run as independent endpoints connected by TCP; the
//! classical channel is HMAC-SHA256 authenticated with a pre-shared key.
//! Only the qubit transmission itself is simulated (see the simulation
//! boundary notes in `qkd_network::p2p`). Both ends derive the same key,
//! which is then used for an AES-GCM encrypt/decrypt round trip.
//!
//! Usage (single process, two async endpoints over localhost TCP):
//!   cargo run -p qkd-network --example qkd_p2p_demo
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --protocol six-state
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --protocol b92 --qubits 8192
//!
//! Usage (two separate processes):
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --role alice --listen 127.0.0.1:9461
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --role bob --connect 127.0.0.1:9461
//!
//! Environment overrides (read first, CLI flags win):
//!   QUANTUMACY_PROTOCOL=bb84|six-state|b92
//!   QUANTUMACY_QUBITS=<usize>
//!   QUANTUMACY_NOISE=<f64>
//!   QUANTUMACY_EAVESDROPPER=on|off
//!   QUANTUMACY_AUTH_KEY=<pre-shared classical-channel authentication key>

use std::env;
use std::time::Duration;

use qkd_core::channel::{ChannelConfig, EavesdropperConfig, EveStrategy};
use qkd_core::types::{ProtocolType, QkdStats, SecureKey};
use qkd_network::p2p::{run_alice, run_bob, P2pSessionConfig};
use qkd_network::secure_channel::SecureChannel;
use tokio::net::{TcpListener, TcpStream};

const DEFAULT_AUTH_KEY: &str = "quantumacy-demo-preshared-auth-key";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Alice,
    Bob,
}

#[derive(Debug, Clone)]
struct DemoConfig {
    protocol: ProtocolType,
    num_qubits: usize,
    noise_rate: f64,
    eavesdropper: bool,
    role: Option<Role>,
    listen: String,
    connect: String,
    auth_key: String,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            protocol: ProtocolType::BB84,
            num_qubits: 4096,
            noise_rate: 0.02,
            eavesdropper: false,
            role: None,
            listen: "127.0.0.1:9461".to_string(),
            connect: "127.0.0.1:9461".to_string(),
            auth_key: DEFAULT_AUTH_KEY.to_string(),
        }
    }
}

fn parse_protocol(value: &str) -> Option<ProtocolType> {
    match value.to_ascii_lowercase().as_str() {
        "bb84" => Some(ProtocolType::BB84),
        "six-state" | "sixstate" | "six_state" => Some(ProtocolType::SixState),
        "b92" => Some(ProtocolType::B92),
        _ => None,
    }
}

fn load_config() -> DemoConfig {
    let mut cfg = DemoConfig::default();

    if let Ok(p) = env::var("QUANTUMACY_PROTOCOL") {
        if let Some(proto) = parse_protocol(&p) {
            cfg.protocol = proto;
        }
    }
    if let Ok(q) = env::var("QUANTUMACY_QUBITS") {
        if let Ok(parsed) = q.parse() {
            cfg.num_qubits = parsed;
        }
    }
    if let Ok(n) = env::var("QUANTUMACY_NOISE") {
        if let Ok(parsed) = n.parse() {
            cfg.noise_rate = parsed;
        }
    }
    if matches!(
        env::var("QUANTUMACY_EAVESDROPPER").as_deref(),
        Ok("on") | Ok("1") | Ok("true")
    ) {
        cfg.eavesdropper = true;
    }
    if let Ok(k) = env::var("QUANTUMACY_AUTH_KEY") {
        cfg.auth_key = k;
    }

    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--protocol" | "-p" => {
                if let Some(v) = args.next() {
                    if let Some(proto) = parse_protocol(&v) {
                        cfg.protocol = proto;
                    }
                }
            }
            "--qubits" | "-q" => {
                if let Some(v) = args.next() {
                    if let Ok(parsed) = v.parse() {
                        cfg.num_qubits = parsed;
                    }
                }
            }
            "--noise" => {
                if let Some(v) = args.next() {
                    if let Ok(parsed) = v.parse() {
                        cfg.noise_rate = parsed;
                    }
                }
            }
            "--eavesdropper" => cfg.eavesdropper = true,
            "--role" => {
                if let Some(v) = args.next() {
                    cfg.role = match v.to_ascii_lowercase().as_str() {
                        "alice" => Some(Role::Alice),
                        "bob" => Some(Role::Bob),
                        _ => None,
                    };
                }
            }
            "--listen" => {
                if let Some(v) = args.next() {
                    cfg.listen = v;
                }
            }
            "--connect" => {
                if let Some(v) = args.next() {
                    cfg.connect = v;
                }
            }
            "--auth-key" => {
                if let Some(v) = args.next() {
                    cfg.auth_key = v;
                }
            }
            "--help" | "-h" => {
                println!(
                    "qkd_p2p_demo:\n  --protocol bb84|six-state|b92\n  --qubits <usize>\n  --noise <f64>\n  --eavesdropper\n  --role alice|bob   (omit to run both endpoints in one process)\n  --listen <addr>    (alice)\n  --connect <addr>   (bob)\n  --auth-key <str>   (pre-shared classical-channel authentication key)"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }

    cfg
}

fn session_config(cfg: &DemoConfig) -> P2pSessionConfig {
    P2pSessionConfig {
        num_qubits: cfg.num_qubits,
        channel: ChannelConfig {
            noise_rate: cfg.noise_rate,
            eavesdropper: cfg.eavesdropper.then_some(EavesdropperConfig {
                intercept_rate: 0.5,
                strategy: EveStrategy::InterceptResend,
            }),
            ..ChannelConfig::default()
        },
        ..P2pSessionConfig::for_protocol(cfg.protocol)
    }
}

fn print_result(who: &str, key: &SecureKey, stats: &QkdStats) {
    println!(
        "{who}: derived key id={} ({} bits, {} bytes)",
        key.key_id,
        key.length_bits,
        key.material.len()
    );
    println!(
        "{who}: qber={:.4}  sifted_rate={:.3}  raw_bits={}  leaked_parity_bits={}  key_rate={:.4}",
        stats.qber, stats.sifting_rate, stats.raw_key_bits, stats.leaked_bits, stats.key_rate
    );
}

fn aes_round_trip(key: &SecureKey, peer_key: &SecureKey) -> Result<(), Box<dyn std::error::Error>> {
    let sender = SecureChannel::from_key(key)?;
    let receiver = SecureChannel::from_key(peer_key)?;
    let plaintext = b"alice -> bob: privileged medical record handoff";
    let envelope = sender.encrypt(plaintext)?;
    let decrypted = receiver.decrypt(&envelope)?;
    println!(
        "secure channel: encrypted {} -> {} ciphertext bytes (nonce {} bytes)",
        plaintext.len(),
        envelope.ciphertext.len(),
        envelope.nonce.len()
    );
    assert_eq!(decrypted, plaintext);
    println!("secure channel: round-trip decrypt OK");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config();
    let session = session_config(&cfg);
    let auth_key = cfg.auth_key.clone().into_bytes();

    println!("=== Quantumacy P2P QKD Demo (real TCP, authenticated classical channel) ===");
    println!(
        "protocol={:?}  qubits={}  noise={:.3}  eavesdropper={}",
        cfg.protocol, cfg.num_qubits, cfg.noise_rate, cfg.eavesdropper
    );

    match cfg.role {
        // Two-process mode: this process is Alice (listens).
        Some(Role::Alice) => {
            let listener = TcpListener::bind(&cfg.listen).await?;
            println!("alice: listening on {}", listener.local_addr()?);
            let (stream, peer) = listener.accept().await?;
            println!("alice: peer connected from {peer}");
            let (key, stats) = run_alice(stream, &auth_key, session).await?;
            print_result("alice", &key, &stats);
            aes_round_trip(&key, &key)?;
        }
        // Two-process mode: this process is Bob (connects, with retries).
        Some(Role::Bob) => {
            let mut attempts = 0;
            let stream = loop {
                match TcpStream::connect(&cfg.connect).await {
                    Ok(s) => break s,
                    Err(e) if attempts < 50 => {
                        attempts += 1;
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        if attempts == 50 {
                            return Err(e.into());
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            println!("bob: connected to {}", cfg.connect);
            let (key, stats) = run_bob(stream, &auth_key).await?;
            print_result("bob", &key, &stats);
            aes_round_trip(&key, &key)?;
        }
        // Default: both endpoints in this process, still over real TCP.
        None => {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            println!("alice listening on {addr}; bob connecting");

            let alice_auth = auth_key.clone();
            let alice_task = tokio::spawn(async move {
                let (stream, _) = listener.accept().await?;
                run_alice(stream, &alice_auth, session).await
            });
            let bob_auth = auth_key.clone();
            let bob_task = tokio::spawn(async move {
                let stream = TcpStream::connect(addr).await?;
                run_bob(stream, &bob_auth).await
            });

            let (alice_key, alice_stats) = alice_task.await??;
            let (bob_key, bob_stats) = bob_task.await??;

            print_result("alice", &alice_key, &alice_stats);
            print_result("bob", &bob_key, &bob_stats);

            assert_eq!(
                alice_key.material, bob_key.material,
                "both ends must derive identical key material"
            );
            println!(
                "key agreement: both ends derived the SAME {}-bit key (id {})",
                alice_key.length_bits, alice_key.key_id
            );

            // Encrypt with Alice's copy, decrypt with Bob's copy.
            aes_round_trip(&alice_key, &bob_key)?;
        }
    }

    Ok(())
}
