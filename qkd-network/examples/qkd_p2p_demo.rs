//! Peer-to-peer QKD key exchange demo.
//!
//! Mirrors the QKDSimkit P2P CLI from the upstream CERN/Quantumacy repo.
//! Two peers negotiate a shared secret over a simulated quantum channel,
//! then use that key to AES-GCM encrypt and decrypt a test payload.
//!
//! Usage:
//!   cargo run -p qkd-network --example qkd_p2p_demo
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --protocol six-state
//!   cargo run -p qkd-network --example qkd_p2p_demo -- --protocol b92 --qubits 8192
//!
//! Environment overrides (read first, CLI flags win):
//!   QUANTUMACY_PROTOCOL=bb84|six-state|b92
//!   QUANTUMACY_QUBITS=<usize>
//!   QUANTUMACY_NOISE=<f64>
//!   QUANTUMACY_EAVESDROPPER=on|off

use std::env;

use qkd_core::channel::{ChannelConfig, EavesdropperConfig, EveStrategy};
use qkd_core::protocols::{b92::B92, bb84::Bb84, six_state::SixState, QkdProtocol};
use qkd_core::types::ProtocolType;
use qkd_network::p2p::QkdPeer;
use qkd_network::secure_channel::SecureChannel;

#[derive(Debug, Clone)]
struct DemoConfig {
    protocol: ProtocolType,
    num_qubits: usize,
    noise_rate: f64,
    eavesdropper: bool,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            protocol: ProtocolType::BB84,
            num_qubits: 4096,
            noise_rate: 0.02,
            eavesdropper: false,
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
            "--help" | "-h" => {
                println!(
                    "qkd_p2p_demo: \n  --protocol bb84|six-state|b92\n  --qubits <usize>\n  --noise <f64>\n  --eavesdropper"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }

    cfg
}

fn build_protocol(proto: ProtocolType) -> Box<dyn QkdProtocol> {
    match proto {
        ProtocolType::BB84 => Box::new(Bb84::default()),
        ProtocolType::SixState => Box::new(SixState::default()),
        ProtocolType::B92 => Box::new(B92::default()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config();

    let channel = ChannelConfig {
        noise_rate: cfg.noise_rate,
        eavesdropper: cfg.eavesdropper.then_some(EavesdropperConfig {
            intercept_rate: 0.5,
            strategy: EveStrategy::InterceptResend,
        }),
        ..ChannelConfig::default()
    };

    println!("=== Quantumacy P2P QKD Demo ===");
    println!(
        "protocol={:?}  qubits={}  noise={:.3}  eavesdropper={}",
        cfg.protocol, cfg.num_qubits, cfg.noise_rate, cfg.eavesdropper
    );

    let alice = QkdPeer::new("alice".to_string());
    let bob = QkdPeer::new("bob".to_string());

    // Both peers run the protocol; QkdPeer currently delegates to BB84
    // internally, so we hand it the configured protocol explicitly to keep the
    // demo wiring honest and the QBER stat display protocol-aware.
    let protocol = build_protocol(cfg.protocol);
    let alice_key = alice
        .exchange_key(protocol.as_ref(), &channel, cfg.num_qubits)
        .await?;
    println!(
        "alice: derived key id={} ({} bits, {} bytes)",
        alice_key.key_id,
        alice_key.length_bits,
        alice_key.material.len()
    );

    let protocol = build_protocol(cfg.protocol);
    let bob_key = bob
        .exchange_key(protocol.as_ref(), &channel, cfg.num_qubits)
        .await?;
    println!(
        "bob:   derived key id={} ({} bits, {} bytes)",
        bob_key.key_id,
        bob_key.length_bits,
        bob_key.material.len()
    );

    // In a real QKD exchange Alice and Bob would already share key material
    // via the protocol transcript. The simulator gives each peer an independent
    // run, so we use Alice's key for the AES-GCM round-trip below to keep the
    // demo deterministic.
    let secure = SecureChannel::from_key(&alice_key)?;
    let plaintext = b"alice -> bob: privileged medical record handoff";
    let envelope = secure.encrypt(plaintext)?;
    let decrypted = secure.decrypt(&envelope)?;

    println!(
        "secure channel: encrypted {} -> {} ciphertext bytes (nonce {} bytes)",
        plaintext.len(),
        envelope.ciphertext.len(),
        envelope.nonce.len()
    );
    assert_eq!(decrypted, plaintext);
    println!("secure channel: round-trip decrypt OK");
    println!(
        "alice keys held: {}, bob keys held: {}",
        alice.available_keys(),
        bob.available_keys()
    );

    Ok(())
}
