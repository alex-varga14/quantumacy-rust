//! QKD client-server demo.
//!
//! Mirrors the client-server side of QKDSimkit: a `QkdServer` mints keys
//! using the configured protocol, distributes them to "clients", and the
//! clients use the keys to seal a message via AES-GCM.
//!
//! Usage:
//!   cargo run -p qkd-network --example qkd_client_server_demo
//!   cargo run -p qkd-network --example qkd_client_server_demo -- --protocol six-state --qubits 8192
//!
//! Environment overrides (CLI flags win):
//!   QUANTUMACY_PROTOCOL=bb84|six-state|b92
//!   QUANTUMACY_QUBITS=<usize>
//!   QUANTUMACY_NOISE=<f64>

use std::env;
use std::sync::Arc;

use qkd_core::channel::ChannelConfig;
use qkd_core::types::ProtocolType;
use qkd_network::secure_channel::SecureChannel;
use qkd_network::server::QkdServer;

#[derive(Debug, Clone)]
struct DemoConfig {
    protocol: ProtocolType,
    num_qubits: usize,
    noise_rate: f64,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            protocol: ProtocolType::BB84,
            num_qubits: 4096,
            noise_rate: 0.02,
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
            "--help" | "-h" => {
                println!(
                    "qkd_client_server_demo:\n  --protocol bb84|six-state|b92\n  --qubits <usize>\n  --noise <f64>"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }

    cfg
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config();

    let channel = ChannelConfig {
        noise_rate: cfg.noise_rate,
        ..ChannelConfig::default()
    };

    println!("=== Quantumacy QKD Client-Server Demo ===");
    println!(
        "protocol={:?}  qubits={}  noise={:.3}",
        cfg.protocol, cfg.num_qubits, cfg.noise_rate
    );

    let server = Arc::new(QkdServer::new(channel, cfg.protocol));

    let session_alice = server.create_session().await;
    let session_bob = server.create_session().await;
    println!(
        "server: created sessions alice={} bob={}",
        session_alice, session_bob
    );

    let key_id_alice = server.generate_key(cfg.num_qubits).await?;
    let key_id_bob = server.generate_key(cfg.num_qubits).await?;
    println!(
        "server: minted keys alice={} bob={} (available={})",
        key_id_alice,
        key_id_bob,
        server.available_keys()
    );

    // Clients fetch their keys from the server and open a secure channel.
    let key_alice = server.get_key(&key_id_alice)?;
    let key_bob = server.get_key(&key_id_bob)?;
    println!(
        "alice: received key ({} bits)   bob: received key ({} bits)",
        key_alice.length_bits, key_bob.length_bits
    );

    let alice_channel = SecureChannel::from_key(&key_alice)?;
    let bob_channel = SecureChannel::from_key(&key_bob)?;

    let alice_msg = b"alice -> aggregator: encrypted gradient bundle";
    let alice_envelope = alice_channel.encrypt(alice_msg)?;
    let alice_decrypted = alice_channel.decrypt(&alice_envelope)?;
    assert_eq!(alice_decrypted, alice_msg);

    let bob_msg = b"bob -> aggregator: encrypted gradient bundle";
    let bob_envelope = bob_channel.encrypt(bob_msg)?;
    let bob_decrypted = bob_channel.decrypt(&bob_envelope)?;
    assert_eq!(bob_decrypted, bob_msg);

    println!(
        "alice: sealed {} -> {} bytes; bob: sealed {} -> {} bytes",
        alice_msg.len(),
        alice_envelope.ciphertext.len(),
        bob_msg.len(),
        bob_envelope.ciphertext.len()
    );

    Ok(())
}
