//! Two-process P2P QKD integration test.
//!
//! # Process-separation level achieved
//!
//! Full OS-process separation: the parent test re-spawns this same test
//! binary twice via `std::process::Command` with `QKD_TP_ROLE=alice|bob`
//! (the well-known re-entrant test-binary pattern, chosen because cargo
//! does not expose example binaries' paths to integration tests the way
//! `CARGO_BIN_EXE_*` does for `[[bin]]` targets). Each child runs one P2P
//! endpoint in its own process with its own tokio runtime and address
//! space; the only connection between them is localhost TCP carrying the
//! HMAC-authenticated classical channel. The parent verifies that both
//! processes print the same key id and key material, and that an AES-GCM
//! message encrypted with Alice's process's key decrypts with Bob's
//! process's key.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

use qkd_core::channel::ChannelConfig;
use qkd_core::types::{ProtocolType, SecureKey};
use qkd_network::p2p::{run_alice, run_bob, P2pSessionConfig};
use qkd_network::secure_channel::SecureChannel;

const AUTH_KEY: &[u8] = b"two-process-test-preshared-auth-key";

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn session_config() -> P2pSessionConfig {
    P2pSessionConfig {
        num_qubits: 8192,
        channel: ChannelConfig {
            noise_rate: 0.01,
            loss_rate: 0.05,
            dark_count_rate: 0.0,
            eavesdropper: None,
        },
        ..P2pSessionConfig::for_protocol(ProtocolType::BB84)
    }
}

/// Child entry point. A no-op (instantly passing test) unless QKD_TP_ROLE
/// is set, in which case this process becomes one of the two QKD endpoints.
#[test]
fn two_process_child() {
    let Ok(role) = std::env::var("QKD_TP_ROLE") else {
        return;
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let (key, stats) = match role.as_str() {
            "alice" => {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                // Tell the parent where to point Bob.
                println!("TPLISTEN {}", listener.local_addr().unwrap());
                std::io::stdout().flush().unwrap();
                let (stream, _) = listener.accept().await.unwrap();
                run_alice(stream, AUTH_KEY, session_config()).await.unwrap()
            }
            "bob" => {
                let addr = std::env::var("QKD_TP_ADDR").unwrap();
                let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
                run_bob(stream, AUTH_KEY).await.unwrap()
            }
            other => panic!("unknown QKD_TP_ROLE: {other}"),
        };
        println!(
            "TPRESULT role={role} key_id={} qber={:.4} leaked_bits={} key_hex={}",
            key.key_id,
            stats.qber,
            stats.leaked_bits,
            hex_encode(&key.material)
        );
        std::io::stdout().flush().unwrap();
    });
}

fn spawn_child(role: &str, addr: Option<&str>) -> Child {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.args(["two_process_child", "--exact", "--nocapture"])
        .env("QKD_TP_ROLE", role)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(addr) = addr {
        cmd.env("QKD_TP_ADDR", addr);
    }
    cmd.spawn().unwrap()
}

/// Parse the `TPRESULT` line out of a child's stdout.
fn parse_result(stdout: &str) -> (String, String, usize, Vec<u8>) {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("TPRESULT "))
        .unwrap_or_else(|| panic!("no TPRESULT line in child output:\n{stdout}"));
    let mut key_id = None;
    let mut qber = None;
    let mut leaked = None;
    let mut material = None;
    for field in line.trim_start_matches("TPRESULT ").split_whitespace() {
        if let Some(v) = field.strip_prefix("key_id=") {
            key_id = Some(v.to_string());
        } else if let Some(v) = field.strip_prefix("qber=") {
            qber = Some(v.to_string());
        } else if let Some(v) = field.strip_prefix("leaked_bits=") {
            leaked = Some(v.parse::<usize>().unwrap());
        } else if let Some(v) = field.strip_prefix("key_hex=") {
            material = Some(hex_decode(v));
        }
    }
    (
        key_id.expect("key_id"),
        qber.expect("qber"),
        leaked.expect("leaked_bits"),
        material.expect("key_hex"),
    )
}

#[test]
fn two_process_key_agreement_over_tcp() {
    // Don't recurse when running as a child.
    if std::env::var("QKD_TP_ROLE").is_ok() {
        return;
    }

    // Spawn Alice and read her listening address off stdout.
    let mut alice = spawn_child("alice", None);
    let mut alice_reader = BufReader::new(alice.stdout.take().unwrap());
    let addr = loop {
        let mut line = String::new();
        let n = alice_reader.read_line(&mut line).unwrap();
        assert!(n > 0, "alice exited before printing TPLISTEN");
        if let Some(rest) = line.trim().strip_prefix("TPLISTEN ") {
            break rest.to_string();
        }
    };

    // Spawn Bob pointed at Alice; collect both outputs.
    let bob = spawn_child("bob", Some(&addr));
    let bob_output = bob.wait_with_output().unwrap();
    assert!(bob_output.status.success(), "bob process failed");

    let mut alice_rest = String::new();
    std::io::Read::read_to_string(&mut alice_reader, &mut alice_rest).unwrap();
    let alice_status = alice.wait().unwrap();
    assert!(alice_status.success(), "alice process failed");

    let (alice_id, alice_qber, alice_leaked, alice_material) = parse_result(&alice_rest);
    let (bob_id, bob_qber, bob_leaked, bob_material) =
        parse_result(&String::from_utf8_lossy(&bob_output.stdout));

    // Both OS processes must agree on key id, leakage accounting, QBER,
    // and — critically — the key material itself.
    assert_eq!(alice_id, bob_id, "key ids differ across processes");
    assert_eq!(alice_qber, bob_qber, "QBER views differ across processes");
    assert_eq!(alice_leaked, bob_leaked, "leakage accounting differs");
    assert_eq!(
        alice_material, bob_material,
        "key material differs across processes"
    );
    assert!(
        alice_material.len() >= 32,
        "key too short for AES-256: {} bytes",
        alice_material.len()
    );

    // AES-GCM round trip across the two independently derived keys:
    // encrypt with the key from Alice's process, decrypt with Bob's.
    let mk_key = |material: Vec<u8>, id: &str| SecureKey {
        key_id: id.to_string(),
        timestamp: chrono::Utc::now(),
        length_bits: material.len() * 8,
        material,
    };
    let alice_key = mk_key(alice_material, &alice_id);
    let bob_key = mk_key(bob_material, &bob_id);

    let sender = SecureChannel::from_key(&alice_key).unwrap();
    let receiver = SecureChannel::from_key(&bob_key).unwrap();
    let plaintext = b"cross-process QKD-derived secure channel";
    let envelope = sender.encrypt(plaintext).unwrap();
    let decrypted = receiver.decrypt(&envelope).unwrap();
    assert_eq!(decrypted, plaintext);
}
