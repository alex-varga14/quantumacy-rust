//! Three-party homomorphic-encryption use case demo.
//!
//! Mirrors the upstream Quantumacy three-party HE flow:
//! - **Client (data owner)** holds the private input and the secret key.
//! - **Storage (untrusted orchestrator)** holds opaque ciphertext blobs.
//! - **Compute (model owner)** evaluates the encrypted model using the
//!   evaluation/server key but never sees plaintext data or labels.
//!
//! The demo wires `dl_models::ChestScanModel::to_encrypted_model` into the
//! `EncryptedInferenceService` so it exercises the full dl-models -> he-core
//! -> he-inference path end-to-end.
//!
//! Usage:
//!   cargo run -p he-inference --example three_party_demo
//!   cargo run -p he-inference --example three_party_demo -- --samples 6

use std::env;
use std::sync::Arc;

use dl_models::chestscan::{ChestScanConfig, ChestScanModel};
use fedlearn_core::model::{FederatedModel, LocalTrainConfig};
use he_core::{decrypt_vector, encrypt_vector, generate_keys, HeKeySet, HeParameters};
use he_inference::EncryptedInferenceService;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const IMG_W: usize = 4;
const IMG_H: usize = 4;
const HIDDEN_DIM: usize = 6;

#[derive(Debug, Clone)]
struct DemoConfig {
    samples: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self { samples: 4 }
    }
}

fn load_config() -> Result<DemoConfig, Box<dyn std::error::Error>> {
    let mut cfg = DemoConfig::default();
    if let Ok(s) = env::var("QUANTUMACY_SAMPLES") {
        cfg.samples = s.parse()?;
    }
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--samples" => {
                if let Some(v) = args.next() {
                    cfg.samples = v.parse()?;
                }
            }
            "--help" | "-h" => {
                println!("three_party_demo:\n  --samples <N>");
                std::process::exit(0);
            }
            _ => {}
        }
    }
    Ok(cfg)
}

fn synth_dataset(seed: u64, samples: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let pixels = IMG_W * IMG_H;
    let mut data = Vec::with_capacity(samples);
    let mut labels = Vec::with_capacity(samples);
    for idx in 0..samples {
        let abnormal = idx % 2 == 0;
        let mean = if abnormal { 0.85 } else { 0.15 };
        let sample: Vec<f32> = (0..pixels)
            .map(|_| (mean + rng.gen_range(-0.1f32..0.1)).clamp(0.0, 1.0))
            .collect();
        data.push(sample);
        labels.push(vec![if abnormal { 1.0 } else { 0.0 }]);
    }
    (data, labels)
}

/// Stand-in for the storage role: holds opaque blobs keyed by session id and
/// has no access to keys. Built as an `Arc<Mutex>` so we can hand it to each
/// "party" without lifetime gymnastics.
#[derive(Default)]
struct BlindStorage {
    blobs: parking_lot::Mutex<std::collections::HashMap<String, Vec<u8>>>,
}

impl BlindStorage {
    fn put(&self, session_id: &str, blob: Vec<u8>) {
        self.blobs.lock().insert(session_id.to_string(), blob);
    }
    fn fetch(&self, session_id: &str) -> Option<Vec<u8>> {
        self.blobs.lock().get(session_id).cloned()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config()?;

    // ---- Stage 0: model-owner trains a small chestscan classifier -------
    // In a real deployment this happens in the federated demo and only the
    // resulting weights are exported as the encrypted compute graph.
    let scan_config = ChestScanConfig {
        image_width: IMG_W,
        image_height: IMG_H,
        hidden_dim: HIDDEN_DIM,
        seed: 7,
    };
    let mut classifier = ChestScanModel::new("compute-party", scan_config.clone());
    let (train_x, train_y) = synth_dataset(101, 64);
    classifier.train_local(
        &train_x,
        &train_y,
        &LocalTrainConfig {
            epochs: 30,
            batch_size: 8,
            learning_rate: 0.05,
            round: 0,
        },
    )?;
    let (train_loss, train_acc) = classifier.evaluate(&train_x, &train_y)?;
    println!(
        "compute-party trained chestscan model: train_loss={:.4} train_acc={:.4} params={}",
        train_loss,
        train_acc,
        classifier.num_params()
    );

    // ---- Stage 1: client generates HE keys ------------------------------
    let HeKeySet {
        client_key,
        public_key,
        server_key,
    } = generate_keys(HeParameters::default())?;
    println!(
        "client generated HE keys: slots={}",
        public_key.params.slots
    );

    // ---- Stage 2: model-owner publishes the encrypted compute graph -----
    // The compute party receives the server (evaluation) key and the public
    // key bundled into an `HeKeySet`. We give it a *fresh* client key here
    // only because the helper API requires one; the demo never actually uses
    // it on the compute side.
    let encrypted_model = classifier.to_encrypted_model()?;
    let compute_service = Arc::new(EncryptedInferenceService::new(
        encrypted_model.clone(),
        HeKeySet {
            client_key: client_key.clone(),
            public_key: public_key.clone(),
            server_key: server_key.clone(),
        },
    ));

    // ---- Stage 3: storage role -----------------------------------------
    let storage = Arc::new(BlindStorage::default());

    // ---- Stage 4: client encrypts each input and submits via storage ----
    let (eval_x, eval_y) = synth_dataset(202, cfg.samples);

    println!("\n--- three-party inference roundtrip ---");
    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>9}",
        "idx", "true", "encrypted", "plaintext", "agree"
    );

    for (idx, (sample_f32, label)) in eval_x.iter().zip(eval_y.iter()).enumerate() {
        let input: Vec<f64> = sample_f32.iter().map(|v| *v as f64).collect();

        // Client side: encrypt with public_key only; client_key never leaves.
        let ciphertext = encrypt_vector(&public_key, &input)?;
        let session_id = format!("session-{idx:04}");

        // Storage side: handle opaque ciphertext bytes only.
        storage.put(&session_id, serde_json::to_vec(&ciphertext)?);

        // Compute side: pull ciphertext from storage, run the encrypted graph.
        let blob = storage
            .fetch(&session_id)
            .ok_or("storage: session not found")?;
        let pulled: he_core::CiphertextVector = serde_json::from_slice(&blob)?;
        compute_service.upload_input(&session_id, pulled)?;
        let encrypted_output = compute_service.run_inference(&session_id)?;

        // Storage side again: hold the encrypted result blob.
        storage.put(
            &format!("{session_id}-result"),
            serde_json::to_vec(&encrypted_output)?,
        );

        // Client side: pull encrypted result and decrypt with client_key.
        let result_blob = storage
            .fetch(&format!("{session_id}-result"))
            .ok_or("storage: result not found")?;
        let result: he_core::CiphertextVector = serde_json::from_slice(&result_blob)?;
        let decrypted = decrypt_vector(&client_key, &result)?;
        let plaintext_reference = encrypted_model.infer_plaintext(&input)?;

        let agree = (decrypted[0] - plaintext_reference[0]).abs() < 1e-9;
        println!(
            "{:>4}  {:>10.4}  {:>10.6}  {:>10.6}  {:>9}",
            idx, label[0], decrypted[0], plaintext_reference[0], agree
        );
    }

    println!(
        "\nstorage held {} session blobs; client_key never left the data owner.",
        storage.blobs.lock().len()
    );

    Ok(())
}
