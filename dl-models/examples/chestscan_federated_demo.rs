//! Chest-scan federated learning demo.
//!
//! Mirrors the upstream Quantumacy chestscan use case: multiple hospital
//! clients train a `ChestScanModel` locally on a synthetic chest-X-ray
//! dataset and submit QKD-encrypted updates to the FedAvg aggregator
//! over `fedlearn-transport`.
//!
//! Usage:
//!   cargo run -p dl-models --example chestscan_federated_demo
//!   cargo run -p dl-models --example chestscan_federated_demo -- --clients 3 --rounds 5
//!
//! Environment overrides (CLI flags win):
//!   QUANTUMACY_CLIENTS=<usize>
//!   QUANTUMACY_ROUNDS=<u32>
//!   QUANTUMACY_DEMO_ADDR=<ip:port>

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use dl_models::chestscan::{ChestScanConfig, ChestScanModel};
use fedlearn_core::aggregation::FedAvgConfig;
use fedlearn_core::model::{FederatedModel, LocalTrainConfig, ModelWeights};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_client::FederatedLearningClient;
use fedlearn_transport::proto::federated_learning_server::FederatedLearningServer;
use fedlearn_transport::proto::key_exchange_client::KeyExchangeClient;
use fedlearn_transport::proto::key_exchange_server::KeyExchangeServer;
use fedlearn_transport::proto::{
    KeyRequest, ModelRequest, RegisterRequest, StatusRequest, UpdateRequest,
};
use fedlearn_transport::secure_channel::SecureFLChannel;
use qkd_core::channel::ChannelConfig;
use qkd_core::types::{ProtocolType, SecureKey};
use qkd_network::secure_channel::SecureChannel;
use qkd_network::server::QkdServer;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use tokio::sync::oneshot;
use tonic::transport::Server;

const IMG_W: usize = 4;
const IMG_H: usize = 4;
const HIDDEN_DIM: usize = 8;

#[derive(Debug, Clone)]
struct DemoConfig {
    addr: SocketAddr,
    clients: usize,
    rounds: u32,
    samples_per_client: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:50071".parse().unwrap(),
            clients: 2,
            rounds: 4,
            samples_per_client: 32,
        }
    }
}

fn load_config() -> Result<DemoConfig, Box<dyn std::error::Error>> {
    let mut cfg = DemoConfig::default();

    if let Ok(addr) = env::var("QUANTUMACY_DEMO_ADDR") {
        cfg.addr = addr.parse()?;
    }
    if let Ok(c) = env::var("QUANTUMACY_CLIENTS") {
        cfg.clients = c.parse()?;
    }
    if let Ok(r) = env::var("QUANTUMACY_ROUNDS") {
        cfg.rounds = r.parse()?;
    }

    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--addr" => {
                if let Some(v) = args.next() {
                    cfg.addr = v.parse()?;
                }
            }
            "--clients" => {
                if let Some(v) = args.next() {
                    cfg.clients = v.parse()?;
                }
            }
            "--rounds" => {
                if let Some(v) = args.next() {
                    cfg.rounds = v.parse()?;
                }
            }
            "--samples" => {
                if let Some(v) = args.next() {
                    cfg.samples_per_client = v.parse()?;
                }
            }
            "--help" | "-h" => {
                println!(
                    "chestscan_federated_demo:\n  --addr <ip:port>\n  --clients <N>\n  --rounds <R>\n  --samples <S>"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }

    if cfg.clients < 2 {
        return Err("--clients must be >= 2 for FedAvg aggregation".into());
    }

    Ok(cfg)
}

/// Synthetic chest-scan dataset: half the samples are "abnormal" (high pixel
/// intensity), half are "normal" (low intensity). Gives the model a learnable
/// signal without needing the real DICOM corpus.
fn synthesize_client_dataset(seed: u64, samples: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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

fn secure_key_from_parts(key_id: String, key_material: Vec<u8>) -> SecureKey {
    SecureKey {
        key_id,
        timestamp: chrono::Utc::now(),
        length_bits: key_material.len() * 8,
        material: key_material,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config()?;

    let scan_config = ChestScanConfig {
        image_width: IMG_W,
        image_height: IMG_H,
        hidden_dim: HIDDEN_DIM,
        seed: 1234,
    };

    // Use the bootstrap model's weights to seed the global aggregator state
    // so client updates align with the agreed-upon shape.
    let bootstrap_model = ChestScanModel::new("bootstrap", scan_config.clone());
    let initial_weights: ModelWeights = bootstrap_model.get_weights()?;

    let aggregation = Arc::new(AggregationService::new(
        initial_weights.clone(),
        TrainingConfig {
            num_rounds: cfg.rounds,
            fedavg: FedAvgConfig {
                min_clients: cfg.clients,
                ..FedAvgConfig::default()
            },
            ..TrainingConfig::default()
        },
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation.clone(), qkd_server);
    let key_service = fl_service.key_exchange_service();

    let addr = cfg.addr;
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_task = tokio::spawn(async move {
        Server::builder()
            .add_service(FederatedLearningServer::new(fl_service))
            .add_service(KeyExchangeServer::new(key_service))
            .serve_with_shutdown(addr, async {
                let _ = shutdown_rx.await;
            })
            .await
    });

    tokio::time::sleep(Duration::from_millis(250)).await;

    let endpoint = format!("http://{addr}");
    let mut fl_client = FederatedLearningClient::connect(endpoint.clone()).await?;
    let mut key_client = KeyExchangeClient::connect(endpoint).await?;

    println!("=== Quantumacy chest-scan federated demo ===");
    println!(
        "addr={}  clients={}  rounds={}  samples/client={}",
        cfg.addr, cfg.clients, cfg.rounds, cfg.samples_per_client
    );
    println!(
        "model: image={}x{}  hidden_dim={}  params={}",
        IMG_W,
        IMG_H,
        HIDDEN_DIM,
        bootstrap_model.num_params()
    );

    // Each client gets its own model + dataset.
    struct ClientCtx {
        client_id: String,
        session_id: String,
        model: ChestScanModel,
        train: (Vec<Vec<f32>>, Vec<Vec<f32>>),
        eval: (Vec<Vec<f32>>, Vec<Vec<f32>>),
    }

    let mut clients: Vec<ClientCtx> = Vec::with_capacity(cfg.clients);
    for idx in 0..cfg.clients {
        let client_id = format!("hospital-{idx:02}");
        let resp = fl_client
            .register(RegisterRequest {
                client_id: client_id.clone(),
                dataset_size: cfg.samples_per_client as u64,
                capabilities: format!("{{\"site\":\"hospital-{idx}\"}}"),
            })
            .await?
            .into_inner();

        let train = synthesize_client_dataset(2000 + idx as u64, cfg.samples_per_client);
        let eval = synthesize_client_dataset(9000 + idx as u64, cfg.samples_per_client / 2);

        clients.push(ClientCtx {
            client_id,
            session_id: resp.session_id,
            model: ChestScanModel::new(format!("hospital-{idx:02}"), scan_config.clone()),
            train,
            eval,
        });
    }

    // Pull the global model once so every client starts from the same point.
    let initial_resp = fl_client
        .get_global_model(ModelRequest {
            session_id: clients[0].session_id.clone(),
            client_id: clients[0].client_id.clone(),
            round: 0,
        })
        .await?
        .into_inner();
    let bootstrap_weights: ModelWeights = serde_json::from_slice(&initial_resp.model_weights)?;
    for client in clients.iter_mut() {
        client.model.set_weights(&bootstrap_weights)?;
    }

    let train_cfg = LocalTrainConfig {
        epochs: 8,
        batch_size: 8,
        learning_rate: 0.05,
        round: 0,
    };

    for round in 0..cfg.rounds {
        let mut round_loss = 0.0f64;
        let mut round_acc = 0.0f64;

        // Pull current global weights and broadcast to all clients.
        let global_resp = fl_client
            .get_global_model(ModelRequest {
                session_id: clients[0].session_id.clone(),
                client_id: clients[0].client_id.clone(),
                round,
            })
            .await?
            .into_inner();
        let global_weights: ModelWeights = serde_json::from_slice(&global_resp.model_weights)?;
        for client in clients.iter_mut() {
            client.model.set_weights(&global_weights)?;
        }

        for client in clients.iter_mut() {
            let local_cfg = LocalTrainConfig {
                round,
                ..train_cfg.clone()
            };
            let update = client
                .model
                .train_local(&client.train.0, &client.train.1, &local_cfg)?;

            let (eval_loss, eval_acc) = client.model.evaluate(&client.eval.0, &client.eval.1)?;
            round_loss += eval_loss;
            round_acc += eval_acc;

            let key = key_client
                .request_key(KeyRequest {
                    client_id: client.client_id.clone(),
                    session_id: client.session_id.clone(),
                    key_bits: 256,
                })
                .await?
                .into_inner();
            let secure_key = secure_key_from_parts(key.key_id.clone(), key.key_material.clone());
            let aes = SecureChannel::from_key(&secure_key)?;
            let fl_channel = SecureFLChannel::new(aes, key.key_id.clone());
            let payload = serde_json::to_vec(&fl_channel.encrypt_update(&update)?)?;

            let resp = fl_client
                .submit_update(UpdateRequest {
                    session_id: client.session_id.clone(),
                    client_id: client.client_id.clone(),
                    round,
                    model_update: payload,
                    key_id: key.key_id,
                    encrypted: true,
                })
                .await?
                .into_inner();
            if !resp.accepted {
                eprintln!(
                    "warning: {} round {round} rejected: {}",
                    client.client_id, resp.message
                );
            }
        }

        let status = fl_client
            .get_status(StatusRequest {
                session_id: clients[0].session_id.clone(),
            })
            .await?
            .into_inner();
        println!(
            "round {:>3}/{:<3} | clients={} | state={:<12} | mean_local_loss={:.4} | mean_local_acc={:.4} | server_loss={:.4}",
            status.current_round,
            status.total_rounds,
            status.participating_clients,
            status.state,
            round_loss / cfg.clients as f64,
            round_acc / cfg.clients as f64,
            status.global_loss
        );
    }

    let final_weights = aggregation.get_global_weights();
    let mut final_model = ChestScanModel::new("global", scan_config);
    final_model.set_weights(&final_weights)?;
    let mut final_loss = 0.0f64;
    let mut final_acc = 0.0f64;
    for client in &clients {
        let (loss, acc) = final_model.evaluate(&client.eval.0, &client.eval.1)?;
        final_loss += loss;
        final_acc += acc;
    }
    println!(
        "\nfinal global model on aggregated eval sets: loss={:.4} accuracy={:.4} (params={})",
        final_loss / cfg.clients as f64,
        final_acc / cfg.clients as f64,
        final_model.num_params()
    );

    let _ = shutdown_tx.send(());
    server_task.await??;
    Ok(())
}
