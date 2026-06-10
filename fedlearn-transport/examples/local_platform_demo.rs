//! End-to-end local FL platform demo (OpenFL-style flow).
//!
//! Mirrors the upstream Quantumacy OpenFL fork: clients register, fetch the
//! global model, request QKD-backed transport keys, submit encrypted updates,
//! and the aggregator advances the round.
//!
//! Configurable via CLI flags or environment variables:
//!   --clients <N>       (default 2)              QUANTUMACY_CLIENTS
//!   --rounds <R>        (default 3)              QUANTUMACY_ROUNDS
//!   --addr <addr>       (default 127.0.0.1:50061) QUANTUMACY_DEMO_ADDR
//!
//! Usage:
//!   cargo run -p fedlearn-transport --example local_platform_demo
//!   cargo run -p fedlearn-transport --example local_platform_demo -- --clients 4 --rounds 5

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use fedlearn_core::aggregation::FedAvgConfig;
use fedlearn_core::model::{LayerWeights, ModelUpdate, ModelWeights, UpdateMetadata};
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
use tokio::sync::oneshot;
use tonic::transport::Server;

#[derive(Debug, Clone)]
struct DemoConfig {
    addr: SocketAddr,
    clients: usize,
    rounds: u32,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:50061".parse().unwrap(),
            clients: 2,
            rounds: 3,
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
            "--help" | "-h" => {
                println!(
                    "local_platform_demo:\n  --addr <ip:port>\n  --clients <N>\n  --rounds <R>"
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

fn make_weights(vals: Vec<f32>) -> ModelWeights {
    let n = vals.len();
    ModelWeights {
        layers: vec![LayerWeights {
            name: "dense".to_string(),
            shape: vec![n],
            data: vals,
        }],
        num_params: n,
    }
}

fn make_update(client_id: &str, vals: Vec<f32>, round: u32, num_samples: usize) -> ModelUpdate {
    let n = vals.len();
    ModelUpdate {
        client_id: client_id.to_string(),
        weights: ModelWeights {
            layers: vec![LayerWeights {
                name: "dense".to_string(),
                shape: vec![n],
                data: vals,
            }],
            num_params: n,
        },
        num_samples,
        loss: 0.25,
        metadata: UpdateMetadata {
            local_epochs: 1,
            learning_rate: 0.01,
            batch_size: 16,
            round,
            training_time_ms: 25,
        },
    }
}

fn secure_key_from_parts(key_id: String, key_material: Vec<u8>) -> SecureKey {
    SecureKey {
        key_id,
        timestamp: chrono::Utc::now(),
        length_bits: key_material.len() * 8,
        material: key_material,
    }
}

/// Synthesise a per-client local update: each client biases toward a different
/// region so the aggregated mean lands somewhere in between, exercising FedAvg.
fn synthetic_update(client_idx: usize, round: u32) -> Vec<f32> {
    let base = client_idx as f32 + 1.0;
    let drift = round as f32 * 0.1;
    vec![base + drift, base * 2.0 + drift]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config()?;

    let aggregation = Arc::new(AggregationService::new(
        make_weights(vec![0.0, 0.0]),
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

    println!("=== Quantumacy local platform demo ===");
    println!(
        "addr={}  clients={}  rounds={}",
        cfg.addr, cfg.clients, cfg.rounds
    );

    // Phase 1: register every client and fetch initial weights.
    let mut sessions = Vec::with_capacity(cfg.clients);
    for idx in 0..cfg.clients {
        let client_id = format!("client-{idx:02}");
        let resp = fl_client
            .register(RegisterRequest {
                client_id: client_id.clone(),
                dataset_size: 100 + idx as u64 * 50,
                capabilities: format!("{{\"cpu\":{}}}", 4 + idx),
            })
            .await?
            .into_inner();
        sessions.push((client_id, resp.session_id));
    }
    let initial_model = fl_client
        .get_global_model(ModelRequest {
            session_id: sessions[0].1.clone(),
            client_id: sessions[0].0.clone(),
            round: 0,
        })
        .await?
        .into_inner();
    let initial_weights: ModelWeights = serde_json::from_slice(&initial_model.model_weights)?;
    println!("initial weights: {:?}", initial_weights.flatten());

    // Phase 2: per-round encrypted submissions.
    for round in 0..cfg.rounds {
        for (client_idx, (client_id, session_id)) in sessions.iter().enumerate() {
            let key = key_client
                .request_key(KeyRequest {
                    client_id: client_id.clone(),
                    session_id: session_id.clone(),
                    key_bits: 256,
                })
                .await?
                .into_inner();

            let secure_key = secure_key_from_parts(key.key_id.clone(), key.key_material.clone());
            let channel = SecureChannel::from_key(&secure_key)?;
            let fl_channel = SecureFLChannel::new(channel, key.key_id.clone());

            let payload = serde_json::to_vec(&fl_channel.encrypt_update(&make_update(
                client_id,
                synthetic_update(client_idx, round),
                round,
                100 + client_idx * 50,
            ))?)?;

            let response = fl_client
                .submit_update(UpdateRequest {
                    session_id: session_id.clone(),
                    client_id: client_id.clone(),
                    round,
                    model_update: payload,
                    key_id: key.key_id,
                    encrypted: true,
                })
                .await?
                .into_inner();

            if !response.accepted {
                eprintln!(
                    "warning: client {client_id} round {round} rejected: {}",
                    response.message
                );
            }
        }

        let status = fl_client
            .get_status(StatusRequest {
                session_id: sessions[0].1.clone(),
            })
            .await?
            .into_inner();
        println!(
            "round {:>3}/{:<3} | clients={} | state={:<12} | loss={:.4} | acc={:.4} | weights={:?}",
            status.current_round,
            status.total_rounds,
            status.participating_clients,
            status.state,
            status.global_loss,
            status.global_accuracy,
            aggregation.get_global_weights().flatten()
        );
    }

    println!(
        "\nfinal aggregated weights: {:?}",
        aggregation.get_global_weights().flatten()
    );
    println!("rounds completed: {}", aggregation.current_round());

    let _ = shutdown_tx.send(());
    server_task.await??;
    Ok(())
}
