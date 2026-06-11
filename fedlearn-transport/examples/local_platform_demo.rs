//! End-to-end local FL platform demo (OpenFL-style flow).
//!
//! Mirrors the upstream Quantumacy OpenFL fork: clients register, fetch the
//! global model, request QKD-backed transport keys, submit encrypted updates,
//! and the aggregator advances the round.
//!
//! By default the demo generates an ephemeral CA at startup and runs the
//! whole flow over mutual TLS with one certificate per client. Pass
//! `--insecure` for the old plaintext behavior.
//!
//! Configurable via CLI flags or environment variables:
//!   --clients <N>       (default 2)              QUANTUMACY_CLIENTS
//!   --rounds <R>        (default 3)              QUANTUMACY_ROUNDS
//!   --addr <addr>       (default 127.0.0.1:50061) QUANTUMACY_DEMO_ADDR
//!   --insecure          plaintext gRPC (demo parity with the MVP flow)
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
use fedlearn_transport::tls::{client_tls_config, TlsSettings};
use qkd_core::channel::ChannelConfig;
use qkd_core::types::{ProtocolType, SecureKey};
use qkd_network::secure_channel::SecureChannel;
use qkd_network::server::QkdServer;
use tokio::sync::oneshot;
use tonic::transport::{Channel, Server};

#[derive(Debug, Clone)]
struct DemoConfig {
    addr: SocketAddr,
    clients: usize,
    rounds: u32,
    insecure: bool,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:50061".parse().unwrap(),
            clients: 2,
            rounds: 3,
            insecure: false,
        }
    }
}

/// Ephemeral PKI for the demo: a CA, a server identity for `localhost`, and
/// per-client certificates. Generated fresh on every run; nothing persists.
struct DemoPki {
    ca_pem: String,
    server_cert_pem: String,
    server_key_pem: String,
    ca_cert: rcgen::Certificate,
    ca_key: rcgen::KeyPair,
}

impl DemoPki {
    fn generate() -> Self {
        use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair};

        let ca_key = KeyPair::generate().expect("ca keypair");
        let mut ca_params = CertificateParams::new(Vec::new()).expect("ca params");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "quantumacy demo ca");
        let ca_cert = ca_params.self_signed(&ca_key).expect("self-sign ca");

        let server_key = KeyPair::generate().expect("server keypair");
        let mut server_params =
            CertificateParams::new(vec!["localhost".to_string()]).expect("server params");
        server_params
            .distinguished_name
            .push(DnType::CommonName, "localhost");
        let server_cert = server_params
            .signed_by(&server_key, &ca_cert, &ca_key)
            .expect("sign server cert");

        Self {
            ca_pem: ca_cert.pem(),
            server_cert_pem: server_cert.pem(),
            server_key_pem: server_key.serialize_pem(),
            ca_cert,
            ca_key,
        }
    }

    fn client_cert(&self, cn: &str) -> (String, String) {
        use rcgen::{CertificateParams, DnType, KeyPair};

        let key = KeyPair::generate().expect("client keypair");
        let mut params = CertificateParams::new(Vec::new()).expect("client params");
        params.distinguished_name.push(DnType::CommonName, cn);
        let cert = params
            .signed_by(&key, &self.ca_cert, &self.ca_key)
            .expect("sign client cert");
        (cert.pem(), key.serialize_pem())
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
            "--insecure" => {
                cfg.insecure = true;
            }
            "--help" | "-h" => {
                println!(
                    "local_platform_demo:\n  --addr <ip:port>\n  --clients <N>\n  --rounds <R>\n  --insecure"
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

    let pki = (!cfg.insecure).then(DemoPki::generate);

    let addr = cfg.addr;
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let mut builder = Server::builder();
    if let Some(pki) = &pki {
        let tls = TlsSettings::Mutual {
            cert_pem: pki.server_cert_pem.clone().into_bytes(),
            key_pem: pki.server_key_pem.clone().into_bytes(),
            client_ca_pem: pki.ca_pem.clone().into_bytes(),
        };
        builder = builder.tls_config(
            tls.server_tls_config()
                .expect("mutual settings always produce a server config"),
        )?;
    }
    let server_task = tokio::spawn(async move {
        builder
            .add_service(FederatedLearningServer::new(fl_service))
            .add_service(KeyExchangeServer::new(key_service))
            .serve_with_shutdown(addr, async {
                let _ = shutdown_rx.await;
            })
            .await
    });

    tokio::time::sleep(Duration::from_millis(250)).await;

    println!("=== Quantumacy local platform demo ===");
    println!(
        "addr={}  clients={}  rounds={}  transport={}",
        cfg.addr,
        cfg.clients,
        cfg.rounds,
        if cfg.insecure {
            "PLAINTEXT (--insecure)"
        } else {
            "mutual TLS (ephemeral demo CA)"
        }
    );

    // Phase 1: each client connects on its own channel (own certificate
    // under mTLS — the CN is the identity), registers, and keeps its stubs.
    struct DemoClient {
        client_id: String,
        session_id: String,
        fl: FederatedLearningClient<Channel>,
        keys: KeyExchangeClient<Channel>,
    }

    let mut clients: Vec<DemoClient> = Vec::with_capacity(cfg.clients);
    for idx in 0..cfg.clients {
        let client_id = format!("client-{idx:02}");

        let channel = match &pki {
            Some(pki) => {
                let (cert, key) = pki.client_cert(&client_id);
                let tls = client_tls_config(
                    pki.ca_pem.as_bytes(),
                    cert.as_bytes(),
                    key.as_bytes(),
                    "localhost",
                );
                Channel::from_shared(format!("https://{addr}"))?
                    .tls_config(tls)?
                    .connect()
                    .await?
            }
            None => {
                Channel::from_shared(format!("http://{addr}"))?
                    .connect()
                    .await?
            }
        };

        let mut fl = FederatedLearningClient::new(channel.clone());
        let resp = fl
            .register(RegisterRequest {
                client_id: client_id.clone(),
                dataset_size: 100 + idx as u64 * 50,
                capabilities: format!("{{\"cpu\":{}}}", 4 + idx),
            })
            .await?
            .into_inner();

        clients.push(DemoClient {
            client_id,
            session_id: resp.session_id,
            fl,
            keys: KeyExchangeClient::new(channel),
        });
    }

    let first_session = clients[0].session_id.clone();
    let first_client_id = clients[0].client_id.clone();
    let initial_model = clients[0]
        .fl
        .get_global_model(ModelRequest {
            session_id: first_session,
            client_id: first_client_id,
            round: 0,
        })
        .await?
        .into_inner();
    let initial_weights: ModelWeights = serde_json::from_slice(&initial_model.model_weights)?;
    println!("initial weights: {:?}", initial_weights.flatten());

    // Phase 2: per-round encrypted submissions.
    for round in 0..cfg.rounds {
        for (client_idx, client) in clients.iter_mut().enumerate() {
            let client_id = client.client_id.clone();
            let session_id = client.session_id.clone();

            let key = client
                .keys
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
                &client_id,
                synthetic_update(client_idx, round),
                round,
                100 + client_idx * 50,
            ))?)?;

            let response = client
                .fl
                .submit_update(UpdateRequest {
                    session_id,
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

        let status_session = clients[0].session_id.clone();
        let status = clients[0]
            .fl
            .get_status(StatusRequest {
                session_id: status_session,
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
