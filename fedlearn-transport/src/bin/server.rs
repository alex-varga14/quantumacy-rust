//! Long-running federated learning + key-exchange gRPC server.
//!
//! Mirrors the inline server setup used by `examples/local_platform_demo.rs`,
//! but exposes it as a deployable binary configured via environment variables:
//!
//! - `QUANTUMACY_BIND_ADDR`  default `127.0.0.1:50051`
//! - `QUANTUMACY_PROTOCOL`   default `bb84` (`bb84` | `six-state` | `b92`)
//! - `QUANTUMACY_KEY_BITS`   default `256`
//! - `QUANTUMACY_ROUNDS`     default `10`
//! - `QUANTUMACY_MIN_CLIENTS` default `2`
//! - `QUANTUMACY_LOG` / `RUST_LOG` standard `tracing-subscriber` filter
//!
//! Transport security (mutual TLS, required unless explicitly disabled):
//! - `QUANTUMACY_TLS_CERT`      server certificate PEM path
//! - `QUANTUMACY_TLS_KEY`       server private key PEM path
//! - `QUANTUMACY_TLS_CLIENT_CA` CA bundle that client certificates must chain to
//! - `QUANTUMACY_INSECURE=1`    serve plaintext (local demos only)
//!
//! Run with:
//!   cargo run -p fedlearn-transport --bin server
//!   QUANTUMACY_BIND_ADDR=0.0.0.0:50051 cargo run -p fedlearn-transport --bin server

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use fedlearn_core::aggregation::FedAvgConfig;
use fedlearn_core::model::{LayerWeights, ModelWeights};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_server::FederatedLearningServer;
use fedlearn_transport::proto::key_exchange_server::KeyExchangeServer;
use fedlearn_transport::tls::TlsSettings;
use qkd_core::channel::ChannelConfig;
use qkd_core::types::ProtocolType;
use qkd_network::server::QkdServer;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone)]
struct ServerConfig {
    bind: SocketAddr,
    protocol: ProtocolType,
    key_bits: u32,
    num_rounds: u32,
    min_clients: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:50051".parse().unwrap(),
            protocol: ProtocolType::BB84,
            key_bits: 256,
            num_rounds: 10,
            min_clients: 2,
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

fn load_config() -> Result<ServerConfig, Box<dyn std::error::Error>> {
    let mut cfg = ServerConfig::default();
    if let Ok(addr) = env::var("QUANTUMACY_BIND_ADDR") {
        cfg.bind = addr.parse()?;
    }
    if let Ok(proto) = env::var("QUANTUMACY_PROTOCOL") {
        if let Some(parsed) = parse_protocol(&proto) {
            cfg.protocol = parsed;
        } else {
            return Err(format!("invalid QUANTUMACY_PROTOCOL: {proto}").into());
        }
    }
    if let Ok(bits) = env::var("QUANTUMACY_KEY_BITS") {
        cfg.key_bits = bits.parse()?;
    }
    if let Ok(rounds) = env::var("QUANTUMACY_ROUNDS") {
        cfg.num_rounds = rounds.parse()?;
    }
    if let Ok(min) = env::var("QUANTUMACY_MIN_CLIENTS") {
        cfg.min_clients = min.parse()?;
    }
    Ok(cfg)
}

fn init_tracing() {
    let filter = env::var("QUANTUMACY_LOG")
        .ok()
        .or_else(|| env::var("RUST_LOG").ok())
        .unwrap_or_else(|| "info,fedlearn_transport=debug,qkd_network=info".to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_target(true)
        .try_init();
}

fn placeholder_weights() -> ModelWeights {
    // Servers come up before any client registers; clients re-broadcast the
    // initial weights they want via SubmitUpdate, so a zero-shaped placeholder
    // is enough for the aggregator to bootstrap.
    ModelWeights {
        layers: vec![LayerWeights {
            name: "placeholder".to_string(),
            shape: vec![0],
            data: vec![],
        }],
        num_params: 0,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let cfg = load_config()?;
    // Fail fast on missing/unreadable TLS material before any service boots.
    let tls_settings = TlsSettings::from_env()?;

    info!(?cfg, "starting Quantumacy federated learning server");

    let aggregation = Arc::new(AggregationService::new(
        placeholder_weights(),
        TrainingConfig {
            num_rounds: cfg.num_rounds,
            fedavg: FedAvgConfig {
                min_clients: cfg.min_clients,
                ..FedAvgConfig::default()
            },
            ..TrainingConfig::default()
        },
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), cfg.protocol));

    let fl_service = GrpcFederatedLearningService::new(aggregation, qkd_server);
    let key_service = fl_service.key_exchange_service();

    info!(addr = %cfg.bind, "binding gRPC server");
    let mut builder = Server::builder();
    if let Some(tls) = tls_settings.server_tls_config() {
        info!("mutual TLS enabled: client certificates required");
        builder = builder.tls_config(tls)?;
    }
    builder
        .add_service(FederatedLearningServer::new(fl_service))
        .add_service(KeyExchangeServer::new(key_service))
        .serve_with_shutdown(cfg.bind, async {
            tokio::signal::ctrl_c().await.ok();
            info!("ctrl-c received, shutting down");
        })
        .await?;

    Ok(())
}
