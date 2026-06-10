use std::net::SocketAddr;
use std::sync::Arc;

use fedlearn_core::model::{LayerWeights, ModelWeights};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_server::FederatedLearningServer;
use fedlearn_transport::proto::key_exchange_server::KeyExchangeServer;
use qkd_core::channel::ChannelConfig;
use qkd_core::types::ProtocolType;
use qkd_network::server::QkdServer;
use tonic::transport::Server;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::var("QUANTUMACY_SERVER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
        .parse()?;

    let aggregation = Arc::new(AggregationService::new(
        make_weights(vec![0.0, 0.0]),
        TrainingConfig::default(),
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation, qkd_server);
    let key_service = fl_service.key_exchange_service();

    println!("Quantumacy transport server listening on {addr}");

    Server::builder()
        .add_service(FederatedLearningServer::new(fl_service))
        .add_service(KeyExchangeServer::new(key_service))
        .serve(addr)
        .await?;

    Ok(())
}
