use std::sync::Arc;

use fedlearn_core::model::{LayerWeights, ModelUpdate, ModelWeights, UpdateMetadata};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_server::FederatedLearning;
use fedlearn_transport::proto::key_exchange_server::KeyExchange;
use fedlearn_transport::proto::{
    KeyRequest, ModelRequest, RegisterRequest, StatusRequest, UpdateRequest,
};
use fedlearn_transport::secure_channel::SecureFLChannel;
use qkd_core::channel::ChannelConfig;
use qkd_core::types::{ProtocolType, SecureKey};
use qkd_network::secure_channel::SecureChannel;
use qkd_network::server::QkdServer;
use tonic::Request;

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

fn make_update(client_id: &str, vals: Vec<f32>, round: u32) -> ModelUpdate {
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
        num_samples: 100,
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

#[tokio::test]
async fn mvp_round_trip_over_public_grpc_api() {
    let aggregation = Arc::new(AggregationService::new(
        make_weights(vec![0.0, 0.0]),
        TrainingConfig::default(),
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation.clone(), qkd_server);
    let key_service = fl_service.key_exchange_service();

    let reg_a = fl_service
        .register(Request::new(RegisterRequest {
            client_id: "alice".to_string(),
            dataset_size: 100,
            capabilities: "{\"cpu\":4}".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();
    let reg_b = fl_service
        .register(Request::new(RegisterRequest {
            client_id: "bob".to_string(),
            dataset_size: 120,
            capabilities: "{\"cpu\":8}".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();

    let model = fl_service
        .get_global_model(Request::new(ModelRequest {
            session_id: reg_a.session_id.clone(),
            client_id: "alice".to_string(),
            round: 0,
        }))
        .await
        .unwrap()
        .into_inner();
    let decoded_weights: ModelWeights = serde_json::from_slice(&model.model_weights).unwrap();
    assert_eq!(decoded_weights.flatten(), vec![0.0, 0.0]);

    let key_a = key_service
        .request_key(Request::new(KeyRequest {
            client_id: "alice".to_string(),
            session_id: reg_a.session_id.clone(),
            key_bits: 256,
        }))
        .await
        .unwrap()
        .into_inner();
    let key_b = key_service
        .request_key(Request::new(KeyRequest {
            client_id: "bob".to_string(),
            session_id: reg_b.session_id.clone(),
            key_bits: 256,
        }))
        .await
        .unwrap()
        .into_inner();

    let channel_a = SecureChannel::from_key(&secure_key_from_parts(
        key_a.key_id.clone(),
        key_a.key_material.clone(),
    ))
    .unwrap();
    let channel_b = SecureChannel::from_key(&secure_key_from_parts(
        key_b.key_id.clone(),
        key_b.key_material.clone(),
    ))
    .unwrap();
    let fl_a = SecureFLChannel::new(channel_a, key_a.key_id.clone());
    let fl_b = SecureFLChannel::new(channel_b, key_b.key_id.clone());

    let payload_a = serde_json::to_vec(&fl_a.encrypt_update(&make_update("alice", vec![1.0, 2.0], 0)).unwrap()).unwrap();
    let payload_b = serde_json::to_vec(&fl_b.encrypt_update(&make_update("bob", vec![3.0, 4.0], 0)).unwrap()).unwrap();

    let submit_a = fl_service
        .submit_update(Request::new(UpdateRequest {
            session_id: reg_a.session_id.clone(),
            client_id: "alice".to_string(),
            round: 0,
            model_update: payload_a,
            key_id: key_a.key_id,
            encrypted: true,
        }))
        .await
        .unwrap()
        .into_inner();
    let submit_b = fl_service
        .submit_update(Request::new(UpdateRequest {
            session_id: reg_b.session_id.clone(),
            client_id: "bob".to_string(),
            round: 0,
            model_update: payload_b,
            key_id: key_b.key_id,
            encrypted: true,
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(submit_a.accepted);
    assert!(submit_b.accepted);

    let status = fl_service
        .get_status(Request::new(StatusRequest {
            session_id: reg_a.session_id,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.current_round, 1);
    assert_eq!(status.participating_clients, 2);
    assert_eq!(aggregation.get_global_weights().flatten(), vec![2.0, 3.0]);
}
