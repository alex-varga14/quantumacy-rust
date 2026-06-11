use std::sync::Arc;

use fedlearn_core::model::{LayerWeights, ModelUpdate, ModelWeights, UpdateMetadata};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_server::FederatedLearning;
use fedlearn_transport::proto::key_exchange_server::KeyExchange;
use fedlearn_transport::proto::{
    KeyRequest, ModelRequest, RegisterRequest, StatusRequest, SubscribeRequest, UpdateRequest,
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

    let payload_a = serde_json::to_vec(
        &fl_a
            .encrypt_update(&make_update("alice", vec![1.0, 2.0], 0))
            .unwrap(),
    )
    .unwrap();
    let payload_b = serde_json::to_vec(
        &fl_b
            .encrypt_update(&make_update("bob", vec![3.0, 4.0], 0))
            .unwrap(),
    )
    .unwrap();

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

#[tokio::test]
async fn derived_keys_never_expose_raw_qkd_material() {
    let aggregation = Arc::new(AggregationService::new(
        make_weights(vec![0.0, 0.0]),
        TrainingConfig::default(),
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation, qkd_server.clone());
    let key_service = fl_service.key_exchange_service();

    let reg = fl_service
        .register(Request::new(RegisterRequest {
            client_id: "alice".to_string(),
            dataset_size: 100,
            capabilities: "{}".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();

    let key = key_service
        .request_key(Request::new(KeyRequest {
            client_id: "alice".to_string(),
            session_id: reg.session_id.clone(),
            key_bits: 256,
        }))
        .await
        .unwrap()
        .into_inner();

    let raw = qkd_server.get_key(&key.key_id).unwrap();

    assert_eq!(key.key_material.len(), 32, "derived keys are 32 bytes");
    assert_ne!(
        key.key_material[..],
        raw.material[..32],
        "response material must not be the raw QKD key"
    );
    assert_eq!(key.derivation, "hkdf-sha256-v1");

    // The encrypted round-trip still works: the server re-derives the same
    // per-round key when decrypting the submission.
    let channel = SecureChannel::from_key(&secure_key_from_parts(
        key.key_id.clone(),
        key.key_material.clone(),
    ))
    .unwrap();
    let fl_channel = SecureFLChannel::new(channel, key.key_id.clone());
    let payload = serde_json::to_vec(
        &fl_channel
            .encrypt_update(&make_update("alice", vec![1.0, 2.0], key.round))
            .unwrap(),
    )
    .unwrap();

    let submit = fl_service
        .submit_update(Request::new(UpdateRequest {
            session_id: reg.session_id,
            client_id: "alice".to_string(),
            round: key.round,
            model_update: payload,
            key_id: key.key_id,
            encrypted: true,
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(
        submit.accepted,
        "round-trip with derived key: {}",
        submit.message
    );
}

#[tokio::test]
async fn subscribe_rounds_observes_round_transition() {
    use tokio_stream::StreamExt;

    let aggregation = Arc::new(AggregationService::new(
        make_weights(vec![0.0]),
        TrainingConfig::default(),
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation.clone(), qkd_server);

    let reg = fl_service
        .register(Request::new(RegisterRequest {
            client_id: "observer".to_string(),
            dataset_size: 0,
            capabilities: "{}".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();

    // Open the subscription stream first; the snapshot frame should arrive
    // immediately, followed by round-transition events from try_aggregate.
    let mut stream = fl_service
        .subscribe_rounds(Request::new(SubscribeRequest {
            session_id: reg.session_id.clone(),
            client_id: "observer".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();

    let snapshot = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .expect("snapshot frame timed out")
        .expect("stream closed before snapshot")
        .unwrap();
    assert_eq!(snapshot.round, 0);
    assert_eq!(snapshot.action, "train");

    // Drive an aggregation by registering two more clients and submitting.
    aggregation.register_client("c1".to_string(), 100);
    aggregation.register_client("c2".to_string(), 100);
    aggregation
        .submit_update(make_update("c1", vec![1.0], 0))
        .unwrap();
    aggregation
        .submit_update(make_update("c2", vec![3.0], 0))
        .unwrap();
    aggregation.try_aggregate().unwrap();

    let event = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .expect("subscriber timed out waiting for round event")
        .expect("stream closed before broadcast")
        .unwrap();
    assert_eq!(event.round, 1);
    assert_eq!(event.action, "train");
    assert!(!event.payload.is_empty());
}
