//! mTLS transport integration tests.

mod common;

use common::TestPki;
use fedlearn_core::model::{LayerWeights, ModelWeights};
use fedlearn_core::round::TrainingConfig;
use fedlearn_transport::grpc_service::{AggregationService, GrpcFederatedLearningService};
use fedlearn_transport::proto::federated_learning_client::FederatedLearningClient;
use fedlearn_transport::proto::RegisterRequest;
use fedlearn_transport::tls::{client_tls_config, serve_with_listener, TlsSettings};
use qkd_core::channel::ChannelConfig;
use qkd_core::types::ProtocolType;
use qkd_network::server::QkdServer;
use std::sync::Arc;
use tonic::transport::Channel;

fn initial_weights() -> ModelWeights {
    ModelWeights {
        layers: vec![LayerWeights {
            name: "dense".to_string(),
            shape: vec![2],
            data: vec![0.0, 0.0],
        }],
        num_params: 2,
    }
}

/// Boot the FL + key-exchange services over the given TLS settings on an
/// ephemeral port; returns the bound port.
async fn spawn_server(settings: TlsSettings) -> u16 {
    let aggregation = Arc::new(AggregationService::new(
        initial_weights(),
        TrainingConfig::default(),
    ));
    let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
    let fl_service = GrpcFederatedLearningService::new(aggregation, qkd_server);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        serve_with_listener(&settings, listener, fl_service)
            .await
            .unwrap();
    });
    // Give the acceptor a beat to start.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    port
}

fn mutual_settings(pki: &TestPki) -> TlsSettings {
    TlsSettings::Mutual {
        cert_pem: pki.server_cert_pem.clone().into_bytes(),
        key_pem: pki.server_key_pem.clone().into_bytes(),
        client_ca_pem: pki.ca_pem.clone().into_bytes(),
    }
}

async fn tls_client(
    pki: &TestPki,
    cert_pem: &str,
    key_pem: &str,
    port: u16,
) -> Result<FederatedLearningClient<Channel>, tonic::transport::Error> {
    let tls = client_tls_config(
        pki.ca_pem.as_bytes(),
        cert_pem.as_bytes(),
        key_pem.as_bytes(),
        "localhost",
    );
    let channel = Channel::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(tls)?
        .connect()
        .await?;
    Ok(FederatedLearningClient::new(channel))
}

fn register_req(client_id: &str) -> RegisterRequest {
    RegisterRequest {
        client_id: client_id.to_string(),
        dataset_size: 100,
        capabilities: "{}".to_string(),
    }
}

#[tokio::test]
async fn test_ca_signed_client_connects_and_registers() {
    let pki = TestPki::generate();
    let port = spawn_server(mutual_settings(&pki)).await;

    let (cert, key) = pki.client_cert("alice");
    let mut client = tls_client(&pki, &cert, &key, port).await.unwrap();

    let response = client.register(register_req("alice")).await.unwrap();
    assert!(!response.into_inner().session_id.is_empty());
}

#[tokio::test]
async fn test_plaintext_client_is_rejected() {
    let pki = TestPki::generate();
    let port = spawn_server(mutual_settings(&pki)).await;

    let result = async {
        let channel = Channel::from_shared(format!("http://127.0.0.1:{port}"))
            .unwrap()
            .connect()
            .await
            .map_err(|e| e.to_string())?;
        FederatedLearningClient::new(channel)
            .register(register_req("alice"))
            .await
            .map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    }
    .await;

    assert!(result.is_err(), "plaintext client must not complete an RPC");
}

#[tokio::test]
async fn test_wrong_ca_client_fails_handshake() {
    let pki = TestPki::generate();
    let port = spawn_server(mutual_settings(&pki)).await;

    let (cert, key, _rogue_ca) = TestPki::wrong_ca_client("mallory");
    let result = async {
        let mut client = tls_client(&pki, &cert, &key, port)
            .await
            .map_err(|e| e.to_string())?;
        client
            .register(register_req("mallory"))
            .await
            .map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    }
    .await;

    assert!(result.is_err(), "wrong-CA client must be rejected");
}

#[tokio::test]
async fn test_register_with_mismatched_cn_is_denied() {
    let pki = TestPki::generate();
    let port = spawn_server(mutual_settings(&pki)).await;

    // Valid certificate for "alice", but claims to be "bob" in the payload.
    let (cert, key) = pki.client_cert("alice");
    let mut client = tls_client(&pki, &cert, &key, port).await.unwrap();

    let err = client.register(register_req("bob")).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[tokio::test]
async fn test_session_hijack_via_stolen_session_id_is_denied() {
    use fedlearn_transport::proto::key_exchange_client::KeyExchangeClient;
    use fedlearn_transport::proto::{KeyRequest, ModelRequest};

    let pki = TestPki::generate();
    let port = spawn_server(mutual_settings(&pki)).await;

    // Bob registers legitimately; his session id leaks to Alice.
    let (bob_cert, bob_key) = pki.client_cert("bob");
    let mut bob = tls_client(&pki, &bob_cert, &bob_key, port).await.unwrap();
    let bob_session = bob
        .register(register_req("bob"))
        .await
        .unwrap()
        .into_inner()
        .session_id;

    // Alice presents her own valid certificate but claims Bob's identity
    // and session. The self-reported client_id field must not be trusted.
    let (alice_cert, alice_key) = pki.client_cert("alice");
    let alice_tls = client_tls_config(
        pki.ca_pem.as_bytes(),
        alice_cert.as_bytes(),
        alice_key.as_bytes(),
        "localhost",
    );
    let alice_channel = Channel::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(alice_tls)
        .unwrap()
        .connect()
        .await
        .unwrap();

    let err = FederatedLearningClient::new(alice_channel.clone())
        .get_global_model(ModelRequest {
            session_id: bob_session.clone(),
            client_id: "bob".to_string(),
            round: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::PermissionDenied);

    let err = KeyExchangeClient::new(alice_channel)
        .request_key(KeyRequest {
            session_id: bob_session,
            client_id: "bob".to_string(),
            key_bits: 256,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[test]
fn test_pki_generates_parseable_material() {
    let pki = TestPki::generate();
    let (client_cert, client_key) = pki.client_cert("client-1");

    for (label, pem) in [
        ("ca", pki.ca_pem.as_str()),
        ("server_cert", pki.server_cert_pem.as_str()),
        ("client_cert", client_cert.as_str()),
    ] {
        let (_, parsed) = x509_parser::pem::parse_x509_pem(pem.as_bytes()).unwrap_or_else(|e| {
            panic!("{label} PEM does not parse: {e}");
        });
        assert!(
            parsed.parse_x509().is_ok(),
            "{label} is not a valid certificate"
        );
    }

    assert!(pki.server_key_pem.contains("PRIVATE KEY"));
    assert!(client_key.contains("PRIVATE KEY"));
}
