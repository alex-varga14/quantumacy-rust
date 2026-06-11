//! gRPC service implementations for federated learning.
//!
//! Bridges the in-memory aggregation service and QKD server to the tonic
//! transport boundary used by external clients.

use crate::proto::federated_learning_server::FederatedLearning;
use crate::proto::key_exchange_server::KeyExchange;
use crate::proto::{
    KeyRequest, KeyResponse, ModelRequest, ModelResponse, RegisterRequest, RegisterResponse,
    RotateKeyRequest, RoundNotification, StatusRequest, StatusResponse, SubscribeRequest,
    UpdateRequest, UpdateResponse,
};
use crate::secure_channel::SecureFLChannel;
use crate::TransportResult;
use fedlearn_core::aggregation::FedAvg;
use fedlearn_core::model::{ModelUpdate, ModelWeights};
use fedlearn_core::privacy::DifferentialPrivacy;
use fedlearn_core::round::{self, RoundResult, TrainingConfig};
use parking_lot::RwLock;
use qkd_network::secure_channel::{EncryptedMessage, SecureChannel};
use qkd_network::server::QkdServer;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

/// Capacity of the broadcast channel used to publish round transitions to
/// `SubscribeRounds` listeners. A small buffer is enough; lagging subscribers
/// fall back to the snapshot already delivered at subscription time.
const ROUND_BROADCAST_CAPACITY: usize = 32;

/// Upper bound on client-requested key size. Qubit simulation cost scales
/// linearly with the request, so an unclamped value is a memory/CPU DoS.
const MAX_KEY_BITS: u32 = 4096;

/// Number of simulated qubits needed to distill a key of `key_bits`,
/// clamped to [1024, MAX_KEY_BITS * 16].
fn qubits_for_key_bits(key_bits: u32) -> usize {
    let bits = key_bits.min(MAX_KEY_BITS) as usize;
    usize::max(bits * 16, 1024)
}

/// Identifier for the derivation scheme carried in `KeyResponse.derivation`.
const KEY_DERIVATION_SCHEME: &str = "hkdf-sha256-v1";

/// Derive the per-round channel key from a server-held QKD key. The raw QKD
/// material never crosses the wire: both encrypt (client, from the
/// KeyExchange response) and decrypt (server, re-derived here) use this key.
fn derive_round_key(qkd_material: &[u8], session_id: &str, round: u32) -> [u8; 32] {
    let hk = hkdf::Hkdf::<sha2::Sha256>::new(Some(session_id.as_bytes()), qkd_material);
    let mut okm = [0u8; 32];
    hk.expand(
        format!("quantumacy-fl-v1:round:{round}").as_bytes(),
        &mut okm,
    )
    .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Payload broadcast on each round transition.
#[derive(Debug, Clone)]
pub struct RoundEvent {
    pub round: u32,
    pub action: String,
    pub payload: Vec<u8>,
}

/// Server-side FL aggregation service
pub struct AggregationService {
    /// Current global model
    global_weights: Arc<RwLock<ModelWeights>>,
    /// FedAvg aggregator
    aggregator: Arc<RwLock<FedAvg>>,
    /// DP mechanism (optional)
    dp: Arc<RwLock<Option<DifferentialPrivacy>>>,
    /// Training configuration
    config: TrainingConfig,
    /// Current round
    current_round: Arc<RwLock<u32>>,
    /// Collected updates for current round
    pending_updates: Arc<RwLock<Vec<ModelUpdate>>>,
    /// Registered clients
    clients: Arc<RwLock<HashMap<String, ClientInfo>>>,
    /// Training history
    history: Arc<RwLock<Vec<RoundResult>>>,
    /// Broadcast sender for round transitions
    round_tx: broadcast::Sender<RoundEvent>,
}

#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub client_id: String,
    pub dataset_size: u64,
    pub registered_at: chrono::DateTime<chrono::Utc>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
struct SessionState {
    client_id: String,
    /// True when the session was established over mTLS and bound to the
    /// client certificate's CN; such sessions re-verify the CN on every RPC.
    authenticated: bool,
}

#[derive(Debug, Clone)]
pub struct AggregationStatus {
    pub current_round: u32,
    pub total_rounds: u32,
    pub participating_clients: usize,
    pub global_loss: f64,
    pub global_accuracy: f64,
    pub state: String,
}

#[derive(Clone)]
struct TransportRuntime {
    aggregation: Arc<AggregationService>,
    qkd_server: Arc<QkdServer>,
    sessions: Arc<RwLock<HashMap<String, SessionState>>>,
}

impl TransportRuntime {
    /// Validate that `client_id` owns `session_id`, and — for sessions
    /// established over mTLS — that the caller's certificate CN matches the
    /// session identity. The self-reported `client_id` field is never the
    /// trust anchor on authenticated sessions.
    fn validate_session(
        &self,
        session_id: &str,
        client_id: &str,
        peer_cn: Option<&str>,
    ) -> Result<(), Status> {
        let sessions = self.sessions.read();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Status::not_found(format!("Session not found: {session_id}")))?;

        if session.client_id != client_id {
            return Err(Status::permission_denied(format!(
                "Client {client_id} does not own session {session_id}"
            )));
        }

        if session.authenticated && peer_cn != Some(session.client_id.as_str()) {
            return Err(Status::permission_denied(
                "certificate identity does not match session owner",
            ));
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct GrpcFederatedLearningService {
    runtime: TransportRuntime,
}

#[derive(Clone)]
pub struct GrpcKeyExchangeService {
    runtime: TransportRuntime,
}

impl AggregationService {
    pub fn new(initial_weights: ModelWeights, config: TrainingConfig) -> Self {
        let dp = config
            .dp
            .as_ref()
            .map(|c| DifferentialPrivacy::new(c.clone()));

        let (round_tx, _) = broadcast::channel(ROUND_BROADCAST_CAPACITY);
        Self {
            global_weights: Arc::new(RwLock::new(initial_weights)),
            aggregator: Arc::new(RwLock::new(FedAvg::new(config.fedavg.clone()))),
            dp: Arc::new(RwLock::new(dp)),
            config,
            current_round: Arc::new(RwLock::new(0)),
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            clients: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            round_tx,
        }
    }

    /// Subscribe to round-transition events. The returned receiver fires on
    /// every successful aggregation; cold subscribers should also call
    /// [`status_snapshot`] to seed their initial state.
    pub fn subscribe(&self) -> broadcast::Receiver<RoundEvent> {
        self.round_tx.subscribe()
    }

    /// Register a new client
    pub fn register_client(&self, client_id: String, dataset_size: u64) -> u32 {
        let now = chrono::Utc::now();
        self.clients.write().insert(
            client_id.clone(),
            ClientInfo {
                client_id: client_id.clone(),
                dataset_size,
                registered_at: now,
                last_seen: now,
            },
        );
        info!(client_id = %client_id, "Client registered");
        *self.current_round.read()
    }

    /// Get current global model weights
    pub fn get_global_weights(&self) -> ModelWeights {
        self.global_weights.read().clone()
    }

    /// Submit a client update
    pub fn submit_update(&self, update: ModelUpdate) -> TransportResult<bool> {
        let current = *self.current_round.read();
        if update.metadata.round != current {
            warn!(
                expected = current,
                got = update.metadata.round,
                client = %update.client_id,
                "Update for wrong round"
            );
            return Ok(false);
        }

        if let Some(client) = self.clients.write().get_mut(&update.client_id) {
            client.last_seen = chrono::Utc::now();
        }

        self.pending_updates.write().push(update);
        Ok(true)
    }

    /// Attempt to aggregate pending updates and advance to next round.
    /// Returns the new global weights if aggregation succeeds.
    pub fn try_aggregate(&self) -> TransportResult<Option<ModelWeights>> {
        let updates: Vec<ModelUpdate> = {
            let mut pending = self.pending_updates.write();
            if pending.len() < self.config.fedavg.min_clients {
                return Ok(None);
            }
            std::mem::take(&mut *pending)
        };

        let global = self.global_weights.read().clone();
        let mut agg = self.aggregator.write();
        let mut dp_guard = self.dp.write();

        let result = round::execute_round(
            *self.current_round.read(),
            &global,
            updates,
            &mut agg,
            dp_guard.as_mut(),
        );

        match result {
            Ok((new_weights, round_result)) => {
                *self.global_weights.write() = new_weights.clone();
                *self.current_round.write() += 1;
                self.history.write().push(round_result);

                let round = *self.current_round.read();
                info!(round, "Advanced to new round");

                if round::should_stop_early(&self.history.read(), &self.config) {
                    info!("Early stopping triggered");
                }

                let action = if round >= self.config.num_rounds {
                    "complete"
                } else {
                    "train"
                };
                let payload = serde_json::to_vec(&new_weights).unwrap_or_default();
                let _ = self.round_tx.send(RoundEvent {
                    round,
                    action: action.to_string(),
                    payload,
                });

                Ok(Some(new_weights))
            }
            Err(e) => {
                warn!(error = %e, "Aggregation failed");
                Ok(None)
            }
        }
    }

    pub fn current_round(&self) -> u32 {
        *self.current_round.read()
    }

    pub fn num_clients(&self) -> usize {
        self.clients.read().len()
    }

    pub fn history(&self) -> Vec<RoundResult> {
        self.history.read().clone()
    }

    pub fn status_snapshot(&self) -> AggregationStatus {
        let history = self.history.read();
        let latest = history.last().cloned();
        let pending = self.pending_updates.read().len();
        let current_round = *self.current_round.read();
        let state = if current_round >= self.config.num_rounds {
            "complete"
        } else if pending >= self.config.fedavg.min_clients {
            "aggregating"
        } else {
            "waiting"
        };

        AggregationStatus {
            current_round,
            total_rounds: self.config.num_rounds,
            participating_clients: self.clients.read().len(),
            global_loss: latest.as_ref().map(|r| r.avg_loss).unwrap_or(0.0),
            global_accuracy: latest.as_ref().and_then(|r| r.accuracy).unwrap_or(0.0),
            state: state.to_string(),
        }
    }
}

impl GrpcFederatedLearningService {
    pub fn new(aggregation: Arc<AggregationService>, qkd_server: Arc<QkdServer>) -> Self {
        Self {
            runtime: TransportRuntime {
                aggregation,
                qkd_server,
                sessions: Arc::new(RwLock::new(HashMap::new())),
            },
        }
    }

    pub fn key_exchange_service(&self) -> GrpcKeyExchangeService {
        GrpcKeyExchangeService {
            runtime: self.runtime.clone(),
        }
    }

    fn decode_update(&self, request: &UpdateRequest) -> Result<ModelUpdate, Status> {
        if request.encrypted {
            let raw = self
                .runtime
                .qkd_server
                .get_key(&request.key_id)
                .map_err(|e| Status::failed_precondition(format!("Unable to load key: {e}")))?;

            // Re-derive the per-round key the client received from
            // KeyExchange; the raw QKD key itself is never used on the wire.
            let derived = derive_round_key(&raw.material, &request.session_id, request.round);
            let key = qkd_core::types::SecureKey {
                key_id: raw.key_id.clone(),
                timestamp: raw.timestamp,
                material: derived.to_vec(),
                length_bits: derived.len() * 8,
            };

            let channel = SecureChannel::from_key(&key)
                .map_err(|e| Status::internal(format!("Unable to create secure channel: {e}")))?;
            let fl_channel = SecureFLChannel::new(channel, request.key_id.clone());

            let encrypted: EncryptedMessage = serde_json::from_slice(&request.model_update)
                .map_err(|e| Status::invalid_argument(format!("Invalid encrypted payload: {e}")))?;

            fl_channel
                .decrypt_update(&encrypted)
                .map_err(|e| Status::invalid_argument(format!("Unable to decrypt update: {e}")))
        } else {
            serde_json::from_slice(&request.model_update)
                .map_err(|e| Status::invalid_argument(format!("Invalid model update payload: {e}")))
        }
    }
}

impl GrpcKeyExchangeService {
    async fn mint_key(
        &self,
        session_id: &str,
        client_id: &str,
        peer_cn: Option<&str>,
        key_bits: u32,
    ) -> Result<KeyResponse, Status> {
        self.runtime
            .validate_session(session_id, client_id, peer_cn)?;

        let num_qubits = qubits_for_key_bits(key_bits);
        let key_id = self
            .runtime
            .qkd_server
            .generate_key(num_qubits)
            .await
            .map_err(|e| Status::internal(format!("Key generation failed: {e}")))?;
        let key = self
            .runtime
            .qkd_server
            .get_key(&key_id)
            .map_err(|e| Status::internal(format!("Generated key unavailable: {e}")))?;

        let round = self.runtime.aggregation.current_round();
        let derived = derive_round_key(&key.material, session_id, round);

        Ok(KeyResponse {
            key_id,
            key_material: derived.to_vec(),
            expires_at: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp() as u64,
            round,
            derivation: KEY_DERIVATION_SCHEME.to_string(),
        })
    }
}

#[tonic::async_trait]
impl FederatedLearning for GrpcFederatedLearningService {
    type SubscribeRoundsStream = ReceiverStream<Result<RoundNotification, Status>>;

    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        if request.client_id.trim().is_empty() {
            return Err(Status::invalid_argument("client_id is required"));
        }

        // On mTLS connections the certificate CN is the identity; the
        // self-reported client_id must match it.
        let authenticated = match &peer_cn {
            Some(cn) if *cn != request.client_id => {
                return Err(Status::permission_denied(format!(
                    "client_id '{}' does not match certificate CN '{cn}'",
                    request.client_id
                )));
            }
            Some(_) => true,
            None => false,
        };

        let current_round = self
            .runtime
            .aggregation
            .register_client(request.client_id.clone(), request.dataset_size);
        let session_id = self.runtime.qkd_server.create_session().await;
        self.runtime.sessions.write().insert(
            session_id.clone(),
            SessionState {
                client_id: request.client_id,
                authenticated,
            },
        );

        let global_model = serde_json::to_vec(&self.runtime.aggregation.get_global_weights())
            .map_err(|e| Status::internal(format!("Unable to serialize model: {e}")))?;

        Ok(Response::new(RegisterResponse {
            session_id,
            current_round,
            global_model,
        }))
    }

    async fn get_global_model(
        &self,
        request: Request<ModelRequest>,
    ) -> Result<Response<ModelResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        self.runtime.validate_session(
            &request.session_id,
            &request.client_id,
            peer_cn.as_deref(),
        )?;

        let weights = self.runtime.aggregation.get_global_weights();
        let serialized = serde_json::to_vec(&weights)
            .map_err(|e| Status::internal(format!("Unable to serialize weights: {e}")))?;

        Ok(Response::new(ModelResponse {
            round: self.runtime.aggregation.current_round(),
            model_weights: serialized,
            training_config: Vec::new(),
        }))
    }

    async fn submit_update(
        &self,
        request: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        self.runtime.validate_session(
            &request.session_id,
            &request.client_id,
            peer_cn.as_deref(),
        )?;

        let update = self.decode_update(&request)?;
        if update.client_id != request.client_id {
            return Err(Status::permission_denied(
                "update client_id does not match session",
            ));
        }
        if update.metadata.round != request.round {
            return Err(Status::invalid_argument(
                "update round does not match request round",
            ));
        }

        let accepted = self
            .runtime
            .aggregation
            .submit_update(update)
            .map_err(|e| Status::internal(format!("Unable to submit update: {e}")))?;

        let aggregated = if accepted {
            self.runtime
                .aggregation
                .try_aggregate()
                .map_err(|e| Status::internal(format!("Unable to aggregate updates: {e}")))?
                .is_some()
        } else {
            false
        };

        let message = if !accepted {
            "Update rejected for current round".to_string()
        } else if aggregated {
            "Update accepted and aggregation completed".to_string()
        } else {
            "Update accepted".to_string()
        };

        Ok(Response::new(UpdateResponse { accepted, message }))
    }

    async fn get_status(
        &self,
        request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        let client_id = {
            let sessions = self.runtime.sessions.read();
            let session = sessions.get(&request.session_id).ok_or_else(|| {
                Status::not_found(format!("Session not found: {}", request.session_id))
            })?;
            session.client_id.clone()
        };
        self.runtime
            .validate_session(&request.session_id, &client_id, peer_cn.as_deref())?;

        let status = self.runtime.aggregation.status_snapshot();
        Ok(Response::new(StatusResponse {
            current_round: status.current_round,
            total_rounds: status.total_rounds,
            participating_clients: status.participating_clients as u32,
            global_loss: status.global_loss,
            global_accuracy: status.global_accuracy,
            state: status.state,
        }))
    }

    async fn subscribe_rounds(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeRoundsStream>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        self.runtime.validate_session(
            &request.session_id,
            &request.client_id,
            peer_cn.as_deref(),
        )?;

        // Snapshot the current state so subscribers don't have to wait for the
        // next aggregation to learn where the platform is.
        let status = self.runtime.aggregation.status_snapshot();
        let payload = serde_json::to_vec(&self.runtime.aggregation.get_global_weights())
            .map_err(|e| Status::internal(format!("Unable to serialize weights: {e}")))?;
        let initial_action = if status.state == "complete" {
            "complete"
        } else {
            "train"
        };

        // Subscribe BEFORE sending the snapshot so we don't miss a round event
        // that fires between the snapshot and the broadcast subscribe.
        let mut rx_round = self.runtime.aggregation.subscribe();

        let (tx, rx) = mpsc::channel::<Result<RoundNotification, Status>>(ROUND_BROADCAST_CAPACITY);
        tx.send(Ok(RoundNotification {
            round: status.current_round,
            action: initial_action.to_string(),
            payload,
        }))
        .await
        .map_err(|e| Status::internal(format!("Unable to publish notification: {e}")))?;

        tokio::spawn(async move {
            loop {
                match rx_round.recv().await {
                    Ok(event) => {
                        if tx
                            .send(Ok(RoundNotification {
                                round: event.round,
                                action: event.action,
                                payload: event.payload,
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(skipped, "subscribe_rounds subscriber lagged");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tonic::async_trait]
impl KeyExchange for GrpcKeyExchangeService {
    async fn request_key(
        &self,
        request: Request<KeyRequest>,
    ) -> Result<Response<KeyResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        let key = self
            .mint_key(
                &request.session_id,
                &request.client_id,
                peer_cn.as_deref(),
                request.key_bits,
            )
            .await?;
        Ok(Response::new(key))
    }

    async fn rotate_key(
        &self,
        request: Request<RotateKeyRequest>,
    ) -> Result<Response<KeyResponse>, Status> {
        let peer_cn = crate::tls::peer_common_name(&request);
        let request = request.into_inner();
        let key = self
            .mint_key(
                &request.session_id,
                &request.client_id,
                peer_cn.as_deref(),
                256,
            )
            .await?;
        Ok(Response::new(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::key_exchange_server::KeyExchange;
    use crate::secure_channel::SecureFLChannel;
    use fedlearn_core::model::{LayerWeights, UpdateMetadata};
    use qkd_core::channel::ChannelConfig;
    use qkd_core::types::{ProtocolType, SecureKey};
    use tonic::Request;

    #[test]
    fn test_round_key_derivation_is_round_and_session_scoped() {
        let material = vec![7u8; 32];
        let k0 = derive_round_key(&material, "session-1", 0);
        let k1 = derive_round_key(&material, "session-1", 1);
        let other_session = derive_round_key(&material, "session-2", 0);

        assert_ne!(k0, k1, "rounds must derive distinct keys");
        assert_ne!(k0, other_session, "sessions must derive distinct keys");
        assert_eq!(
            k0,
            derive_round_key(&material, "session-1", 0),
            "derivation must be deterministic"
        );
    }

    #[test]
    fn test_requested_key_bits_are_clamped() {
        // A hostile client must not be able to drive qubit allocation
        // with an unbounded key_bits request.
        assert_eq!(qubits_for_key_bits(u32::MAX), (MAX_KEY_BITS as usize) * 16);
        assert_eq!(qubits_for_key_bits(0), 1024);
        assert_eq!(qubits_for_key_bits(256), 4096);
    }

    fn make_weights(vals: Vec<f32>) -> ModelWeights {
        let n = vals.len();
        ModelWeights {
            layers: vec![LayerWeights {
                name: "test".to_string(),
                shape: vec![n],
                data: vals,
            }],
            num_params: n,
        }
    }

    fn make_update(id: &str, vals: Vec<f32>, round: u32) -> ModelUpdate {
        let n = vals.len();
        ModelUpdate {
            client_id: id.to_string(),
            weights: ModelWeights {
                layers: vec![LayerWeights {
                    name: "test".to_string(),
                    shape: vec![n],
                    data: vals,
                }],
                num_params: n,
            },
            num_samples: 100,
            loss: 0.5,
            metadata: UpdateMetadata {
                local_epochs: 1,
                learning_rate: 0.01,
                batch_size: 32,
                round,
                training_time_ms: 100,
            },
        }
    }

    fn make_runtime() -> (
        Arc<AggregationService>,
        Arc<QkdServer>,
        GrpcFederatedLearningService,
    ) {
        let aggregation = Arc::new(AggregationService::new(
            make_weights(vec![0.0, 0.0]),
            TrainingConfig::default(),
        ));
        let qkd_server = Arc::new(QkdServer::new(ChannelConfig::default(), ProtocolType::BB84));
        let grpc = GrpcFederatedLearningService::new(aggregation.clone(), qkd_server.clone());
        (aggregation, qkd_server, grpc)
    }

    fn secure_key_from_response(response: &KeyResponse) -> SecureKey {
        SecureKey {
            key_id: response.key_id.clone(),
            timestamp: chrono::Utc::now(),
            material: response.key_material.clone(),
            length_bits: response.key_material.len() * 8,
        }
    }

    #[tokio::test]
    async fn test_subscribe_observes_round_transition() {
        let service =
            AggregationService::new(make_weights(vec![0.0, 0.0]), TrainingConfig::default());

        // Subscribe BEFORE submitting updates so the broadcast event is captured.
        let mut rx = service.subscribe();

        service.register_client("c1".to_string(), 1000);
        service.register_client("c2".to_string(), 2000);
        service
            .submit_update(make_update("c1", vec![1.0, 2.0], 0))
            .unwrap();
        service
            .submit_update(make_update("c2", vec![3.0, 4.0], 0))
            .unwrap();
        let aggregated = service.try_aggregate().unwrap();
        assert!(aggregated.is_some());

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("subscriber timed out waiting for round event")
            .expect("subscriber lost broadcast channel");
        assert_eq!(event.round, 1);
        assert_eq!(event.action, "train");
        assert!(!event.payload.is_empty());
    }

    #[test]
    fn test_register_and_submit() {
        let service =
            AggregationService::new(make_weights(vec![0.0, 0.0]), TrainingConfig::default());

        service.register_client("c1".to_string(), 1000);
        service.register_client("c2".to_string(), 2000);
        assert_eq!(service.num_clients(), 2);

        service
            .submit_update(make_update("c1", vec![1.0, 2.0], 0))
            .unwrap();
        service
            .submit_update(make_update("c2", vec![3.0, 4.0], 0))
            .unwrap();

        let result = service.try_aggregate().unwrap();
        assert!(result.is_some());
        assert_eq!(service.current_round(), 1);
    }

    #[test]
    fn test_wrong_round_rejected() {
        let service = AggregationService::new(make_weights(vec![0.0]), TrainingConfig::default());

        let accepted = service
            .submit_update(make_update("c1", vec![1.0], 5))
            .unwrap();
        assert!(!accepted);
    }

    #[tokio::test]
    async fn test_grpc_register_and_status_flow() {
        let (_aggregation, _qkd_server, grpc) = make_runtime();

        let register = grpc
            .register(Request::new(RegisterRequest {
                client_id: "client-a".to_string(),
                dataset_size: 128,
                capabilities: "{}".to_string(),
            }))
            .await
            .unwrap()
            .into_inner();

        let status = grpc
            .get_status(Request::new(StatusRequest {
                session_id: register.session_id,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(status.current_round, 0);
        assert_eq!(status.participating_clients, 1);
        assert_eq!(status.state, "waiting");
    }

    #[tokio::test]
    async fn test_encrypted_submit_update_aggregates() {
        let (aggregation, _qkd_server, grpc) = make_runtime();
        let key_service = grpc.key_exchange_service();

        let reg_a = grpc
            .register(Request::new(RegisterRequest {
                client_id: "client-a".to_string(),
                dataset_size: 100,
                capabilities: "{}".to_string(),
            }))
            .await
            .unwrap()
            .into_inner();
        let reg_b = grpc
            .register(Request::new(RegisterRequest {
                client_id: "client-b".to_string(),
                dataset_size: 100,
                capabilities: "{}".to_string(),
            }))
            .await
            .unwrap()
            .into_inner();

        let key_a = key_service
            .request_key(Request::new(KeyRequest {
                client_id: "client-a".to_string(),
                session_id: reg_a.session_id.clone(),
                key_bits: 256,
            }))
            .await
            .unwrap()
            .into_inner();
        let key_b = key_service
            .request_key(Request::new(KeyRequest {
                client_id: "client-b".to_string(),
                session_id: reg_b.session_id.clone(),
                key_bits: 256,
            }))
            .await
            .unwrap()
            .into_inner();

        let channel_a = SecureChannel::from_key(&secure_key_from_response(&key_a)).unwrap();
        let channel_b = SecureChannel::from_key(&secure_key_from_response(&key_b)).unwrap();
        let fl_a = SecureFLChannel::new(channel_a, key_a.key_id.clone());
        let fl_b = SecureFLChannel::new(channel_b, key_b.key_id.clone());

        let enc_a = serde_json::to_vec(
            &fl_a
                .encrypt_update(&make_update("client-a", vec![1.0, 2.0], 0))
                .unwrap(),
        )
        .unwrap();
        let enc_b = serde_json::to_vec(
            &fl_b
                .encrypt_update(&make_update("client-b", vec![3.0, 4.0], 0))
                .unwrap(),
        )
        .unwrap();

        let resp_a = grpc
            .submit_update(Request::new(UpdateRequest {
                session_id: reg_a.session_id,
                client_id: "client-a".to_string(),
                round: 0,
                model_update: enc_a,
                key_id: key_a.key_id,
                encrypted: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let resp_b = grpc
            .submit_update(Request::new(UpdateRequest {
                session_id: reg_b.session_id,
                client_id: "client-b".to_string(),
                round: 0,
                model_update: enc_b,
                key_id: key_b.key_id,
                encrypted: true,
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(resp_a.accepted);
        assert!(resp_b.accepted);
        assert_eq!(aggregation.current_round(), 1);
        assert_eq!(aggregation.get_global_weights().flatten(), vec![2.0, 3.0]);
    }
}
