//! QKD key distribution server.
//!
//! Manages QKD sessions, runs protocols on demand, and distributes
//! keys to authenticated clients via gRPC.

use crate::{NetworkError, NetworkResult};
use qkd_core::channel::ChannelConfig;
use qkd_core::key_manager::{KeyManager, KeyManagerConfig};
use qkd_core::protocols::bb84::Bb84;
use qkd_core::protocols::QkdProtocol;
use qkd_core::types::ProtocolType;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// QKD Server state
pub struct QkdServer {
    key_manager: KeyManager,
    channel_config: ChannelConfig,
    protocol: ProtocolType,
    /// Active sessions
    sessions: Arc<RwLock<Vec<String>>>,
}

impl QkdServer {
    pub fn new(channel_config: ChannelConfig, protocol: ProtocolType) -> Self {
        Self {
            key_manager: KeyManager::new(KeyManagerConfig::default()),
            channel_config,
            protocol,
            sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Generate a new QKD key using the configured protocol
    pub async fn generate_key(
        &self,
        num_qubits: usize,
    ) -> NetworkResult<String> {
        let protocol: Box<dyn QkdProtocol> = match self.protocol {
            ProtocolType::BB84 => Box::new(Bb84::default()),
            ProtocolType::SixState => {
                Box::new(qkd_core::protocols::six_state::SixState::default())
            }
            ProtocolType::B92 => {
                Box::new(qkd_core::protocols::b92::B92::default())
            }
        };

        let channel_config = self.channel_config.clone();

        // Run protocol on a blocking thread (CPU-intensive)
        let (key, stats) = tokio::task::spawn_blocking(move || {
            protocol.execute(num_qubits, &channel_config)
        })
        .await
        .map_err(|e| NetworkError::Connection(format!("Task join error: {e}")))?
        .map_err(NetworkError::Qkd)?;

        info!(
            key_id = %key.key_id,
            protocol = ?self.protocol,
            qber = format!("{:.4}", stats.qber),
            final_bits = stats.final_key_bits,
            "Generated new QKD key"
        );

        let key_id = self.key_manager.store(key)?;
        Ok(key_id)
    }

    /// Retrieve a key for distribution to a client
    pub fn get_key(&self, key_id: &str) -> NetworkResult<qkd_core::types::SecureKey> {
        self.key_manager.get(key_id).map_err(NetworkError::Qkd)
    }

    /// Consume a key (one-time pad style)
    pub fn consume_key(&self, key_id: &str) -> NetworkResult<qkd_core::types::SecureKey> {
        self.key_manager.consume(key_id).map_err(NetworkError::Qkd)
    }

    /// Number of available keys
    pub fn available_keys(&self) -> usize {
        self.key_manager.key_count()
    }

    /// Register a new session
    pub async fn create_session(&self) -> String {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.sessions.write().await.push(session_id.clone());
        info!(session_id = %session_id, "New QKD session created");
        session_id
    }
}
