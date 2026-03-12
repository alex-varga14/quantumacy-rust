//! Peer-to-peer QKD key exchange.
//!
//! Direct key exchange between two parties without a central server.
//! Both peers run the protocol simultaneously and derive a shared key.

use crate::NetworkResult;
use qkd_core::channel::ChannelConfig;
use qkd_core::key_manager::{KeyManager, KeyManagerConfig};
use qkd_core::protocols::QkdProtocol;
use qkd_core::types::SecureKey;
use tracing::info;

/// P2P QKD peer
pub struct QkdPeer {
    peer_id: String,
    key_manager: KeyManager,
}

impl QkdPeer {
    pub fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            key_manager: KeyManager::new(KeyManagerConfig::default()),
        }
    }

    /// Execute a key exchange with a remote peer
    pub async fn exchange_key(
        &self,
        protocol: &dyn QkdProtocol,
        channel_config: &ChannelConfig,
        num_qubits: usize,
    ) -> NetworkResult<SecureKey> {
        let cc = channel_config.clone();
        let nq = num_qubits;

        // In a real implementation, this would involve network I/O between peers.
        // Here we simulate the full protocol locally.
        let proto_name = protocol.name().to_string();

        // Since QkdProtocol is not Send-safe in all impls, we construct fresh
        // For now, use BB84 as default P2P protocol
        let (key, stats) = tokio::task::spawn_blocking(move || {
            let proto = qkd_core::protocols::bb84::Bb84::default();
            proto.execute(nq, &cc)
        })
        .await
        .map_err(|e| crate::NetworkError::Connection(format!("Join error: {e}")))?
        .map_err(crate::NetworkError::Qkd)?;

        info!(
            peer = %self.peer_id,
            protocol = %proto_name,
            key_id = %key.key_id,
            qber = format!("{:.4}", stats.qber),
            "P2P key exchange complete"
        );

        self.key_manager.store(key.clone())?;
        Ok(key)
    }

    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }

    pub fn available_keys(&self) -> usize {
        self.key_manager.key_count()
    }
}
