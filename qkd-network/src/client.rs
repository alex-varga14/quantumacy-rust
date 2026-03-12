//! QKD client for requesting keys from a QKD server.

use crate::NetworkResult;
use tracing::info;

/// Client configuration
#[derive(Debug, Clone)]
pub struct QkdClientConfig {
    /// Server address (host:port)
    pub server_addr: String,
    /// Client identifier
    pub client_id: String,
    /// Number of qubits per key request
    pub default_qubits: usize,
}

impl Default for QkdClientConfig {
    fn default() -> Self {
        Self {
            server_addr: "127.0.0.1:50051".to_string(),
            client_id: uuid::Uuid::new_v4().to_string(),
            default_qubits: 10_000,
        }
    }
}

/// QKD Client
pub struct QkdClient {
    config: QkdClientConfig,
}

impl QkdClient {
    pub fn new(config: QkdClientConfig) -> Self {
        info!(
            client_id = %config.client_id,
            server = %config.server_addr,
            "QKD client initialized"
        );
        Self { config }
    }

    pub fn client_id(&self) -> &str {
        &self.config.client_id
    }

    pub fn server_addr(&self) -> &str {
        &self.config.server_addr
    }
}
