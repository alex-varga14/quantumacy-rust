//! QKD-secured transport channel for federated learning.
//!
//! Wraps model updates in AES-256-GCM encryption using keys
//! derived from QKD sessions. Supports key rotation per round.

use crate::{TransportError, TransportResult};
use fedlearn_core::model::{ModelUpdate, ModelWeights};
use qkd_network::secure_channel::{EncryptedMessage, SecureChannel};
use tracing::debug;

/// Encrypts federated learning payloads using QKD-derived keys
pub struct SecureFLChannel {
    inner: SecureChannel,
    key_id: String,
    round: u32,
}

impl SecureFLChannel {
    /// Create from a QKD secure channel
    pub fn new(channel: SecureChannel, key_id: String) -> Self {
        Self {
            inner: channel,
            key_id,
            round: 0,
        }
    }

    /// Encrypt a model update for transmission
    pub fn encrypt_update(&self, update: &ModelUpdate) -> TransportResult<EncryptedMessage> {
        let serialized =
            serde_json::to_vec(update).map_err(|e| TransportError::Serialization(e.to_string()))?;

        debug!(
            client = %update.client_id,
            round = update.metadata.round,
            payload_bytes = serialized.len(),
            "Encrypting model update"
        );

        self.inner
            .encrypt(&serialized)
            .map_err(|e| TransportError::Serialization(e.to_string()))
    }

    /// Decrypt a received model update
    pub fn decrypt_update(&self, msg: &EncryptedMessage) -> TransportResult<ModelUpdate> {
        let plaintext = self
            .inner
            .decrypt(msg)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        serde_json::from_slice(&plaintext).map_err(|e| TransportError::Serialization(e.to_string()))
    }

    /// Encrypt model weights for distribution
    pub fn encrypt_weights(&self, weights: &ModelWeights) -> TransportResult<EncryptedMessage> {
        let serialized = serde_json::to_vec(weights)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        self.inner
            .encrypt(&serialized)
            .map_err(|e| TransportError::Serialization(e.to_string()))
    }

    /// Decrypt received model weights
    pub fn decrypt_weights(&self, msg: &EncryptedMessage) -> TransportResult<ModelWeights> {
        let plaintext = self
            .inner
            .decrypt(msg)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        serde_json::from_slice(&plaintext).map_err(|e| TransportError::Serialization(e.to_string()))
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn set_round(&mut self, round: u32) {
        self.round = round;
    }
}
