//! QKD-secured transport channel for federated learning.
//!
//! Wraps model updates in AES-256-GCM encryption using keys
//! derived from QKD sessions. Supports key rotation per round.

use crate::{TransportError, TransportResult};
use chrono::{DateTime, Duration, Utc};
use fedlearn_core::model::{ModelUpdate, ModelWeights};
use qkd_network::secure_channel::{EncryptedMessage, SecureChannel};
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::debug;

/// Key usage limits for a [`SecureFLChannel`].
///
/// Once either limit is reached, encryption calls fail with
/// [`TransportError::KeyExpired`] and the caller must rotate the key
/// via the KeyExchange `rotate_key` RPC. Decryption is intentionally
/// not limited so the server can still decrypt in-flight messages.
#[derive(Debug, Clone)]
pub struct KeyPolicy {
    /// Maximum number of messages that may be encrypted under one key
    pub max_messages: u32,
    /// Maximum age of the channel key before rotation is required
    pub max_age: Duration,
}

impl Default for KeyPolicy {
    fn default() -> Self {
        Self {
            max_messages: 1000,
            max_age: Duration::hours(1),
        }
    }
}

/// Encrypts federated learning payloads using QKD-derived keys
pub struct SecureFLChannel {
    inner: SecureChannel,
    key_id: String,
    round: u32,
    policy: KeyPolicy,
    created_at: DateTime<Utc>,
    /// Messages encrypted so far. Atomic so the existing `&self`
    /// encryption API (shared with the gRPC service) keeps working.
    messages_sent: AtomicU32,
}

impl SecureFLChannel {
    /// Create from a QKD secure channel with the default [`KeyPolicy`]
    pub fn new(channel: SecureChannel, key_id: String) -> Self {
        Self::with_policy(channel, key_id, KeyPolicy::default())
    }

    /// Create from a QKD secure channel with an explicit [`KeyPolicy`]
    pub fn with_policy(channel: SecureChannel, key_id: String, policy: KeyPolicy) -> Self {
        Self {
            inner: channel,
            key_id,
            round: 0,
            policy,
            created_at: Utc::now(),
            messages_sent: AtomicU32::new(0),
        }
    }

    /// Check the key policy and reserve one message slot.
    ///
    /// The age check uses `>=` so a `max_age` of zero deterministically
    /// expires the key before any message is encrypted.
    fn authorize_encrypt(&self) -> TransportResult<()> {
        let age = Utc::now() - self.created_at;
        if age >= self.policy.max_age {
            return Err(TransportError::KeyExpired(format!(
                "key {} exceeded max age ({}s); rotate via the KeyExchange rotate_key RPC",
                self.key_id,
                self.policy.max_age.num_seconds()
            )));
        }

        let sent = self.messages_sent.fetch_add(1, Ordering::SeqCst);
        if sent >= self.policy.max_messages {
            return Err(TransportError::KeyExpired(format!(
                "key {} exceeded max message count ({}); rotate via the KeyExchange rotate_key RPC",
                self.key_id, self.policy.max_messages
            )));
        }

        Ok(())
    }

    /// Encrypt a model update for transmission
    pub fn encrypt_update(&self, update: &ModelUpdate) -> TransportResult<EncryptedMessage> {
        self.authorize_encrypt()?;
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
        self.authorize_encrypt()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use fedlearn_core::model::{LayerWeights, ModelUpdate, ModelWeights, UpdateMetadata};
    use qkd_core::types::SecureKey;

    fn make_channel() -> SecureChannel {
        let key = SecureKey {
            key_id: "test-key".to_string(),
            timestamp: Utc::now(),
            material: vec![0x42; 32],
            length_bits: 256,
        };
        SecureChannel::from_key(&key).unwrap()
    }

    fn make_update(id: &str, vals: Vec<f32>) -> ModelUpdate {
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
                round: 0,
                training_time_ms: 100,
            },
        }
    }

    #[test]
    fn test_default_policy_allows_normal_use() {
        let fl = SecureFLChannel::new(make_channel(), "test-key".to_string());
        let update = make_update("alice", vec![1.0, 2.0]);

        let encrypted = fl.encrypt_update(&update).unwrap();
        let decrypted = fl.decrypt_update(&encrypted).unwrap();
        assert_eq!(decrypted.client_id, "alice");

        let weights = update.weights.clone();
        let encrypted = fl.encrypt_weights(&weights).unwrap();
        let decrypted = fl.decrypt_weights(&encrypted).unwrap();
        assert_eq!(decrypted.num_params, 2);
    }

    #[test]
    fn test_max_messages_limit_enforced() {
        let policy = KeyPolicy {
            max_messages: 2,
            max_age: chrono::Duration::hours(1),
        };
        let fl = SecureFLChannel::with_policy(make_channel(), "test-key".to_string(), policy);
        let update = make_update("alice", vec![1.0]);

        fl.encrypt_update(&update).unwrap();
        fl.encrypt_update(&update).unwrap();

        let err = fl.encrypt_update(&update).unwrap_err();
        assert!(matches!(err, TransportError::KeyExpired(_)));
        assert!(err.to_string().contains("rotate_key"));
    }

    #[test]
    fn test_max_messages_applies_to_weights_too() {
        let policy = KeyPolicy {
            max_messages: 1,
            max_age: chrono::Duration::hours(1),
        };
        let fl = SecureFLChannel::with_policy(make_channel(), "test-key".to_string(), policy);

        let weights = make_update("alice", vec![1.0]).weights;
        fl.encrypt_weights(&weights).unwrap();
        let err = fl.encrypt_weights(&weights).unwrap_err();
        assert!(matches!(err, TransportError::KeyExpired(_)));
    }

    #[test]
    fn test_max_age_zero_expires_immediately() {
        let policy = KeyPolicy {
            max_messages: 1000,
            max_age: chrono::Duration::zero(),
        };
        let fl = SecureFLChannel::with_policy(make_channel(), "test-key".to_string(), policy);

        let err = fl
            .encrypt_update(&make_update("alice", vec![1.0]))
            .unwrap_err();
        assert!(matches!(err, TransportError::KeyExpired(_)));
    }

    #[test]
    fn test_decryption_not_limited_by_policy() {
        // The server must still be able to decrypt in-flight messages
        // after the sender's key policy has expired.
        let policy = KeyPolicy {
            max_messages: 1,
            max_age: chrono::Duration::hours(1),
        };
        let fl = SecureFLChannel::with_policy(make_channel(), "test-key".to_string(), policy);
        let update = make_update("alice", vec![1.0]);

        let encrypted = fl.encrypt_update(&update).unwrap();
        // Encryption budget is now exhausted...
        assert!(fl.encrypt_update(&update).is_err());
        // ...but decryption still works.
        let decrypted = fl.decrypt_update(&encrypted).unwrap();
        assert_eq!(decrypted.client_id, "alice");
    }

    #[test]
    fn test_default_policy_values() {
        let policy = KeyPolicy::default();
        assert_eq!(policy.max_messages, 1000);
        assert_eq!(policy.max_age, chrono::Duration::hours(1));
    }
}
