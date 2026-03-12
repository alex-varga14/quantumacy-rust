//! Thread-safe key manager with automatic expiration and zeroization.
//!
//! Stores derived QKD keys with configurable TTL and provides
//! lookup, rotation, and secure deletion.

use crate::error::{QkdError, QkdResult};
use crate::types::SecureKey;
use chrono::{Duration, Utc};
use dashmap::DashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for key management
#[derive(Debug, Clone)]
pub struct KeyManagerConfig {
    /// Maximum number of keys to store
    pub max_keys: usize,
    /// Key time-to-live
    pub key_ttl: Duration,
    /// Whether to auto-expire keys
    pub auto_expire: bool,
}

impl Default for KeyManagerConfig {
    fn default() -> Self {
        Self {
            max_keys: 1000,
            key_ttl: Duration::hours(1),
            auto_expire: true,
        }
    }
}

/// Thread-safe key store with automatic expiration
#[derive(Clone)]
pub struct KeyManager {
    keys: Arc<DashMap<String, SecureKey>>,
    config: KeyManagerConfig,
}

impl KeyManager {
    pub fn new(config: KeyManagerConfig) -> Self {
        Self {
            keys: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Store a new key. Evicts expired keys if at capacity.
    pub fn store(&self, key: SecureKey) -> QkdResult<String> {
        if self.config.auto_expire {
            self.evict_expired();
        }

        if self.keys.len() >= self.config.max_keys {
            // Evict oldest key
            if let Some(oldest) = self
                .keys
                .iter()
                .min_by_key(|entry| entry.value().timestamp)
            {
                let id = oldest.key().clone();
                drop(oldest);
                self.keys.remove(&id);
                debug!(key_id = %id, "Evicted oldest key");
            }
        }

        let key_id = key.key_id.clone();
        info!(key_id = %key_id, bits = key.length_bits, "Stored new key");
        self.keys.insert(key_id.clone(), key);
        Ok(key_id)
    }

    /// Retrieve a key by ID. Returns error if expired or not found.
    pub fn get(&self, key_id: &str) -> QkdResult<SecureKey> {
        let entry = self
            .keys
            .get(key_id)
            .ok_or_else(|| QkdError::KeyNotFound(key_id.to_string()))?;

        if self.config.auto_expire {
            let age = Utc::now() - entry.value().timestamp;
            if age > self.config.key_ttl {
                drop(entry);
                self.keys.remove(key_id);
                return Err(QkdError::KeyExpired(key_id.to_string()));
            }
        }

        Ok(entry.value().clone())
    }

    /// Consume and remove a key (one-time use)
    pub fn consume(&self, key_id: &str) -> QkdResult<SecureKey> {
        let (_, key) = self
            .keys
            .remove(key_id)
            .ok_or_else(|| QkdError::KeyNotFound(key_id.to_string()))?;

        if self.config.auto_expire {
            let age = Utc::now() - key.timestamp;
            if age > self.config.key_ttl {
                return Err(QkdError::KeyExpired(key_id.to_string()));
            }
        }

        info!(key_id = %key_id, "Key consumed");
        Ok(key)
    }

    /// Remove expired keys
    pub fn evict_expired(&self) {
        let now = Utc::now();
        let expired: Vec<String> = self
            .keys
            .iter()
            .filter(|entry| (now - entry.value().timestamp) > self.config.key_ttl)
            .map(|entry| entry.key().clone())
            .collect();

        for id in &expired {
            self.keys.remove(id);
        }

        if !expired.is_empty() {
            debug!(count = expired.len(), "Evicted expired keys");
        }
    }

    /// Current number of stored keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// List all key IDs (for management/debugging)
    pub fn list_keys(&self) -> Vec<String> {
        self.keys.iter().map(|e| e.key().clone()).collect()
    }

    /// Securely delete all keys
    pub fn clear(&self) {
        let count = self.keys.len();
        self.keys.clear();
        info!(count, "Cleared all keys");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(id: &str) -> SecureKey {
        SecureKey {
            key_id: id.to_string(),
            timestamp: Utc::now(),
            material: vec![0xAB; 32],
            length_bits: 256,
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let mgr = KeyManager::new(KeyManagerConfig::default());
        let key = make_key("test-1");
        mgr.store(key.clone()).unwrap();

        let retrieved = mgr.get("test-1").unwrap();
        assert_eq!(retrieved.key_id, "test-1");
        assert_eq!(retrieved.material, vec![0xAB; 32]);
    }

    #[test]
    fn test_consume_removes_key() {
        let mgr = KeyManager::new(KeyManagerConfig::default());
        mgr.store(make_key("test-2")).unwrap();

        let consumed = mgr.consume("test-2").unwrap();
        assert_eq!(consumed.key_id, "test-2");

        assert!(mgr.get("test-2").is_err());
    }

    #[test]
    fn test_not_found() {
        let mgr = KeyManager::new(KeyManagerConfig::default());
        assert!(matches!(
            mgr.get("nonexistent"),
            Err(QkdError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_capacity_eviction() {
        let config = KeyManagerConfig {
            max_keys: 2,
            key_ttl: Duration::hours(1),
            auto_expire: false,
        };
        let mgr = KeyManager::new(config);

        mgr.store(make_key("a")).unwrap();
        mgr.store(make_key("b")).unwrap();
        mgr.store(make_key("c")).unwrap(); // Should evict oldest

        assert_eq!(mgr.key_count(), 2);
    }
}
