//! Secure channel using QKD-derived keys with AES-256-GCM encryption.
//!
//! Provides authenticated encryption for data transmitted between
//! QKD peers. Keys are consumed on use (one-time pad discipline).

use crate::{NetworkError, NetworkResult};
use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, KeyInit, Nonce};
use qkd_core::types::SecureKey;
use serde::{Deserialize, Serialize};

/// AES-256-GCM nonce length in bytes (96 bits)
const NONCE_LEN: usize = 12;

/// Encrypted message with nonce for AES-GCM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMessage {
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
    pub key_id: String,
}

/// A secure channel backed by a QKD key
pub struct SecureChannel {
    cipher: Aes256Gcm,
    key_id: String,
}

impl SecureChannel {
    /// Create a new secure channel from a QKD key.
    /// The key material must be at least 32 bytes (256 bits).
    pub fn from_key(key: &SecureKey) -> NetworkResult<Self> {
        if key.material.len() < 32 {
            return Err(NetworkError::Encryption(format!(
                "Key too short: {} bytes, need 32",
                key.material.len()
            )));
        }

        let aes_key = Key::<Aes256Gcm>::from_slice(&key.material[..32]);
        let cipher = Aes256Gcm::new(aes_key);

        Ok(Self {
            cipher,
            key_id: key.key_id.clone(),
        })
    }

    /// Encrypt a plaintext message
    pub fn encrypt(&self, plaintext: &[u8]) -> NetworkResult<EncryptedMessage> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| NetworkError::Encryption(format!("Encrypt failed: {e}")))?;

        Ok(EncryptedMessage {
            nonce: nonce.to_vec(),
            ciphertext,
            key_id: self.key_id.clone(),
        })
    }

    /// Decrypt an encrypted message
    pub fn decrypt(&self, msg: &EncryptedMessage) -> NetworkResult<Vec<u8>> {
        // The nonce arrives from the network; from_slice panics on wrong length.
        if msg.nonce.len() != NONCE_LEN {
            return Err(NetworkError::Encryption(format!(
                "Invalid nonce length: {} bytes, need {NONCE_LEN}",
                msg.nonce.len()
            )));
        }
        let nonce = Nonce::from_slice(&msg.nonce);
        let plaintext = self
            .cipher
            .decrypt(nonce, msg.ciphertext.as_slice())
            .map_err(|e| NetworkError::Encryption(format!("Decrypt failed: {e}")))?;

        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_test_key() -> SecureKey {
        SecureKey {
            key_id: "test-key".to_string(),
            timestamp: Utc::now(),
            material: vec![0x42; 32],
            length_bits: 256,
        }
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = make_test_key();
        let channel = SecureChannel::from_key(&key).unwrap();

        let plaintext = b"Hello, quantum-safe world!";
        let encrypted = channel.encrypt(plaintext).unwrap();
        let decrypted = channel.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_different_nonces() {
        let key = make_test_key();
        let channel = SecureChannel::from_key(&key).unwrap();

        let msg = b"test message";
        let enc1 = channel.encrypt(msg).unwrap();
        let enc2 = channel.encrypt(msg).unwrap();

        // Different nonces → different ciphertexts
        assert_ne!(enc1.ciphertext, enc2.ciphertext);
        assert_ne!(enc1.nonce, enc2.nonce);
    }

    #[test]
    fn test_malformed_nonce_returns_error_not_panic() {
        let key = make_test_key();
        let channel = SecureChannel::from_key(&key).unwrap();

        let mut msg = channel.encrypt(b"payload").unwrap();
        msg.nonce = vec![0u8; 5]; // attacker-controlled, wrong length

        assert!(channel.decrypt(&msg).is_err());
    }

    #[test]
    fn test_key_too_short() {
        let key = SecureKey {
            key_id: "short".to_string(),
            timestamp: Utc::now(),
            material: vec![0x42; 16], // Only 128 bits
            length_bits: 128,
        };
        assert!(SecureChannel::from_key(&key).is_err());
    }
}
