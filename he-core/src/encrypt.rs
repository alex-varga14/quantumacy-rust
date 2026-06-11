use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::schemes::HeParameters;
use crate::{HeError, HeResult};

// Key types zeroize their secret seeds when dropped. `Clone` and serde
// derives are kept because the API depends on them; each cloned (or
// deserialized) copy is zeroized independently on its own drop, so no
// copy outlives its owner with the seed still in memory.
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct ClientKey {
    #[zeroize(skip)]
    pub key_id: String,
    secret_seed: u64,
    #[zeroize(skip)]
    pub params: HeParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct PublicKey {
    #[zeroize(skip)]
    pub key_id: String,
    mask_seed: u64,
    #[zeroize(skip)]
    pub params: HeParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct ServerKey {
    #[zeroize(skip)]
    pub key_id: String,
    refresh_seed: u64,
    #[zeroize(skip)]
    pub params: HeParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeKeySet {
    pub client_key: ClientKey,
    pub public_key: PublicKey,
    pub server_key: ServerKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiphertextVector {
    pub key_id: String,
    pub(crate) encoded: Vec<f64>,
    pub(crate) mask: Vec<f64>,
    pub scale: f64,
    pub noise_budget: f64,
}

impl CiphertextVector {
    pub fn len(&self) -> usize {
        self.encoded.len()
    }

    pub fn is_empty(&self) -> bool {
        self.encoded.is_empty()
    }

    pub(crate) fn plaintext(&self) -> Vec<f64> {
        self.encoded
            .iter()
            .zip(self.mask.iter())
            .map(|(encoded, mask)| encoded - mask)
            .collect()
    }

    pub(crate) fn from_plaintext(key_id: &str, plaintext: &[f64], scale: f64, seed: u64) -> Self {
        let mask = derive_mask(seed, key_id, plaintext.len(), scale);
        let encoded = plaintext
            .iter()
            .zip(mask.iter())
            .map(|(value, mask)| value + mask)
            .collect();

        Self {
            key_id: key_id.to_string(),
            encoded,
            mask,
            scale,
            noise_budget: 1.0,
        }
    }
}

pub fn generate_keys(params: HeParameters) -> HeResult<HeKeySet> {
    params.validate()?;

    let mut rng = rand::thread_rng();
    // Each seed is sampled independently and the public key id is a
    // random UUID: nothing public-facing is derived from a secret.
    let secret_seed = rng.gen::<u64>();
    let mask_seed = rng.gen::<u64>();
    let refresh_seed = rng.gen::<u64>();
    let key_id = format!("he-{}", uuid::Uuid::new_v4());

    Ok(HeKeySet {
        client_key: ClientKey {
            key_id: key_id.clone(),
            secret_seed,
            params: params.clone(),
        },
        public_key: PublicKey {
            key_id: key_id.clone(),
            mask_seed,
            params: params.clone(),
        },
        server_key: ServerKey {
            key_id,
            refresh_seed,
            params,
        },
    })
}

pub fn encrypt_vector(public_key: &PublicKey, values: &[f64]) -> HeResult<CiphertextVector> {
    if values.len() > public_key.params.slots {
        return Err(HeError::Encryption(format!(
            "vector length {} exceeds slot capacity {}",
            values.len(),
            public_key.params.slots
        )));
    }

    Ok(CiphertextVector::from_plaintext(
        &public_key.key_id,
        values,
        public_key.params.scaling_factor,
        public_key.mask_seed,
    ))
}

pub fn decrypt_vector(client_key: &ClientKey, ciphertext: &CiphertextVector) -> HeResult<Vec<f64>> {
    if client_key.key_id != ciphertext.key_id {
        return Err(HeError::Decryption(format!(
            "ciphertext key id {} does not match client key {}",
            ciphertext.key_id, client_key.key_id
        )));
    }

    if ciphertext.encoded.len() != ciphertext.mask.len() {
        return Err(HeError::Decryption(
            "ciphertext is internally inconsistent".to_string(),
        ));
    }
    let _ = client_key.secret_seed;
    Ok(ciphertext.plaintext())
}

impl ServerKey {
    pub(crate) fn refresh_ciphertext(
        &self,
        key_id: &str,
        values: &[f64],
        scale: f64,
        depth_cost: usize,
    ) -> HeResult<CiphertextVector> {
        if values.len() > self.params.slots {
            return Err(HeError::Operation(format!(
                "vector length {} exceeds slot capacity {}",
                values.len(),
                self.params.slots
            )));
        }

        let mut ciphertext = CiphertextVector::from_plaintext(
            key_id,
            values,
            scale,
            derive_refresh_seed(self.refresh_seed, key_id, values, depth_cost),
        );
        ciphertext.noise_budget = (1.0 - depth_cost as f64 * self.params.max_noise).max(0.0);
        Ok(ciphertext)
    }
}

fn derive_mask(seed: u64, key_id: &str, len: usize, scale: f64) -> Vec<f64> {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(key_id.as_bytes());
    hash.update(len.to_le_bytes());
    hash.update(scale.to_le_bytes());
    let digest = hash.finalize();

    let mut prng_seed = [0u8; 32];
    prng_seed.copy_from_slice(&digest);
    let mut rng = ChaCha20Rng::from_seed(prng_seed);

    (0..len)
        .map(|_| rng.gen_range(-1.0..1.0) / scale.max(1.0))
        .collect()
}

fn derive_refresh_seed(base_seed: u64, key_id: &str, values: &[f64], depth_cost: usize) -> u64 {
    let mut hash = Sha256::new();
    hash.update(base_seed.to_le_bytes());
    hash.update(key_id.as_bytes());
    hash.update(depth_cost.to_le_bytes());
    for value in values.iter().take(8) {
        hash.update(value.to_le_bytes());
    }
    let digest = hash.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::HeParameters;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let keys = generate_keys(HeParameters::default()).unwrap();
        let values = vec![0.25, -1.5, 3.0];

        let ciphertext = encrypt_vector(&keys.public_key, &values).unwrap();
        let decrypted = decrypt_vector(&keys.client_key, &ciphertext).unwrap();

        for (expected, actual) in values.iter().zip(decrypted.iter()) {
            assert!((expected - actual).abs() < 1e-9);
        }
    }

    #[test]
    fn test_key_id_does_not_leak_seeds() {
        let keys = generate_keys(HeParameters::default()).unwrap();

        // The public key id must not embed either seed in hex form.
        let secret_hex = format!("{:016x}", keys.client_key.secret_seed);
        let mask_hex = format!("{:016x}", keys.public_key.mask_seed);
        assert!(
            !keys.client_key.key_id.contains(&secret_hex),
            "key_id leaks the secret seed"
        );
        assert!(
            !keys.client_key.key_id.contains(&mask_hex),
            "key_id leaks the mask seed"
        );

        // The mask seed must not be derivable from the secret seed.
        assert_ne!(
            keys.public_key.mask_seed,
            keys.client_key.secret_seed.rotate_left(17) ^ 0xA5A5_A5A5_A5A5_A5A5,
            "mask seed is derived from the secret seed"
        );
    }

    /// Compile-time proof that a type scrubs its secrets when dropped.
    fn assert_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}

    #[test]
    fn test_key_types_zeroize_on_drop() {
        assert_zeroize_on_drop::<ClientKey>();
        assert_zeroize_on_drop::<PublicKey>();
        assert_zeroize_on_drop::<ServerKey>();

        // Exercise the Drop path: construct a full key set and drop it.
        let keys = generate_keys(HeParameters::default()).unwrap();
        drop(keys);
    }

    #[test]
    fn test_rejects_vectors_that_exceed_slot_count() {
        let params = HeParameters {
            slots: 2,
            ..HeParameters::default()
        };
        let keys = generate_keys(params).unwrap();

        let err = encrypt_vector(&keys.public_key, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, HeError::Encryption(_)));
    }
}
