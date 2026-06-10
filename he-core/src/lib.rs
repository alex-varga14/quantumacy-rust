//! # Homomorphic Encryption Core
//!
//! Simulation-grade homomorphic encryption primitives for
//! privacy-preserving machine learning inference.
//!
//! <div class="warning">
//!
//! **⚠️ NOT CRYPTOGRAPHICALLY SECURE.** This crate is a *simulation* of a
//! CKKS-style HE workflow, not an implementation of homomorphic encryption.
//! Ciphertexts produced here provide **zero confidentiality**: the masking
//! values are carried inside the ciphertext itself and decryption does not
//! depend on the secret key. Its purpose is to provide a stable API surface
//! for research and integration work until the internals are replaced with a
//! real HE backend (e.g. `tfhe-rs`). Never use this crate to protect real
//! data. See `SECURITY.md` at the repository root.
//!
//! </div>
//!
//! This crate intentionally models the shape of a CKKS-style API while
//! remaining dependency-light and testable in constrained environments.

pub mod encrypt;
pub mod operations;
pub mod schemes;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HeError {
    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Decryption error: {0}")]
    Decryption(String),

    #[error("Key generation error: {0}")]
    KeyGen(String),

    #[error("Operation error: {0}")]
    Operation(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

pub type HeResult<T> = Result<T, HeError>;

pub use encrypt::{
    decrypt_vector, encrypt_vector, generate_keys, CiphertextVector, ClientKey, HeKeySet,
    PublicKey, ServerKey,
};
pub use operations::{
    add_ciphertexts, add_plaintext, apply_activation, apply_polynomial, linear_layer,
    multiply_ciphertexts, multiply_plaintext,
};
pub use schemes::{Activation, HeParameters, HeScheme};
