//! # Homomorphic Encryption Core
//!
//! Simulation-grade homomorphic encryption primitives for
//! privacy-preserving machine learning inference.
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
    CiphertextVector,
    ClientKey,
    HeKeySet,
    PublicKey,
    ServerKey,
    decrypt_vector,
    encrypt_vector,
    generate_keys,
};
pub use operations::{
    add_ciphertexts,
    add_plaintext,
    apply_activation,
    apply_polynomial,
    linear_layer,
    multiply_ciphertexts,
    multiply_plaintext,
};
pub use schemes::{Activation, HeParameters, HeScheme};
