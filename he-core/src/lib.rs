//! # Homomorphic Encryption Core
//!
//! Wrapper around tfhe-rs providing CKKS scheme operations for
//! privacy-preserving machine learning inference.
//!
//! ## Status: Phase 4 (Scaffolded)
//!
//! Will implement:
//! - CKKS scheme for approximate encrypted arithmetic
//! - Key generation, encryption, decryption services
//! - Encrypted tensor operations for neural network inference
//! - Three-party architecture (client, storage, processing)

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
