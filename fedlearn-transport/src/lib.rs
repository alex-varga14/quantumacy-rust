//! # Federated Learning Transport
//!
//! QKD-secured gRPC transport layer for federated learning.
//! Provides secure communication between FL clients and the aggregation server
//! using quantum-derived keys for authenticated encryption.

// `tonic::Status` and `qkd_network::NetworkError` are inherently large variants.
// Boxing the error type would force API churn through every consumer; the MVP
// keeps the existing surface and accepts the larger `Result`.
#![allow(clippy::result_large_err)]

pub mod grpc_service;
pub mod secure_channel;
pub mod tls;
pub mod proto {
    tonic::include_proto!("quantumacy.fedlearn");
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("QKD network error: {0}")]
    QkdNetwork(#[from] qkd_network::NetworkError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type TransportResult<T> = Result<T, TransportError>;
