//! # Federated Learning Transport
//!
//! QKD-secured gRPC transport layer for federated learning.
//! Provides secure communication between FL clients and the aggregation server
//! using quantum-derived keys for authenticated encryption.

pub mod secure_channel;
pub mod grpc_service;
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
}

pub type TransportResult<T> = Result<T, TransportError>;
