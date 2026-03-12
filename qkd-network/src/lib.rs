//! # QKD Network
//!
//! Async networking layer for QKD key distribution.
//! Supports peer-to-peer and client-server architectures.
//!
//! ## Architecture
//!
//! - **Server**: Manages QKD sessions and distributes keys to clients
//! - **Client**: Requests and receives QKD keys
//! - **P2P**: Direct key exchange between two peers
//! - **Secure Channel**: AES-GCM encrypted channel using QKD-derived keys

pub mod server;
pub mod client;
pub mod p2p;
pub mod secure_channel;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("RPC error: {0}")]
    Rpc(#[from] tonic::Status),

    #[error("QKD error: {0}")]
    Qkd(#[from] qkd_core::QkdError),

    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type NetworkResult<T> = Result<T, NetworkError>;
