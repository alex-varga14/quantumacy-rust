//! # QKD Core
//!
//! Quantum Key Distribution protocol implementations for the Quantumacy project.
//! Provides BB84, Six-State, and B92 protocol simulators with realistic noise models,
//! error correction (CASCADE), and privacy amplification.
//!
//! ## Architecture
//!
//! - **Protocols**: Pluggable QKD protocol implementations behind a common trait
//! - **Channel**: Quantum channel simulation with configurable noise/eavesdropping
//! - **Error Correction**: CASCADE protocol for sifted key reconciliation
//! - **Privacy Amplification**: Universal hashing to distill secure final keys
//! - **Key Manager**: Thread-safe key storage with automatic zeroization

pub mod channel;
pub mod error;
pub mod error_correction;
pub mod key_manager;
pub mod privacy_amplification;
pub mod protocols;
pub mod types;

pub use error::{QkdError, QkdResult};
pub use key_manager::KeyManager;
pub use types::*;
