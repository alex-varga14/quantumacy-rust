//! # HE Inference
//!
//! Encrypted model inference engine. Runs neural network forward
//! passes on homomorphically encrypted data.
//!
//! ## Status: Phase 4 (Scaffolded)

pub mod model;
pub mod server;

pub use he_core::HeError;
