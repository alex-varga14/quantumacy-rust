//! # HE Inference
//!
//! Encrypted model inference engine. Runs neural network forward
//! passes on homomorphically encrypted data.

pub mod model;
pub mod server;

pub use he_core::{Activation, CiphertextVector, HeError, HeKeySet, HeResult};
pub use model::{DenseLayer, EncryptedModel};
pub use server::{EncryptedInferenceService, SessionStatus};
