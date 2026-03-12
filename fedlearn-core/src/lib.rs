//! # Federated Learning Core
//!
//! Provides federated learning primitives:
//! - **FedAvg**: Federated Averaging aggregation algorithm
//! - **Model**: Trait-based model abstraction for any ML framework
//! - **Privacy**: Differential privacy via gradient clipping and noise
//! - **Client/Server**: Roles for federated training rounds
//!
//! Designed to be ML-framework-agnostic; the `Model` trait can be
//! implemented for Candle, Burn, or any tensor library.

pub mod aggregation;
pub mod model;
pub mod privacy;
pub mod round;
pub mod error;

pub use error::{FedError, FedResult};
pub use model::{ModelUpdate, ModelWeights};
pub use aggregation::FedAvg;
pub use privacy::DifferentialPrivacy;
