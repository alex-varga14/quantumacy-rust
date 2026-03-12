//! # Deep Learning Models
//!
//! Medical imaging models for federated training and encrypted
//! inference.

mod common;
pub mod chestscan;
pub mod histology;

pub use chestscan::ChestScanModel;
pub use histology::HistologyModel;
