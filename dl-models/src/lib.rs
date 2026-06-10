//! # Deep Learning Models
//!
//! Medical imaging models for federated training and encrypted
//! inference.

// The dense backprop kernels in `common.rs` use index-based loops to keep the
// row/column math readable and to mirror standard ML pseudocode. Rewriting them
// as `enumerate()`/zip combinations is more idiomatic but obscures the math
// without changing behaviour, so we silence the related style lints.
#![allow(clippy::needless_range_loop, clippy::vec_init_then_push)]

pub mod chestscan;
mod common;
pub mod histology;

pub use chestscan::ChestScanModel;
pub use histology::HistologyModel;
