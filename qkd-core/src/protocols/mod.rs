//! QKD protocol trait and implementations.

pub mod bb84;
pub mod six_state;
pub mod b92;

use crate::channel::ChannelConfig;
use crate::error::QkdResult;
use crate::types::{QkdStats, SecureKey};

/// Common trait for all QKD protocols
pub trait QkdProtocol: Send + Sync {
    /// Run the full protocol and produce a secure key
    fn execute(&self, num_qubits: usize, channel: &ChannelConfig) -> QkdResult<(SecureKey, QkdStats)>;

    /// QBER threshold above which eavesdropping is assumed
    fn qber_threshold(&self) -> f64;

    /// Protocol name for logging/display
    fn name(&self) -> &'static str;
}
