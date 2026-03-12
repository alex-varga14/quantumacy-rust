use thiserror::Error;

pub type QkdResult<T> = Result<T, QkdError>;

#[derive(Error, Debug)]
pub enum QkdError {
    #[error("QBER {qber:.4} exceeds threshold {threshold:.4} — eavesdropping likely")]
    EavesdroppingDetected { qber: f64, threshold: f64 },

    #[error("Insufficient sifted key bits: got {got}, need {need}")]
    InsufficientKeyBits { got: usize, need: usize },

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Key expired: {0}")]
    KeyExpired(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Error correction failed: {0}")]
    ErrorCorrection(String),

    #[error("Privacy amplification failed: {0}")]
    PrivacyAmplification(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
