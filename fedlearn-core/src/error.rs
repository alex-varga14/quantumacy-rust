use thiserror::Error;

pub type FedResult<T> = Result<T, FedError>;

#[derive(Error, Debug)]
pub enum FedError {
    #[error("No client updates received for aggregation")]
    NoUpdates,

    #[error("Model dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Training failed: {0}")]
    Training(String),

    #[error("Aggregation failed: {0}")]
    Aggregation(String),

    #[error("Client {client_id} error: {message}")]
    Client { client_id: String, message: String },

    #[error("Round {round} timed out")]
    Timeout { round: u32 },

    #[error("Privacy budget exhausted (ε = {epsilon:.4})")]
    PrivacyBudgetExhausted { epsilon: f64 },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
