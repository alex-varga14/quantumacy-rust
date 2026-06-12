use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Qubit polarization basis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Basis {
    /// Rectilinear basis (horizontal/vertical): |0⟩, |1⟩
    Rectilinear,
    /// Diagonal basis: |+⟩, |-⟩
    Diagonal,
    /// Circular basis (used in Six-State): |L⟩, |R⟩
    Circular,
}

/// Qubit measurement outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QubitValue {
    Zero,
    One,
}

impl QubitValue {
    pub fn as_bit(&self) -> u8 {
        match self {
            QubitValue::Zero => 0,
            QubitValue::One => 1,
        }
    }
}

/// A prepared qubit with basis and value
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Qubit {
    pub basis: Basis,
    pub value: QubitValue,
}

/// Sifted key material before error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiftedKey {
    pub bits: Vec<u8>,
    pub length: usize,
}

/// Final distilled key with zeroization on drop
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct SecureKey {
    pub key_id: String,
    #[zeroize(skip)]
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub material: Vec<u8>,
    pub length_bits: usize,
}

/// Session metadata for a QKD exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QkdSession {
    pub session_id: String,
    pub protocol: ProtocolType,
    pub num_qubits: usize,
    pub qber: f64,
    pub sifted_key_length: usize,
    pub final_key_length: usize,
    pub eavesdropping_detected: bool,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Supported QKD protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolType {
    BB84,
    SixState,
    B92,
}

/// Statistics from a QKD run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QkdStats {
    pub protocol: ProtocolType,
    pub qubits_sent: usize,
    pub qubits_received: usize,
    pub sifting_rate: f64,
    pub qber: f64,
    pub raw_key_bits: usize,
    pub final_key_bits: usize,
    pub key_rate: f64,
    pub eavesdropping_detected: bool,
    /// Parity bits revealed on the classical channel during error
    /// correction (subtracted during privacy amplification).
    #[serde(default)]
    pub leaked_bits: usize,
}
