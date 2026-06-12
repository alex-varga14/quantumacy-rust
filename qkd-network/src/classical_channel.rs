//! HMAC-authenticated classical channel for QKD.
//!
//! QKD's hard requirement is an *authenticated* classical channel: without
//! it, BB84 (and every other prepare-and-measure protocol) is vulnerable to a
//! full man-in-the-middle even with perfect quantum hardware, because Eve can
//! simply run two independent QKD sessions, one with each party. The
//! classical channel does not need to be confidential — basis announcements
//! and parities are public — but every message must be integrity-protected
//! with a key Eve does not hold.
//!
//! This module frames serde-encoded messages over any async byte stream
//! (normally a [`tokio::net::TcpStream`]) as
//!
//! ```text
//! [u32 BE body length][JSON payload][32-byte HMAC-SHA256 tag]
//! ```
//!
//! where the tag is computed over `direction-sequence-number || payload`
//! with a pre-shared authentication key. The per-direction sequence number
//! is implicit (both sides count), so replayed, reordered, or dropped frames
//! also fail verification. Tag verification uses the `hmac` crate's
//! constant-time comparison; any failure is a hard error and the channel
//! must be abandoned.

use crate::{NetworkError, NetworkResult};
use hmac::{Hmac, Mac};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::Sha256;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

type HmacSha256 = Hmac<Sha256>;

/// HMAC-SHA256 tag length in bytes.
pub const TAG_LEN: usize = 32;

/// Upper bound on a single frame body (payload + tag), to keep a malicious
/// or corrupted length prefix from forcing huge allocations.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// A framed, HMAC-authenticated message channel over an async byte stream.
///
/// Both ends must construct it with the same pre-shared authentication key.
/// The key authenticates only — frames are not encrypted, which matches the
/// QKD model (classical traffic is public but must be untamperable).
pub struct AuthenticatedChannel<S> {
    stream: S,
    auth_key: Vec<u8>,
    send_seq: u64,
    recv_seq: u64,
}

impl<S: AsyncRead + AsyncWrite + Unpin> AuthenticatedChannel<S> {
    pub fn new(stream: S, auth_key: &[u8]) -> Self {
        Self {
            stream,
            auth_key: auth_key.to_vec(),
            send_seq: 0,
            recv_seq: 0,
        }
    }

    fn mac(&self, seq: u64, payload: &[u8]) -> NetworkResult<HmacSha256> {
        let mut mac = HmacSha256::new_from_slice(&self.auth_key)
            .map_err(|e| NetworkError::Authentication(format!("Invalid MAC key: {e}")))?;
        mac.update(&seq.to_be_bytes());
        mac.update(payload);
        Ok(mac)
    }

    /// Serialize, tag, and send one message.
    pub async fn send<T: Serialize>(&mut self, msg: &T) -> NetworkResult<()> {
        let payload = serde_json::to_vec(msg)?;
        let tag = self.mac(self.send_seq, &payload)?.finalize().into_bytes();

        let body_len = payload.len() + TAG_LEN;
        if body_len > MAX_FRAME_BYTES {
            return Err(NetworkError::Protocol(format!(
                "Frame too large: {body_len} bytes (max {MAX_FRAME_BYTES})"
            )));
        }

        self.stream
            .write_all(&(body_len as u32).to_be_bytes())
            .await?;
        self.stream.write_all(&payload).await?;
        self.stream.write_all(&tag).await?;
        self.stream.flush().await?;
        self.send_seq += 1;
        Ok(())
    }

    /// Receive and verify one message. A bad tag is a hard
    /// [`NetworkError::Authentication`] error: the channel must be dropped.
    pub async fn recv<T: DeserializeOwned>(&mut self) -> NetworkResult<T> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let body_len = u32::from_be_bytes(len_buf) as usize;

        if body_len > MAX_FRAME_BYTES {
            return Err(NetworkError::Protocol(format!(
                "Frame too large: {body_len} bytes (max {MAX_FRAME_BYTES})"
            )));
        }
        if body_len < TAG_LEN {
            return Err(NetworkError::Authentication(
                "Frame shorter than authentication tag".into(),
            ));
        }

        let mut body = vec![0u8; body_len];
        self.stream.read_exact(&mut body).await?;
        let (payload, tag) = body.split_at(body_len - TAG_LEN);

        // Constant-time verification via hmac's Mac::verify_slice.
        self.mac(self.recv_seq, payload)?
            .verify_slice(tag)
            .map_err(|_| {
                NetworkError::Authentication(
                    "HMAC verification failed — message forged, tampered, or replayed".into(),
                )
            })?;
        self.recv_seq += 1;

        Ok(serde_json::from_slice(payload)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestMsg {
        kind: String,
        bits: Vec<u8>,
    }

    fn msg() -> TestMsg {
        TestMsg {
            kind: "parity".into(),
            bits: vec![1, 0, 1, 1],
        }
    }

    #[tokio::test]
    async fn test_round_trip_with_shared_key() {
        let (a, b) = tokio::io::duplex(64 * 1024);
        let mut alice = AuthenticatedChannel::new(a, b"pre-shared-auth-key");
        let mut bob = AuthenticatedChannel::new(b, b"pre-shared-auth-key");

        alice.send(&msg()).await.unwrap();
        let got: TestMsg = bob.recv().await.unwrap();
        assert_eq!(got, msg());

        // And in the other direction, repeatedly (sequence numbers advance).
        for _ in 0..3 {
            bob.send(&msg()).await.unwrap();
            let got: TestMsg = alice.recv().await.unwrap();
            assert_eq!(got, msg());
        }
    }

    #[tokio::test]
    async fn test_wrong_key_is_hard_error() {
        let (a, b) = tokio::io::duplex(64 * 1024);
        let mut alice = AuthenticatedChannel::new(a, b"correct-key");
        let mut bob = AuthenticatedChannel::new(b, b"wrong-key");

        alice.send(&msg()).await.unwrap();
        let err = bob.recv::<TestMsg>().await.unwrap_err();
        assert!(matches!(err, NetworkError::Authentication(_)), "{err:?}");
    }

    #[tokio::test]
    async fn test_tampered_frame_is_hard_error() {
        // Capture a valid frame, flip one payload byte in transit, and
        // verify the receiver rejects it.
        let (a, mut b) = tokio::io::duplex(64 * 1024);
        let mut alice = AuthenticatedChannel::new(a, b"shared-key");
        alice.send(&msg()).await.unwrap();

        let mut len_buf = [0u8; 4];
        b.read_exact(&mut len_buf).await.unwrap();
        let body_len = u32::from_be_bytes(len_buf) as usize;
        let mut body = vec![0u8; body_len];
        b.read_exact(&mut body).await.unwrap();
        body[2] ^= 0x01; // tamper one payload byte

        let (c, d) = tokio::io::duplex(64 * 1024);
        let mut wire = c;
        wire.write_all(&len_buf).await.unwrap();
        wire.write_all(&body).await.unwrap();

        let mut receiver = AuthenticatedChannel::new(d, b"shared-key");
        let err = receiver.recv::<TestMsg>().await.unwrap_err();
        assert!(matches!(err, NetworkError::Authentication(_)), "{err:?}");
    }

    #[tokio::test]
    async fn test_replayed_frame_is_hard_error() {
        // Send the same valid frame twice: the second copy fails because the
        // receiver's sequence number has advanced.
        let (a, mut b) = tokio::io::duplex(64 * 1024);
        let mut alice = AuthenticatedChannel::new(a, b"shared-key");
        alice.send(&msg()).await.unwrap();

        let mut frame = Vec::new();
        let mut len_buf = [0u8; 4];
        b.read_exact(&mut len_buf).await.unwrap();
        let body_len = u32::from_be_bytes(len_buf) as usize;
        let mut body = vec![0u8; body_len];
        b.read_exact(&mut body).await.unwrap();
        frame.extend_from_slice(&len_buf);
        frame.extend_from_slice(&body);

        let (mut c, d) = tokio::io::duplex(64 * 1024);
        c.write_all(&frame).await.unwrap();
        c.write_all(&frame).await.unwrap(); // replay

        let mut receiver = AuthenticatedChannel::new(d, b"shared-key");
        let first: TestMsg = receiver.recv().await.unwrap();
        assert_eq!(first, msg());
        let err = receiver.recv::<TestMsg>().await.unwrap_err();
        assert!(matches!(err, NetworkError::Authentication(_)), "{err:?}");
    }
}
