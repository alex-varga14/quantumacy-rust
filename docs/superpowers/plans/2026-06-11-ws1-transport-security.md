# Workstream 1: Transport Security (TLS, mTLS, authn/z) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close threats T1 (network attacker) and T2 (unauthorized client) from SECURITY_ROADMAP.md: all gRPC traffic runs over mTLS, clients are authenticated by CA-signed certificates bound to their client id, and raw QKD key material never leaves the server process.

**Architecture:** A new `tls` module in `fedlearn-transport` owns cert loading (env-configured), an mTLS-required server builder, and an authenticated-identity extractor that parses the client cert CN out of `Request::peer_certs()`. Sessions bind to the authenticated CN at registration; every later RPC re-checks it. `KeyExchange` responses carry HKDF-derived per-round keys instead of raw QKD material. Tests generate ephemeral CAs/certs with `rcgen` at runtime — no certs are ever committed to the repo.

**Tech Stack:** `tonic 0.11` with the `tls` feature (rustls under the hood), `rcgen` (dev-dep, test/demo certs), `x509-parser` (CN extraction from DER), `hkdf` + existing `sha2` (key derivation).

**Design deviation from SECURITY_ROADMAP.md:** the roadmap's HKDF input included TLS exporter material (RFC 5705); tonic does not expose it. Derivation here is `HKDF-SHA256(ikm = QKD key, salt = session_id, info = "quantumacy-fl-v1:round:{round}")`. Residual risk (documented in SECURITY.md when this lands): derived keys still transit the wire, TLS-protected; they are no longer the raw QKD key, and rotation bounds exposure to one round. True elimination needs distributed QKD endpoints (Workstream 5).

**Verification bar for every task:** `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check` all green before commit. Insecure plaintext mode stays available only behind `QUANTUMACY_INSECURE=1` with a loud `warn!`.

---

### Task 1: Dependencies and feature wiring

**Files:**
- Modify: `Cargo.toml` (workspace deps: add `hkdf = "0.12"`, `x509-parser = "0.16"`, `rcgen = "0.13"`; change `tonic = { version = "0.11", features = ["tls"] }`)
- Modify: `fedlearn-transport/Cargo.toml` (use workspace `hkdf`, `x509-parser`; `rcgen` under `[dev-dependencies]`)

- [x] **Step 1:** Add the dependencies as above.
- [x] **Step 2:** Run `cargo check --workspace --all-targets` — expect clean (tls feature compiles, nothing uses new deps yet). If a version doesn't resolve, relax the minor version until `cargo update -p <crate>` succeeds, keeping the same major.
- [x] **Step 3:** Commit: `chore(ws1): add tls/hkdf/x509 dependencies`

### Task 2: TLS settings module with env loading

**Files:**
- Create: `fedlearn-transport/src/tls.rs`
- Modify: `fedlearn-transport/src/lib.rs` (add `pub mod tls;`)

- [x] **Step 1: Write failing tests** (in `tls.rs` `#[cfg(test)]`): `TlsSettings::from_env_map` with all three paths set returns `TlsSettings::Mutual{..}`; with `QUANTUMACY_INSECURE=1` returns `TlsSettings::Insecure`; with partial paths returns an error naming the missing variable.
- [x] **Step 2:** Run `cargo test -p fedlearn-transport tls` — expect compile failure (module missing).
- [x] **Step 3: Implement.** Core shape:

```rust
pub enum TlsSettings {
    /// mTLS: server identity + required client CA.
    Mutual { cert_pem: Vec<u8>, key_pem: Vec<u8>, client_ca_pem: Vec<u8> },
    /// Plaintext, demo-only. Constructing this logs a warning at serve time.
    Insecure,
}

impl TlsSettings {
    /// Reads QUANTUMACY_TLS_CERT / _KEY / _CLIENT_CA (file paths) or
    /// QUANTUMACY_INSECURE=1. Pure function over a map so tests don't
    /// touch process env.
    pub fn from_env_map(vars: &HashMap<String, String>) -> TransportResult<Self> { .. }
    pub fn from_env() -> TransportResult<Self> { Self::from_env_map(&std::env::vars().collect()) }
}
```

File reads happen inside `from_env_map` (error includes the path). New `TransportError::Config(String)` variant if one doesn't exist.
- [x] **Step 4:** Tests pass; full verification bar; commit: `feat(ws1): TLS settings with env-based configuration`

### Task 3: Ephemeral test-cert generation helper

**Files:**
- Create: `fedlearn-transport/tests/common/mod.rs` (shared by integration tests)

- [x] **Step 1:** Implement `TestPki::generate()` using rcgen — a CA, a server cert for `localhost`, and `client_cert(cn: &str)` minting CA-signed client certs with the CN set:

```rust
pub struct TestPki { pub ca_pem: String, pub server_cert_pem: String, pub server_key_pem: String, ca_cert: rcgen::Certificate, ca_key: rcgen::KeyPair }
impl TestPki {
    pub fn generate() -> Self { /* CA via CertificateParams + IsCa::Ca(..).self_signed; server leaf for "localhost" signed_by CA */ }
    pub fn client_cert(&self, cn: &str) -> (String, String) { /* leaf with DnType::CommonName = cn, signed_by CA; returns (cert_pem, key_pem) */ }
    pub fn wrong_ca_client(cn: &str) -> (String, String, String) { /* fresh CA + leaf — for the rejection test */ }
}
```

(Adapt to the exact rcgen 0.13 API at compile time; the shape above is what matters.)
- [x] **Step 2:** Smoke test in the same module: generated PEMs are non-empty and parse via `x509_parser::pem`. Run it.
- [x] **Step 3:** Commit: `test(ws1): ephemeral PKI helper for transport tests`

### Task 4: mTLS server builder + TLS client helper

**Files:**
- Modify: `fedlearn-transport/src/tls.rs`
- Test: `fedlearn-transport/tests/tls_handshake.rs`

- [x] **Step 1: Failing integration test:** start the FL service on an ephemeral port with `TestPki` material via the new builder; (a) a client with CA-signed cert + CA root connects and `register` succeeds; (b) a plaintext client errors; (c) a `wrong_ca_client` fails the handshake.
- [x] **Step 2: Implement** in `tls.rs`:

```rust
impl TlsSettings {
    pub fn server_tls_config(&self) -> TransportResult<Option<ServerTlsConfig>> {
        match self {
            Self::Mutual { cert_pem, key_pem, client_ca_pem } => Ok(Some(
                ServerTlsConfig::new()
                    .identity(Identity::from_pem(cert_pem, key_pem))
                    .client_ca_root(Certificate::from_pem(client_ca_pem)),
            )),
            Self::Insecure => { warn!("QUANTUMACY_INSECURE=1: serving PLAINTEXT gRPC — demo only"); Ok(None) }
        }
    }
}
/// Client-side: CA root + client identity + domain "localhost" override for tests.
pub fn client_tls_config(ca_pem: &[u8], cert_pem: &[u8], key_pem: &[u8], domain: &str) -> ClientTlsConfig { .. }
```

Wire into a `serve(settings, addr, fl_svc, kx_svc)` helper used by both the binary and tests (`Server::builder()` + `.tls_config(..)` when `Some`).
- [x] **Step 3:** Tests pass; verification bar; commit: `feat(ws1): mTLS-required server builder and TLS client config`

### Task 5: Authenticated identity extraction (cert CN → client id)

**Files:**
- Modify: `fedlearn-transport/src/tls.rs` (add `peer_common_name`)
- Modify: `fedlearn-transport/src/grpc_service.rs`
- Test: extend `fedlearn-transport/tests/tls_handshake.rs`

- [x] **Step 1: Failing test:** over mTLS, `register` with `client_id` ≠ cert CN returns `PERMISSION_DENIED`; with matching CN it succeeds.
- [x] **Step 2: Implement:**

```rust
/// Extract the CN of the first peer certificate. None on plaintext connections.
pub fn peer_common_name<T>(request: &Request<T>) -> Option<String> {
    let certs = request.peer_certs()?;
    let der = certs.first()?;
    let (_, cert) = x509_parser::parse_x509_certificate(der.as_ref()).ok()?;
    cert.subject().iter_common_name().next()?.as_str().ok().map(str::to_owned)
}
```

In `register`: if `peer_common_name` is `Some(cn)` and `cn != request.client_id` → `PERMISSION_DENIED`. If `None` (plaintext/insecure mode) registration proceeds (insecure mode keeps MVP behavior). Store the authenticated flag in `SessionState { client_id, authenticated: bool }`.
- [x] **Step 3:** Tests pass; verification bar; commit: `feat(ws1): bind sessions to mTLS client certificate identity`

### Task 6: Authorization enforcement on every RPC

**Files:**
- Modify: `fedlearn-transport/src/grpc_service.rs`
- Test: extend `fedlearn-transport/tests/tls_handshake.rs`

- [x] **Step 1: Failing tests:** (a) client A (cert CN "alice") calling `submit_update`/`get_global_model`/`request_key` against client B's session id → `PERMISSION_DENIED`, even though A knows B's session id (today only the self-reported `client_id` field is checked); (b) `request_key` for a session not owned by the caller's CN → `PERMISSION_DENIED`.
- [x] **Step 2: Implement:** extend both `validate_session` impls to take the `&Request<T>` (or the extracted CN), and when the session is `authenticated`, require `peer_common_name(request) == Some(session.client_id)` — the self-reported field is no longer the trust anchor. Plaintext sessions (insecure mode) keep the old field check.
- [x] **Step 3:** Tests pass; verification bar; commit: `feat(ws1): enforce certificate identity on session-scoped RPCs`

### Task 7: HKDF-derived per-round keys replace raw QKD material

**Files:**
- Modify: `fedlearn-transport/src/grpc_service.rs` (`mint_key`)
- Modify: `fedlearn-transport/proto/fedlearn.proto` (KeyResponse: add `uint32 round`, `string derivation` fields)
- Test: extend `fedlearn-transport/tests/mvp_flow.rs`

- [x] **Step 1: Failing test:** the key material returned by `request_key` differs from the raw QKD key stored server-side (`qkd_server.get_key(key_id)`), has length 32, and two requests for different rounds derive different keys; both ends of the FL flow still round-trip an encrypted update (server re-derives the same key when decrypting).
- [x] **Step 2: Implement:**

```rust
fn derive_round_key(qkd_key: &SecureKey, session_id: &str, round: u32) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(session_id.as_bytes()), &qkd_key.material);
    let mut okm = [0u8; 32];
    hk.expand(format!("quantumacy-fl-v1:round:{round}").as_bytes(), &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}
```

`mint_key` returns the derived key; `decode_update` re-derives (it knows key_id + session + round) instead of using raw material. The raw QKD key never appears in any proto message.
- [x] **Step 3:** Tests pass; verification bar; commit: `feat(ws1): HKDF per-round key derivation; raw QKD keys stay server-side`

### Task 8: Server binary + demo wiring

**Files:**
- Modify: `fedlearn-transport/src/bin/server.rs` (use `TlsSettings::from_env`, refuse to start without TLS unless `QUANTUMACY_INSECURE=1`)
- Modify: `fedlearn-transport/examples/local_platform_demo.rs` (generate ephemeral PKI via rcgen at startup — rcgen is a dev-dep, available to examples — and run the whole demo over mTLS; `--insecure` flag for the old behavior)

- [x] **Step 1:** Implement both; demo prints which mode it runs in.
- [x] **Step 2:** Run `cargo run -p fedlearn-transport --example local_platform_demo -- --clients 4 --rounds 5` — full mTLS round summary, same numbers as before. Run the binary without env vars — expect a clear startup error naming the missing variables.
- [x] **Step 3:** Verification bar; commit: `feat(ws1): server binary and platform demo run over mTLS by default`

### Task 9: Documentation sweep (same PR, never drifts)

**Files:**
- Modify: `SECURITY.md` (transport row: real mTLS + authn/z; KeyExchange row: derived keys, residual documented), `SECURITY_ROADMAP.md` (WS1 checkboxes + deviation note), `RELEASE_READINESS.md` (Gate 3 TLS/auth items), `README.md` (new env vars: `QUANTUMACY_TLS_CERT`, `QUANTUMACY_TLS_KEY`, `QUANTUMACY_TLS_CLIENT_CA`, `QUANTUMACY_INSECURE`), `PROJECT_STATE.md` (test counts, status)

- [x] **Step 1:** Update all five docs; SECURITY.md is the source of truth and must state the residual: "derived keys transit inside mTLS; raw QKD keys never leave the server".
- [x] **Step 2:** Full verification bar one last time; commit: `docs(ws1): security posture updates for mTLS transport`

### Task 10: Finish the branch

- [ ] Push `ws1-transport-security`, open a PR (or hand off for merge), confirm CI green on the PR.

---

## Self-review notes

- Exporter-material deviation is declared up top and lands in SECURITY.md (Task 9) — no silent scope change.
- `peer_certs()` availability requires tonic's `tls` feature (Task 1) — extraction in Task 5 compiles only after Task 1; tasks are ordered for that.
- rcgen's API moved between 0.12/0.13; Task 3 pins the *shape* and tells the implementer to adapt names at compile time rather than trust this plan's recollection.
- Insecure mode keeps the MVP demo path alive so `mvp_flow.rs`'s existing tests stay meaningful; new tests cover the secured path separately.
