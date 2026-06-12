# Post-MVP Security Hardening Roadmap

> **For agentic workers:** This is a phased roadmap, not a single implementation
> plan. Each workstream below is an independent subsystem; when a workstream is
> scheduled, write a dedicated implementation plan for it (see
> `superpowers:writing-plans`) and execute that plan task-by-task. Checkboxes
> here track workstream-level progress.

**Goal:** Take quantumacy-rust from "honest research simulation" to a platform
whose security claims are real, in risk-ordered increments that each leave the
workspace green (`cargo test --workspace`, `clippy -D warnings`).

**Architecture:** Keep the existing crate boundaries and public APIs stable —
`he-core`'s API was explicitly designed so its simulation internals can be
swapped for a real backend, and `fedlearn-transport` isolates all network
exposure. Harden from the outside in: transport first (cheapest, blocks
everything else), then key lifecycle, then the cryptographic cores.

**Tech stack targets:** `tonic` + `rustls` (TLS/mTLS), `hkdf`/`sha2` (key
derivation), `zeroize` (already a workspace dep), a real HE backend (decision
point: Zama `tfhe-rs` vs CKKS bindings — see Workstream 3), `cargo-audit`/
`cargo-deny` in CI.

---

## Threat model (what we are defending against, post-MVP)

**Assets**
- A1. Patient inference inputs/outputs (the three-party HE flow)
- A2. Model updates / gradients (training-data leakage via inversion attacks)
- A3. Global model weights (IP of the model owner)
- A4. Key material (QKD-derived AES keys, HE key sets)

**Adversaries**
- T1. Network attacker (passive eavesdropper or active MITM on gRPC)
- T2. Unauthorized client (joins federation, poisons or steals the model)
- T3. Honest-but-curious storage/compute party (reads ciphertexts it relays)
- T4. Compromised process memory / disk (key scraping)
- T5. Malicious dependency or build pipeline (supply chain)

**Current exposure (MVP, all documented in [SECURITY.md](SECURITY.md))**
- T1 wins trivially: no TLS, and `KeyExchange` returns raw key bytes in plaintext.
- T2 wins trivially: registration accepts any client.
- T3 wins trivially against the HE flow: `he-core` ciphertexts carry their own
  masks (`he-core/src/encrypt.rs`, `CiphertextVector.mask`).
- T4 partially mitigated: `qkd-core::KeyManager` zeroizes, but HE keys and
  channel keys are plain structs.
- T5 unmitigated: no `cargo audit`/`cargo deny` gate.

Out of scope permanently (documented, not planned): real quantum hardware.
QKD remains a simulator; its value is protocol research, not key secrecy.

---

## Workstream 1 — Transport security: TLS, mTLS, authn/authz

**Threats closed:** T1, T2. **Priority: first** — every other guarantee is
meaningless while keys cross the wire in plaintext.

**Files:** `fedlearn-transport/src/grpc_service.rs`,
`fedlearn-transport/src/bin/server.rs`, `fedlearn-transport/Cargo.toml`
(enable `tonic/tls`), new `fedlearn-transport/src/auth.rs`,
`fedlearn-transport/tests/`.

- [x] Enable `rustls` on the tonic server and client builders; env-configured
      cert/key paths (`QUANTUMACY_TLS_CERT`, `QUANTUMACY_TLS_KEY`,
      `QUANTUMACY_TLS_CLIENT_CA`); plaintext mode only behind an explicit
      `QUANTUMACY_INSECURE=1` for local demos.
- [x] mTLS client identity: require client certificates signed by the
      federation CA; map cert subject → client id at registration.
- [x] Authorization checks: sessions bind to the certificate CN and every
      session-scoped RPC (model fetch, submission, status, subscription,
      key exchange) re-verifies it; cross-session access is denied even
      with a stolen session id.
- [x] **Raw QKD keys no longer transit.** `KeyExchange` returns
      HKDF-SHA256(qkd_key, salt = session id, info = round) per-round keys;
      the server re-derives on decrypt. *Deviation from the original plan:*
      tonic does not expose TLS exporter material (RFC 5705), so the
      derived key — never the raw QKD key — still travels inside the mTLS
      channel. Full elimination moves to Workstream 5 (distributed QKD
      endpoints).
- [x] Integration tests: plaintext client rejected; wrong-CA cert rejected;
      CN/client_id mismatch rejected; session hijack with a stolen session
      id rejected; full encrypted round over mTLS passes
      (`fedlearn-transport/tests/tls_handshake.rs`, `tests/mvp_flow.rs`).

**Exit criteria (met 2026-06-11, branch `ws1-transport-security`):** wire
traffic carries only TLS-protected frames; raw QKD material never leaves the
server process; 12 new tests cover the rejection and round-trip paths.

## Workstream 2 — Key lifecycle: rotation, TTL, zeroization

**Threats closed:** T4 (and residual T1 exposure window).

**Files:** `qkd-core/src/key_manager.rs` (TTL exists — wire it to policy),
`fedlearn-transport/src/secure_channel.rs`, `he-core/src/encrypt.rs`.

- [x] Session policy defined and enforced in `SecureFLChannel` via
      `KeyPolicy` (defaults: 1000 messages or 1 hour, whichever first);
      encrypt paths return `TransportError::KeyExpired` directing rotation
      via the `rotate_key` RPC; decryption stays unlimited so in-flight
      messages drain.
- [x] `Zeroize + ZeroizeOnDrop` on `ClientKey`, `PublicKey`, `ServerKey`
      (he-core; `SecureKey` in qkd-core already had it). `Clone` retained —
      each clone zeroizes independently on its own drop (API depends on it).
- [x] **Secret-derived identifiers fixed:** `he-core` key ids are random
      UUIDs and `mask_seed` is independent randomness; a test asserts no
      seed material appears in public identifiers.
- [x] Tests: encrypt refused after N messages / max age; TTL expiry covered
      for both `get()` and `consume()` (neither was covered before);
      compile-time `ZeroizeOnDrop` assertions on all key types.

**Exit criteria (met 2026-06-11, branch `ws2-key-lifecycle`):** rotation
policy enforced and documented in SECURITY.md; no key type without
zeroization; no public identifier derived from a secret.

## Workstream 3 — Real homomorphic encryption backend

**Threats closed:** T3 — the headline gap. Largest workstream; do not start
before Workstream 1 lands (no point doing real HE over plaintext transport).

**Decision to make first** (spike, ~1 week, document in an ADR):
- **Option A: `tfhe-rs` (Zama).** Pure Rust, actively maintained — but TFHE is
  exact integer/boolean FHE, so the dense models must be quantized and
  polynomial activations replaced with programmable bootstrapping. Best
  long-term fit for the "pure Rust workspace" constraint.
- **Option B: CKKS via FFI** (OpenFHE/SEAL bindings). Matches the existing
  CKKS-style API (approximate `f64` arithmetic, `slots`, `scaling_factor`)
  almost 1:1, so `he-inference` and `dl-models` barely change — at the cost
  of a C++ build dependency, which breaks the constrained-env story.

**Files:** `he-core/src/{encrypt,operations,schemes}.rs` (internals only —
keep the public API), `he-core/Cargo.toml` (feature flags
`backend-sim` / `backend-real`, sim remains default for CI speed),
`he-inference/src/model.rs` (tolerance constants), `dl-models/src/common.rs`
(quantized export path if Option A).

- [x] ADR committed as `docs/adr/0001-he-backend.md` (decision: `tfhe-rs`,
      openfhe-FFI fallback). Hands-on spike gate defined there; spike
      implementation is the next WS3 step.
- [ ] Implement backend behind feature flag; `CiphertextVector` loses the
      `mask` field in the real backend (serialized ciphertext only).
- [ ] Port the existing parity tests: encrypted vs plaintext inference
      agreement within backend-appropriate tolerance (CKKS: ~1e-3, not the
      sim's 1e-9; TFHE: exact at chosen quantization).
- [ ] Negative tests: decryption with the wrong client key fails; a
      storage-party view of the ciphertext yields no information (statistical
      sanity test: ciphertext bytes pass a basic randomness check).
- [ ] Benchmark in `he-core/benches/` and record results in README (set
      expectations: real HE is 10³–10⁶× slower than the sim).
- [ ] Update SECURITY.md: move `he-core` from "simulation" to "real backend,
      unaudited" — keep the warning until an external review happens.

**Exit criteria:** `three_party_demo --features backend-real` runs with
genuine ciphertexts; sim backend clearly labeled and non-default in release
docs.

## Workstream 4 — Differential privacy audit

**Threats closed:** A2 leakage through aggregated updates (complements, not
replaces, transport security).

**Files:** `fedlearn-core/src/privacy.rs`, `fedlearn-core/src/round.rs`.

- [x] Gaussian mechanism calibration verified with property tests over
      σ ∈ [0.5, 10], δ ∈ [1e-7, 1e-3] (closed-form equality, monotonicity,
      σ ≤ 0 refusal); `compute_round_epsilon` simplified to state the
      actual formula.
- [x] RDP accountant (`fedlearn-core/src/rdp.rs`, Mironov 2017:
      ε_RDP(α) = k·α/2σ² over an 18-order grid, converted via
      min_α[ε_RDP(α) + ln(1/δ)/(α−1)]) replaces naive ε-summation; budget
      enforcement is fail-closed (prospective check before noise release).
- [x] RNG hygiene verified: production noise uses `ChaCha20Rng::from_entropy`;
      `with_seed` is documented test-only and grep-verified unused outside
      `#[cfg(test)]`.
- [x] Delivered privacy documented: at σ = 1.0, δ = 1e-5 — single round
      ε ≈ 5.30; 100 rounds ε ≈ 98.0 (vs ≈ 484 naive, ~4.9× tighter).

**Exit criteria (met 2026-06-11, branch `ws4-dp-audit`):** accountant
verified against analytically derived values; SECURITY.md DP row upgraded
to "reviewed".

## Workstream 5 — Real P2P transport and QKD classical-channel authentication

**Threats closed:** residual T1 on the QKD path; honesty of the QKD
simulation itself.

**Files:** `qkd-network/src/p2p.rs` (rewrite), `qkd-network/src/client.rs`,
`qkd-core/src/error_correction/cascade.rs`,
`qkd-core/src/privacy_amplification.rs`.

- [x] Real TCP P2P: `run_alice`/`run_bob` endpoints over `tokio` TCP; the
      qubit channel remains simulated behind an explicitly-labeled
      `SimulatedQuantum` message (the simulation boundary is documented in
      `qkd-network/src/p2p.rs`); the old in-process peer stays as a test
      fixture. (Pre-shared HMAC keys chosen over rustls for the classical
      channel — matches the QKD-literature authentication model.)
- [x] Authenticated classical channel: length-prefixed frames with
      HMAC-SHA256 over per-direction sequence number + payload
      (`qkd-network/src/classical_channel.rs`); tampering or wrong keys
      abort the protocol; constant-time verification.
- [x] CASCADE: true two-party message-driven reconciliation
      (`CascadeMessage` state machines, BINARY search, multi-pass with
      cascade re-checks, SHA-256 convergence verification); every parity
      response counted as one leaked bit.
- [x] Privacy amplification: Toeplitz universal hash over GF(2), output
      sized as floor(n·rate) − leaked_bits − safety margin; all three
      protocol pipelines use it. (Conservative: leakage is subtracted on
      top of the rate heuristic's existing EC term — documented.)
- [x] Two-process integration test: the test binary re-spawns itself as two
      OS processes connected only by localhost TCP; identical key id/QBER/
      leakage/material asserted across processes; QBER abort and tamper
      abort covered. Sample run: 4096 qubits at 2% noise → QBER 1.6%,
      157 parity bits leaked, 1056-bit final key, identical on both ends.

**Exit criteria (met 2026-06-12, branch `ws5-real-p2p`):** two real OS
processes derive identical keys over an authenticated channel; leakage
accounting measured and documented.

Follow-up (new): per-session nonce in the channel `Hello` to rule out
cross-session replay (sequence numbers are currently per-connection only).

## Workstream 6 — Supply chain and operational security (start now, ongoing)

**Threats closed:** T5. Cheap; the first two items should land with the next
CI change rather than waiting their turn.

**Files:** `.github/workflows/ci.yml`, new `deny.toml`, new
`.github/dependabot.yml`.

- [x] `cargo audit` job in CI (RUSTSEC advisories fail the build).
- [x] `cargo deny` for license + duplicate-version policy (`deny.toml`;
      one triaged ignore: rustls-pemfile unmaintained, RUSTSEC-2025-0134).
- [x] Dependabot for `Cargo.toml` (weekly, grouped minor/patch) and GitHub
      Actions versions. (Dependabot alerts + security updates enabled in
      repo settings 2026-06-11; first alert batch — rustls-webpki — was
      fixed by the tonic 0.12 upgrade.)
- [ ] Fuzz the deserialization surfaces with `cargo-fuzz`: protobuf-decoded
      messages into `grpc_service.rs`, and `serde_json` paths in
      `fedlearn-core/src/model.rs`.
- [ ] Pin GitHub Actions by SHA; sign release tags.

**Exit criteria:** CI fails on known-vulnerable deps; fuzz targets run in a
weekly scheduled job with zero outstanding crashes.

---

## Sequencing and review gates

| Phase | Workstreams | Gate to next phase |
|---|---|---|
| 0 (done) | Honest disclosure: SECURITY.md, crate warnings, README reframe | — |
| 1 | WS1 (transport) + WS6 first two items | Packet capture clean; CI audit gate green |
| 2 | WS2 (keys) + WS4 (DP) — independent, parallelizable | Rotation policy enforced; DP accountant verified |
| 3 | WS3 (real HE) | Real-backend three-party demo green |
| 4 | WS5 (P2P/QKD) | Two-process QKD key agreement |
| 5 | External security review of WS1–WS3 before any "beta" label | Findings triaged |

Rules of the road:
- Every task lands with tests, on a branch, with `cargo test --workspace` and
  `clippy -D warnings` green — same bar as the MVP.
- SECURITY.md is updated in the *same commit* as any change that alters a
  security claim. The table there is the single source of truth users read.
- Nothing in this repo claims "production-grade" until Phase 5's external
  review completes.
