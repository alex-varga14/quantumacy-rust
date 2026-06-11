# Quantumacy-RS: Project State

**Date**: 2026-06-11
**MVP Status**: RESEARCH-PARITY MVP READY — release prep and pre-public audit complete (LICENSE, SECURITY.md, CI, [SECURITY_ROADMAP.md](SECURITY_ROADMAP.md); audit history in [SECURITY.md](SECURITY.md))
**Total Rust LOC**: ~7,900 (incl. demos + server binary)
**Tests Authored**: 67
**Local Test Execution**: ✅ Verified — `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` are both green.
**Workspace Crates**: 7

---

## Overall Status

| Phase | Description | Status | Completion |
|-------|-------------|--------|------------|
| Phase 1 | QKD Foundation | **COMPLETE** | 100% |
| Phase 2 | Federated Learning Core | **COMPLETE** | 100% |
| Phase 3 | QKD + FL Integration | **MVP COMPLETE** | 100% |
| Phase 4 | Homomorphic Encryption | **MVP COMPLETE** | 100% |
| Phase 5 | ML Models + Full Integration | **MVP COMPLETE** | 100% |

The repository now has an end-to-end MVP path:
- QKD-derived secure channels
- Federated learning primitives and transport
- Simulation-grade homomorphic encrypted inference
- Trainable medical-imaging model implementations that bridge FL and HE

---

## Crate Status

### `qkd-core` — COMPLETE
**21 tests authored**

Fully implemented QKD protocol library:
- **BB84 protocol** (`protocols/bb84.rs`): Complete key-generation pipeline from qubit preparation through privacy amplification, with configurable sample fraction, QBER threshold, and deterministic seeding.
- **Six-State protocol** (`protocols/six_state.rs`): Three-basis extension of BB84 with stronger eavesdropping detection and lower sifting rate.
- **B92 protocol** (`protocols/b92.rs`): Two-state protocol with lower throughput and simpler state preparation.
- **Quantum channel simulation** (`channel.rs`): Noise, loss, dark counts, and eavesdropping strategies including intercept-resend and Breidbart.
- **CASCADE error correction** (`error_correction/cascade.rs`): Multi-pass reconciliation with block-size selection from estimated QBER.
- **Privacy amplification** (`privacy_amplification.rs`): SHA-256-based compression sized by secret-key-rate heuristics.
- **Key manager** (`key_manager.rs`): Thread-safe storage with TTL, capacity control, expiration, and zeroization.

### `qkd-network` — COMPLETE
**4 tests authored**

Async networking layer:
- **QKD server** (`server.rs`): Runs blocking protocol work on background threads and manages sessions.
- **P2P abstraction** (`p2p.rs`): Simulated peer-to-peer exchange flow for local coordination.
- **Secure channel** (`secure_channel.rs`): AES-256-GCM message protection with QKD-derived keys.

### `fedlearn-core` — COMPLETE
**19 tests authored**

Federated learning primitives:
- **FedAvg aggregation** (`aggregation.rs`): Uniform and sample-weighted averaging, outlier rejection, momentum, and minimum participation thresholds.
- **Framework-agnostic model types** (`model.rs`): Serializable layered weights plus the `FederatedModel` trait.
- **Differential privacy** (`privacy.rs`): Gradient clipping, Gaussian noise, budget accounting, and local/central DP modes.
- **Round orchestration** (`round.rs`): Single-round execution with DP integration and early stopping controls.

### `fedlearn-transport` — MVP COMPLETE
**7 tests authored** (+ 1 server binary smoke-tested)

Transport and service layer:
- **Secure FL channel** (`secure_channel.rs`): QKD-backed AES-GCM transport for `ModelUpdate` and `ModelWeights`.
- **Generated protobufs** (`build.rs`, `src/lib.rs`): `tonic-build` code generation from `proto/fedlearn.proto`. Build script uses `protoc-bin-vendored` as a fallback so constrained environments build without a system `protoc`.
- **Aggregation gRPC service** (`grpc_service.rs`): Registration, model fetch, update submission, status queries, and a broadcast-backed round subscription that fires on every aggregation transition.
- **Key exchange gRPC service** (`grpc_service.rs`): Session-scoped key issuance and rotation backed by the QKD server.
- **Server binary** (`src/bin/server.rs`): Long-running gRPC server with env-based config (`QUANTUMACY_BIND_ADDR`, `QUANTUMACY_PROTOCOL`, `QUANTUMACY_KEY_BITS`, `QUANTUMACY_ROUNDS`, `QUANTUMACY_MIN_CLIENTS`) and `tracing-subscriber`.
- **Integration flow** (`tests/mvp_flow.rs`): Registration, key minting, encrypted submission, aggregation round-trip, and a `SubscribeRounds` stream test that observes a round transition after aggregation.

### `he-core` — MVP COMPLETE
**7 tests authored**

Simulation-grade homomorphic encryption core:
- **HE scheme config** (`schemes.rs`): CKKS-style simulation parameters plus polynomial-friendly activation definitions.
- **Key generation + encryption** (`encrypt.rs`): Client/public/server key set generation, vector encryption, decryption, and ciphertext refresh.
- **Encrypted arithmetic** (`operations.rs`): Ciphertext addition, plaintext scaling, ciphertext multiplication, polynomial transforms, and encrypted linear layers.
- **MVP intent**: Provides a stable API surface for encrypted ML now, while leaving room to replace internals with `tfhe-rs` later.

### `he-inference` — MVP COMPLETE
**4 tests authored**

Encrypted inference runtime:
- **Encrypted dense model** (`model.rs`): Layered encrypted forward pass over ciphertext vectors with polynomial activations.
- **Inference service** (`server.rs`): Session creation, encrypted input storage, processing, result retrieval, and client-side decryption flow.
- **Three-party MVP**: Supports client encryption, storage/process separation, and encrypted result return through a simple service facade.

### `dl-models` — MVP COMPLETE
**4 tests authored**

Medical-imaging model layer:
- **Shared dense classifier core** (`common.rs`): Lightweight two-layer neural network with backprop training, loss/accuracy evaluation, weight import/export, and encrypted-model conversion.
- **Chest X-ray model** (`chestscan.rs`): Binary abnormality classifier over flattened grayscale images, implementing `FederatedModel`.
- **Histology model** (`histology.rs`): Multiclass tissue classifier implementing `FederatedModel`.
- **HE bridge**: Both models can export directly into `he-inference::EncryptedModel` for encrypted forward passes.

---

## Architecture Decisions Made

1. **FL abstraction first**: `FederatedModel` remains the stable boundary so model implementations can evolve without disturbing aggregation or transport.
2. **MVP HE approach**: Phase 4 now uses a simulation-grade CKKS-style API instead of pulling in `tfhe-rs` immediately. This keeps the workspace self-contained and unblocked in the current environment.
3. **MVP model approach**: Phase 5 uses lightweight dense medical-imaging baselines instead of Candle-backed CNNs. This provides trainable models and encrypted inference integration now without external ML runtime dependencies.
4. **Encrypted activations**: Polynomial approximations are used for HE-compatible activations (`ReluApprox`, `SigmoidApprox`, `TanhApprox`).
5. **Deployment model**: Hybrid remains the intended target: containerized or bare-metal deployment depending on hospital constraints.
6. **QKD interface strategy**: Internal implementations stay trait-based so external standards-aligned adapters can still be added later.

---

## Test Coverage Summary

| Crate | Tests Authored | Status |
|------|----------------|--------|
| `qkd-core` | 21 | ✅ Passing |
| `qkd-network` | 4 | ✅ Passing |
| `fedlearn-core` | 19 | ✅ Passing |
| `fedlearn-transport` | 7 (+ 1 binary) | ✅ Passing |
| `he-core` | 7 | ✅ Passing |
| `he-inference` | 4 | ✅ Passing |
| `dl-models` | 4 | ✅ Passing |

`cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` are both green on macOS / Linux. The constrained-env path (no system `protoc`) is supported via `protoc-bin-vendored` and exercised by the same workspace test command.

Key newly covered scenarios:
- HE encrypt/decrypt roundtrip
- Encrypted ciphertext addition and linear-layer evaluation
- Polynomial activation evaluation on ciphertexts
- Encrypted model forward pass parity with plaintext inference
- Inference-service session lifecycle and three-party flow
- Chest X-ray local training on synthetic binary data
- Histology local training on synthetic multiclass data
- Model-to-encrypted-model conversion parity

---

## Known Issues / Technical Debt

1. **HE is simulation-grade, not production-grade**: `he-core` currently models CKKS-style workflows but does not yet provide true cryptographic homomorphic security. Replacing internals with `tfhe-rs` is the main Phase 4 production follow-up.
2. **Medical models are lightweight baselines**: `dl-models` currently uses small dense networks rather than Candle CNNs. This is enough for the MVP integration path, but not a production medical-imaging stack.
3. **P2P remains simulated**: `qkd-network/src/p2p.rs` still models both peers locally instead of running real network transport.
4. **CASCADE remains simulation-oriented**: Reconciliation still uses Alice's bits as canonical output rather than a full two-party reconciliation transcript.
5. **Privacy amplification is simplified**: SHA-256 counter-style compression stands in for a stronger universal-hash construction.
6. **No TLS overlay yet**: QKD-secured channels do not yet run alongside a rustls/TLS transport layer.
7. **Key bootstrap is still permissive for MVP**: `KeyExchange` returns raw key bytes to bootstrap the secure transport.

Resolved during the research-parity MVP push (2026-04-25):
- `cargo test --workspace` is green; `protoc` is vendored via `protoc-bin-vendored`.
- `SubscribeRounds` is now backed by a `tokio::sync::broadcast` channel that fires on every successful aggregation.
- All four CERN modules have runnable single-command Rust demos (see [CERN_PARITY.md](CERN_PARITY.md)).

---

## MVP Readiness

The repo is now at **MVP-ready** status for development/demo purposes because it supports:
- QKD simulation and secure channel generation
- Federated learning rounds and transport plumbing
- Medical-model implementations that satisfy the FL trait boundary
- Encrypted inference over exported model weights

Production readiness still depends on:
- Real HE backend integration
- Stronger transport/bootstrap hardening
- Real P2P networking
- Executing and stabilizing the full workspace with a Rust toolchain

---

## Build & Run

Workspace-wide verification:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Module parity demos (see [CERN_PARITY.md](CERN_PARITY.md) for the side-by-side mapping):

```bash
cargo run -p qkd-network        --example qkd_p2p_demo
cargo run -p qkd-network        --example qkd_client_server_demo
cargo run -p fedlearn-transport --example local_platform_demo -- --clients 4 --rounds 5
cargo run -p dl-models          --example chestscan_federated_demo
cargo run -p he-inference       --example three_party_demo
cargo run -p fedlearn-transport --bin server
```

---

## File Structure Snapshot

```
quantumacy-rust/
├── Cargo.toml
├── qkd-core/
├── qkd-network/
├── fedlearn-core/
├── fedlearn-transport/
├── he-core/
│   └── src/
│       ├── lib.rs (54)
│       ├── schemes.rs (101)
│       ├── encrypt.rs (231)
│       └── operations.rs (216)
├── he-inference/
│   └── src/
│       ├── lib.rs (11)
│       ├── model.rs (244)
│       └── server.rs (166)
└── dl-models/
    └── src/
        ├── lib.rs (11)
        ├── common.rs (447)
        ├── chestscan.rs (175)
        └── histology.rs (185)
```
