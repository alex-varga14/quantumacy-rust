# quantumacy-rust
Quantumacy-RS is a Rust rewrite of CERN's Quantumacy research platform - a production-grade, quantum-safe cryptography system combining Quantum Key Distribution (QKD), federated learning, and homomorphic encryption for privacy-preserving machine learning on sensitive medical data.

## Current State

The repository is currently best treated as a **closed-alpha MVP platform**:
- QKD simulation and secure channels are implemented
- Federated learning services and transport are implemented
- Homomorphic encryption is represented by a simulation-grade API for MVP development
- Medical imaging models are lightweight baselines wired into the FL and HE layers

See [PROJECT_STATE.md](PROJECT_STATE.md) for current implementation status and [RELEASE_READINESS.md](RELEASE_READINESS.md) for release gates.

## Local Verification

From the repo root:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### Build environments

The workspace is designed to build out of the box on two environments:

- **Full env (recommended)**: macOS / Linux dev box with `protoc` available
  (e.g. `brew install protobuf` or `apt-get install protobuf-compiler`).
  The build script in `fedlearn-transport/build.rs` will use the system
  `protoc` if `PROTOC` is set or one is on `PATH`.
- **Constrained env**: hosts without `protoc` installed. The transport crate
  declares `protoc-bin-vendored` as a build dependency and the build script
  falls back to the vendored binary automatically. No extra setup required.

If you want to force a specific compiler, set `PROTOC=/path/to/protoc` before
invoking `cargo`.

## Runnable Demo Paths

Each of the four upstream Quantumacy modules has a single-command Rust counterpart. See [CERN_PARITY.md](CERN_PARITY.md) for the side-by-side mapping.

### QKDSimkit parity (qkd-network)

```bash
cargo run -p qkd-network --example qkd_p2p_demo
cargo run -p qkd-network --example qkd_p2p_demo -- --protocol six-state --qubits 8192
cargo run -p qkd-network --example qkd_client_server_demo
```

Both demos print the protocol parameters, derived key id and length, and run an AES-GCM seal/open round-trip on top of the QKD-derived key.

### OpenFL-style federated learning (fedlearn-transport)

```bash
cargo run -p fedlearn-transport --example local_platform_demo
cargo run -p fedlearn-transport --example local_platform_demo -- --clients 4 --rounds 5
```

Spins up the FL + key-exchange gRPC services, registers `--clients N`, runs `--rounds R` rounds of QKD-encrypted FedAvg, and prints a per-round summary.

### Chestscan dl-models parity (dl-models)

```bash
cargo run -p dl-models --example chestscan_federated_demo
cargo run -p dl-models --example chestscan_federated_demo -- --clients 3 --rounds 6
```

Trains the `ChestScanModel` on a synthetic chest-X-ray dataset across multiple federated clients and prints local/global loss + accuracy per round.

### Three-party HE use case (he-inference)

```bash
cargo run -p he-inference --example three_party_demo
```

Wires `ChestScanModel::to_encrypted_model` into the `EncryptedInferenceService`, splitting the flow across **client** (data owner, holds secret key), **storage** (blind orchestrator), and **compute** (model owner, holds evaluation key). Verifies that decrypted predictions match the plaintext reference within `< 1e-9`.

### Persistent server binary

```bash
cargo run -p fedlearn-transport --bin server
QUANTUMACY_BIND_ADDR=0.0.0.0:50051 QUANTUMACY_PROTOCOL=six-state cargo run -p fedlearn-transport --bin server
```

The binary reads `QUANTUMACY_BIND_ADDR`, `QUANTUMACY_PROTOCOL`, `QUANTUMACY_KEY_BITS`, `QUANTUMACY_ROUNDS`, `QUANTUMACY_MIN_CLIENTS`, and a `QUANTUMACY_LOG`/`RUST_LOG` filter for `tracing-subscriber`.

## Research parity vs production hardening

The MVP targets **research parity** with the upstream CERN/Quantumacy reference: same demos, same protocol coverage, same end-to-end flow, simulation-grade homomorphic encryption. The following are explicitly **out of scope** for this milestone and tracked as post-MVP work:

- Real HE backend (`tfhe-rs` / production CKKS) replacing the `he-core` simulation
- TLS / rustls overlay, authn/z, raw-key elimination
- Real TCP/TLS P2P transport replacing the simulated `qkd-network/src/p2p.rs`
- Candle-backed CNN replacing the dense baseline in `dl-models`
