# quantumacy-rust

Quantumacy-RS is a Rust **research platform** inspired by [CERN's Quantumacy](https://github.com/CERN/Quantumacy) project. It reimplements Quantumacy's four modules — QKD simulation, federated learning, medical-imaging models, and three-party encrypted inference — as a single idiomatic Rust workspace, exploring how quantum-safe, privacy-preserving machine learning on sensitive data could be architected in Rust.

> **Not affiliated with or endorsed by CERN.** This is an independent
> reimplementation; see [CERN_PARITY.md](CERN_PARITY.md) for how it maps to
> the upstream modules.

> ⚠️ **Not a security product.** The homomorphic-encryption layer is a
> simulation with **zero confidentiality** and the QKD layer is a software
> simulator. The gRPC transport does require mutual TLS with
> certificate-bound sessions, but the platform as a whole is research-grade.
> Read [SECURITY.md](SECURITY.md) before doing anything with real data.

## Current State

The repository is a **research-parity MVP**:
- QKD simulation (BB84 / B92 / Six-State) and AES-GCM secure channels are implemented
- Federated learning services and gRPC transport are implemented
- Homomorphic encryption is represented by a simulation-grade API for MVP development
- Medical imaging models are lightweight baselines wired into the FL and HE layers

See [PROJECT_STATE.md](PROJECT_STATE.md) for current implementation status, [RELEASE_READINESS.md](RELEASE_READINESS.md) for release gates, and [SECURITY_ROADMAP.md](SECURITY_ROADMAP.md) for the security hardening plan.

## Local Verification

Requires a stable Rust toolchain (developed and CI-tested on recent stable;
no nightly features). From the repo root:

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
QUANTUMACY_TLS_CERT=server.pem QUANTUMACY_TLS_KEY=server.key \
QUANTUMACY_TLS_CLIENT_CA=ca.pem cargo run -p fedlearn-transport --bin server

# Plaintext mode, local demos only:
QUANTUMACY_INSECURE=1 cargo run -p fedlearn-transport --bin server
```

The binary **requires mutual TLS**: it refuses to start unless
`QUANTUMACY_TLS_CERT`, `QUANTUMACY_TLS_KEY`, and `QUANTUMACY_TLS_CLIENT_CA`
(the CA that client certificates must chain to) are set, or plaintext is
explicitly requested with `QUANTUMACY_INSECURE=1`. It also reads
`QUANTUMACY_BIND_ADDR`, `QUANTUMACY_PROTOCOL`, `QUANTUMACY_KEY_BITS`,
`QUANTUMACY_ROUNDS`, `QUANTUMACY_MIN_CLIENTS`, and a
`QUANTUMACY_LOG`/`RUST_LOG` filter for `tracing-subscriber`. Client identity
is the certificate CN: sessions bind to it at registration and every RPC
re-verifies it.

## Research parity vs production hardening

The MVP targets **research parity** with the upstream CERN/Quantumacy reference: same demos, same protocol coverage, same end-to-end flow, simulation-grade homomorphic encryption. The following are explicitly **out of scope** for this milestone and tracked as post-MVP work:

- Real HE backend (`tfhe-rs` / production CKKS) replacing the `he-core` simulation
- Real TCP/TLS P2P transport replacing the simulated `qkd-network/src/p2p.rs`
- Candle-backed CNN replacing the dense baseline in `dl-models`
- Key rotation / session-lifetime policy (transport mTLS, certificate-bound
  authn/z, and HKDF per-round keys landed with security Workstream 1)

The security side of this work is planned in detail in [SECURITY_ROADMAP.md](SECURITY_ROADMAP.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be licensed as above, without any
additional terms or conditions.
