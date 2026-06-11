# Release Readiness

## Current Recommendation

**Recommended release target**: Research-parity closed alpha / partner pilot

The repository now has 1:1 research parity with [CERN/Quantumacy](https://github.com/CERN/Quantumacy)'s four modules: QKDSimkit, the OpenFL fork, the chestscan dl-models, and the three-party HE use case. It is suitable for MVP-platform validation, internal demos, and design-partner testing.

**Out of scope for this release** and explicitly tracked as post-MVP work: real HE backend (`tfhe-rs` / production CKKS), TLS / rustls overlay, authn/z + raw-key elimination, real TCP/TLS P2P transport, Candle-backed CNNs.

## Release Gates

### Gate 1: Local Engineering Validation
- [x] `cargo check --workspace --all-targets`
- [x] `cargo test --workspace` (67 tests passing)
- [x] `cargo test -p fedlearn-transport`
- [x] `cargo test -p he-core`
- [x] `cargo test -p he-inference`
- [x] `cargo test -p dl-models`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] Constrained-env build path works without a system `protoc` (vendored via `protoc-bin-vendored`).
- [x] `cargo fmt --all --check`
- [x] GitHub Actions CI (fmt, check, clippy, test on ubuntu + macos) — `.github/workflows/ci.yml`

### Gate 2: Runnable Platform Flow
- [x] Transport services exist for registration, model fetch, update submission, status, and key exchange.
- [x] `SubscribeRounds` is backed by a `tokio::sync::broadcast` channel that fires on every successful aggregation.
- [x] Local gRPC demo flow exists in `fedlearn-transport/examples/local_platform_demo.rs` with `--clients/--rounds` flags and a per-round summary.
- [x] QKDSimkit-parity demos: `qkd-network/examples/qkd_p2p_demo.rs` and `qkd-network/examples/qkd_client_server_demo.rs`.
- [x] Chestscan-parity demo: `dl-models/examples/chestscan_federated_demo.rs`.
- [x] Three-party HE demo: `he-inference/examples/three_party_demo.rs`.
- [x] Long-lived app server binary exists at `fedlearn-transport/src/bin/server.rs`.
- [x] Environment-based configuration (`QUANTUMACY_BIND_ADDR`, `QUANTUMACY_PROTOCOL`, `QUANTUMACY_KEY_BITS`, `QUANTUMACY_ROUNDS`, `QUANTUMACY_MIN_CLIENTS`, `QUANTUMACY_LOG`/`RUST_LOG`).

### Gate 3: Security Boundary Clarity
- [x] Document that `he-core` is simulation-grade and not production HE.
- [x] Document that key bootstrap is permissive for MVP.
- [x] Document research-parity vs production scope (see [README.md](README.md) and [CERN_PARITY.md](CERN_PARITY.md)).
- [x] SECURITY.md with per-component security status and reporting policy.
- [x] Crate-level simulation warnings in `he-core` and `qkd-core` rustdoc.
- [x] Post-MVP hardening plan ([SECURITY_ROADMAP.md](SECURITY_ROADMAP.md)).
- [x] Add TLS/rustls around transport (Workstream 1 — mutual TLS required by default).
- [x] Add authentication/authorization for clients (Workstream 1 — certificate-bound sessions).
- [x] Raw QKD keys no longer transit the wire (HKDF per-round derivation; see SECURITY.md for the documented residual).
- [x] Define session lifetime and key rotation policy (Workstream 2 — `KeyPolicy`, zeroized key types, no secret-derived identifiers).
- [x] DP accounting reviewed (Workstream 4 — property-tested calibration, RDP composition).

### Gate 3.5: Open-Source Release Prep
- [x] Apache-2.0 LICENSE file (matches `Cargo.toml` workspace license).
- [x] README reframed as research platform with CERN non-affiliation notice.
- [x] Junk files untracked (`.DS_Store`, internal handoff notes).
- [x] All work committed and pushed.
- [x] CONTRIBUTING.md and `v0.1.0` tag.
- [x] Implied-affiliation metadata scrubbed (authors, repository URL); co-author trailers removed from history.
- [x] Internal pre-release audit (2026-06-11): two network-facing findings fixed in `0e5e4ba`; `cargo audit` clean. See the audit history table in [SECURITY.md](SECURITY.md).
- [ ] Repository made public.
- [ ] GitHub: private vulnerability reporting enabled (SECURITY.md references it).
- [ ] GitHub: Dependabot alerts enabled.
- [ ] First CI run verified green on GitHub Actions.

### Gate 4: Productization
- [x] Structured logging via `tracing-subscriber` with env-filter on the server binary.
- [ ] Add deployment docs and container strategy.
- [ ] Add operational metrics.
- [ ] Add example datasets and repeatable benchmarks.
- [ ] Add app-shell integration docs for frontend/backend teams.

## Next Technical Priorities

1. Replace simulation-grade `he-core` internals with a real CKKS backend (e.g. `tfhe-rs`).
2. Add TLS/rustls overlay, authentication/authorization, and a rotation policy.
3. Replace simulated P2P with real TCP/TLS transport in `qkd-network`.
4. Replace baseline dense models with Candle-based CNNs in `dl-models`.
5. Add deployment docs, container strategy, and operational metrics.
