# Security Policy

## ⚠️ Read this before using anything in this repository

Quantumacy-RS is a **research platform**. It exists to explore the
architecture of quantum-safe, privacy-preserving federated machine learning
in Rust. It is **not** a security product, and several components that look
like cryptography are explicitly simulations:

| Component | Status | What that means |
|---|---|---|
| `he-core` / `he-inference` | **Simulation — zero confidentiality** | The "ciphertext" carries its own masking values and decryption does not depend on the secret key. It models the *workflow* of CKKS-style homomorphic encryption, not the cryptography. Never put real data through it. |
| `qkd-core` / `qkd-network` | **Software simulation** | There is no quantum hardware. Qubit preparation, transmission, and measurement are simulated (as in upstream QKDSimkit). Derived keys are only as secret as the classical RNG and the process memory that produced them. |
| `fedlearn-transport` secure channel | **Real AES-256-GCM, weak bootstrap** | Message protection uses real AES-256-GCM, but the key-exchange service returns raw key bytes over an unauthenticated, un-TLS'd gRPC channel. Anyone on the network path can read the keys. |
| `fedlearn-core` differential privacy | **Real mechanism, unreviewed accounting** | Gradient clipping and Gaussian noise are implemented, but the privacy budget accounting has not been audited. Do not rely on it for formal DP guarantees. |
| Transport (gRPC) | **No TLS, no authn/authz** | All services run in plaintext and accept any client. |

If you need real privacy-preserving ML today, use audited tooling
(e.g. `tfhe-rs`, OpenFL, Opacus) rather than this repository.

The roadmap for closing these gaps is tracked in
[SECURITY_ROADMAP.md](SECURITY_ROADMAP.md).

## Audit history

| Date | Scope | Outcome |
|---|---|---|
| 2026-06-11 | Internal pre-release audit of all network-facing code paths (gRPC handlers, AES-GCM channel, key manager, DP mechanism, FedAvg validation, HE simulation consistency), plus `cargo audit` dependency scan. **Not an external review.** | Two findings, both fixed in `0e5e4ba`: a remote-reachable panic on malformed nonce length in `SecureChannel::decrypt`, and an unclamped client-supplied `key_bits` allowing unbounded allocation in the key-exchange service. Verified sound: AES-GCM nonce handling (fresh 96-bit `OsRng` nonce per message), `SecureKey` zeroization on drop, session-ownership checks on all authenticated RPCs, single-round Gaussian DP bound. Dependency scan: 0 vulnerabilities across 229 crates; one accepted warning-level advisory (RUSTSEC-2026-0097, `rand 0.8.5` — unsound only with a custom logger re-entering the RNG, a pattern this codebase does not use; resolution is the rand 0.9 migration, tracked as post-MVP). |

The "external security review" gate before any beta label is tracked in
[SECURITY_ROADMAP.md](SECURITY_ROADMAP.md) Phase 5.

## Supported versions

There are no supported releases yet. Security fixes land on `main` only.

## Reporting a vulnerability

Even though the platform is research-grade, reports are welcome — especially
anything that contradicts the threat-model claims above (e.g. a path where
simulated components are presented as secure, or a flaw in the AES-GCM
channel or DP implementation).

Please report privately via GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab). Expect an
acknowledgement within a week. Please do not open public issues for
exploitable findings.
