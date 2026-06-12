# ADR 0001: Real homomorphic-encryption backend for `he-core`

**Status:** Accepted (pending hands-on spike validation — see "First implementation task")
**Date:** 2026-06-12
**Context:** SECURITY_ROADMAP.md Workstream 3

## Context

`he-core` is a simulation with zero confidentiality (the ciphertext carries its
own mask; decryption ignores the secret key). Its API was deliberately shaped
like CKKS — approximate `f64` vectors, `slots`, `scaling_factor`, polynomial
activations — so the internals could be replaced by a real backend. Workstream
3 requires choosing that backend.

Repository constraints that matter for the choice:

1. **Pure-Rust, self-contained builds** are a load-bearing feature of this
   workspace (vendored `protoc`, no system deps, "constrained env" CI path).
2. The consumers are small dense models (`dl-models`) doing linear layers +
   polynomial activations through `he-inference`.
3. Security claims must be defensible: the backend needs a credible
   maintainer and audit story, not just a working algorithm.

## Options considered (surveyed 2026-06)

| Option | Scheme | Security/maturity | Build story | API fit |
|---|---|---|---|---|
| A. `tfhe-rs` (Zama) | TFHE — exact boolean/integer, programmable bootstrapping | Strong: actively maintained, widely used, GPU/HPU backends, v1.x | Pure Rust ✅ | Poor as-is: requires quantizing models to integers and replacing polynomial activations with programmable bootstrapping |
| B. `openfhe` crate (FFI to OpenFHE C++ 1.5.x) | CKKS (incl. bootstrapping), BGV, BFV | Core library strong (DARPA-lineage, active); the Rust bindings are young (0.1.x) | C++ toolchain + cmake required ❌ | Excellent: CKKS matches the existing API almost 1:1 |
| C. `fhe.rs` (tlepoint) | BFV — exact integers | Respected pure-Rust implementation, small maintainer surface | Pure Rust ✅ | Same quantization burden as A, without A's ecosystem/momentum |
| D. `ckks-engine` | CKKS | Academic project, no audit pedigree; not defensible for security claims | Pure Rust | Good shape, unacceptable assurance |

## Decision

**Option A: `tfhe-rs`.** The pure-Rust, no-system-deps constraint is what makes
this repository reproducible and contributable; breaking it for option B's
C++/cmake toolchain costs more than the quantization work option A demands.
Zama's maintenance, documentation, and adoption give the strongest assurance
story of the pure-Rust options. The model-quantization work (f32 weights →
fixed-point integers, polynomial activations → programmable-bootstrapping
lookup tables) is real but bounded: the dense baselines in `dl-models` are
two-layer networks with ~100–150 parameters.

Consequences accepted:

- `he-core`'s public API changes semantics: the real backend computes on
  quantized integers, so parity tolerances move from ~1e-9 (sim) to the chosen
  quantization step. The `backend-sim` feature remains the default for CI
  speed; `backend-tfhe` is opt-in until benchmarked.
- Real FHE latency is orders of magnitude above the simulation; benchmarks in
  `he-core/benches/` must set expectations in the README.
- CKKS-style fractional semantics (`scaling_factor`, approximate slots) get
  reinterpreted as fixed-point parameters; `HeParameters` keeps its shape with
  documented meaning changes.

Option B is recorded as the fallback if the spike shows quantized inference
accuracy degrading unacceptably on the chestscan/histology baselines: CKKS via
FFI would preserve float semantics at the cost of the build story (likely
behind a non-default feature so the pure-Rust path survives).

## First implementation task (spike gate)

Before committing to the full port, a one-day spike must demonstrate, behind a
`backend-tfhe` feature flag:

1. Encrypt an 8-element i16 fixed-point vector, run one encrypted dense layer
   (matrix-vector + bias) with `tfhe-rs` integer ops, decrypt.
2. A sigmoid-shaped activation via programmable bootstrapping lookup table.
3. Agreement with the plaintext reference within one quantization step on the
   chestscan model's first layer.
4. Wall-clock measurement recorded in the ADR (expectation: seconds per layer
   on CPU — acceptable for research demos, documented honestly).

If (3) fails after reasonable quantization tuning (i8/i16, per-layer scales),
re-open this ADR and execute Option B.
