# Contributing to quantumacy-rust

Thanks for your interest! This is a research platform exploring quantum-safe,
privacy-preserving federated ML in Rust. Contributions of all sizes are
welcome — from typo fixes to taking on a whole workstream from the
[security roadmap](SECURITY_ROADMAP.md).

## Before you start

- **Read [SECURITY.md](SECURITY.md) first.** Several components are
  intentionally simulations. PRs that quietly treat simulated crypto as real
  (in code, docs, or tests) will be asked to reframe.
- For anything bigger than a small fix, open an issue first so we can agree
  on direction before you invest time. The
  [security roadmap](SECURITY_ROADMAP.md) and the "Next Technical Priorities"
  in [RELEASE_READINESS.md](RELEASE_READINESS.md) are the best places to find
  meaningful work.

## Development setup

A stable Rust toolchain is all you need. `protoc` is optional — the build
falls back to a vendored binary (see the "Build environments" section of the
[README](README.md)).

```bash
git clone https://github.com/alex-varga14/quantumacy-rust.git
cd quantumacy-rust
cargo test --workspace
```

## Pull request checklist

CI runs these on ubuntu and macos; please run them locally first:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Additionally:

- New behavior comes with tests. Bug fixes come with a test that fails
  without the fix.
- Public APIs keep rustdoc comments in the style of the existing crates.
- If your change alters a security property (strengthens *or* weakens),
  update the table in [SECURITY.md](SECURITY.md) in the same PR.
- Keep commits focused; reference the issue you're addressing.

## Reporting security issues

Please do **not** open public issues for exploitable findings — see the
reporting instructions in [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), per the terms stated in the README. No CLA
required.
