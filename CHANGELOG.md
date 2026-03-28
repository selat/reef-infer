# Changelog

## 1.0.0 — 2026-03-26

Initial public release.

### Features

- Pure-Rust runtime for the Google Coral Edge TPU USB Accelerator
- Userspace USB communication — no kernel driver required
- Automatic firmware upload and chip initialization
- FlatBuffer-based model parsing for Edge TPU–compiled TFLite models
- Async inference via Tokio
- Python bindings via PyO3/maturin
- CLI for single-shot inference and latency benchmarking
- Instruction-level model parser and disassembler
