# Code review findings — 2026-10-05

Review of reef-infer 1.0.0 (commit 78e57b1). The project builds cleanly and
`cargo clippy`, `cargo fmt --check`, and `cargo doc` pass on the library and
binary. The USB protocol work is careful and well-commented. The problems below
are concentrated in the inference loop's bookkeeping, input validation, and the
test setup. Items are ranked by impact within each section.

## 1. Correctness bugs in the runtime

### 1.1 Pending USB transfers grow by one per inference
- Where: `src/runner.rs:278-279` (`load_params`) and `src/runner.rs:380-381`
  (`run_inference`).
- Every call submits a fresh events buffer and a fresh interrupts buffer, and
  every completion inside the loop resubmits. Each successful call therefore
  ends with one more queued transfer than it started with. A 1000-rep bench
  leaves ~1000 queued URBs (512 B + 64 B each) in the kernel.
- On a `Timeout` error the already-submitted transfer stays queued; the next
  call consumes that stale completion first and desynchronises.
- Fix: submit once in `Device::open` (or guard with
  `if self.events.pending() == 0`), and on the error path call `cancel_all()`
  and drain the queue.

### 1.2 Input length is never validated
- Where: `Device::run_inference`, `Device::run_inference_f32`, `py.rs`.
- A wrong-sized buffer is sent as-is; the chip stalls and the user gets
  "TPU did not respond in time" two seconds later with no hint.
- Fix: compare `input_data.len()` against the sum of the execution
  executable's `input_layers[*].size_bytes` and return a dedicated
  `DeviceError::InputSize { expected, got }`.

### 1.3 int8 models are read in the wrong domain by the examples
- Verified with `cargo run --example dump_model -- tests/models/matmul_edgetpu.tflite`:
  the DarwiNN executable reports uint8-domain zero points (zp=128 for a TFLite
  INT8 tensor with zp=0; zp=124 for zp=-4).
- `examples/infer.py:50` casts the raw output to `np.int8`, which is off by 128
  for int8 models. `examples/classify.py` and `webcam_demo.py` read uint8,
  which is correct only for uint8 models.
- Fix: either flip the sign bit (`^ 0x80`) in the runtime when the TFLite
  tensor type is INT8, as libedgetpu does, or expose `input_dtype` /
  `output_dtype` on `Model` so callers can convert.

### 1.4 Multi-output models get outputs in arrival order
- Where: `src/runner.rs:424-430`.
- Tag-3 chunks are pushed as they arrive and concatenated, but
  `dequantize_output` and the Python callers assume `output_layers` order.
- Each layer already has a distinct address in `output_addrs`; place each
  chunk at `ev.device_address - output_base`.
- The loop also finishes on chunk count (`== num_outputs`), so a layer that
  arrives in more than one event ends the loop early. Finish on bytes received
  reaching `output_size` instead.

### 1.5 Fixed round caps instead of a deadline
- Where: `for round in 0..20` (`load_params`) and `for round in 0..30`
  (`run_inference`).
- Every hardware interrupt and every tag-2 parameter request consumes one
  round. An EXECUTION_ONLY executable with inline parameters (e.g. SSD
  post-processing) can exceed 30.
- Fix: replace the counter with an overall `Instant`-based deadline (or
  `tokio::time::timeout` around the whole loop).

### 1.6 Scratch memory is silently ignored
- Where: `patch_chunk` maps `BASE_ADDRESS_SCRATCH => 0`; `scratch_size_bytes`
  is parsed but unused.
- A model that spills will DMA to address 0 and hang or produce garbage.
- Fix: reject models with `scratch_size_bytes > 0` in `load_params` with a
  clear `DeviceError::Unsupported` until scratch is implemented.

### 1.7 Chip init proceeds when polls fail
- Where: `src/chip.rs:178-183` (scalarCoreRunControl), `:192-197` and
  `:237-241` (tileconfig0 ack).
- The loops fall through after 100 attempts without error.
- Fix: return an error. `poll_field` / `poll_register` in
  `src/usb/registers.rs` also spin with no delay between control transfers;
  add a short sleep.

## 2. Robustness and API

- **Blocking sleeps inside async fns.** `std::thread::sleep` at
  `src/usb/mod.rs:197` and `src/usb/dfu.rs:623` stalls a tokio worker. Use
  `tokio::time::sleep`.
- **Max packet size hardcoded to 512** in `bulk_recv`
  (`src/usb/transfer.rs:271`). nusb requires IN requests to be a multiple of
  the endpoint's real max packet size, and the accelerator negotiates 1024 on
  a SuperSpeed (USB 3) port. Use `ep_in.max_packet_size()`.
- **EP 0x01 / 0x81 reopened on every call** in `bulk_send` / `bulk_recv`.
  Open them once in `Device`, as is already done for events and interrupts.
- **Every open failure becomes `NotFound`** (`src/runner.rs:130,144,148,153`).
  A udev permission denial reads as "no Edge TPU device found". Add an
  `Open(nusb::Error)` variant and document the udev rule in the README
  (vendor IDs 1a6e and 18d1). `TransferError::Unknown(0)` is also used as a
  catch-all in several places.
- **`LoadedParams.params_buf` is `pub`.** Its pointer is baked into the cached
  instructions; any reallocation breaks later inferences silently. Make it
  private.
- **No guard against stale `LoadedParams`.** Calling `run_inference` with a
  `LoadedParams` after `load_params` was run for a different model uses the
  wrong SRAM contents. Add a generation counter on `Device` and
  `LoadedParams`.
- **Panics on layer-less executables.** `input_quant`, `output_quant`, and
  `output_len` index `[0]`. Use `.first()` and return `Option`/error.
- **Untrusted model files hit unsafe paths.** `flatbuffers::root_unchecked`
  at `src/model.rs:247`, `from_utf8_unchecked` in `BinaryStr`, and
  `copy_u32_bits` with an unchecked `offset_bit`. At minimum bounds-check
  field offsets against the bitstream length and state the trust assumption
  in the docs.
- **Device::open picks the first known device** even if it is a DFU-mode
  device while an app-mode device is also present; no way to select among
  multiple TPUs. No `Drop` cleanup (cancel transfers / reset chip).
- **Python binding holds the GIL for the whole inference.** Wrap the
  `block_on` calls in `py.detach(...)` so other Python threads (camera
  capture) keep running.
- **Python `run_inference` takes `&[u8]`**, which only accepts `bytes`
  (numpy arrays / `bytearray` are rejected). Accept the buffer protocol.
- **Python error formatting** uses `{e:?}` (Debug) in `load_params` /
  `run_inference` but Display in `open_device`.
- **`Runtime::new()` per device** spawns a full multi-thread runtime; a
  `current_thread` runtime is enough.
- **Unused code:** `check_credits` (which writes 0xffff to the e-fuse data
  register `REG_OMC0_00`), `read_chip_info`, `dump_endpoints`, `read_event`,
  `poll_register`, `Program::extract_params`, `Program::movi_instructions`.

## 3. Tests, tooling, docs

### 3.1 The integration test is not reproducible
- `tests/inference.rs` compares against hard-coded reference outputs whose
  quantization scale (180.39…) comes from an **unseeded** random
  representative dataset in `tests/gen_models.py`.
- The compiled model `tests/models/matmul_edgetpu.tflite` is gitignored
  (both via `tests/models/` and `*.tflite`), so a fresh clone cannot
  regenerate the same model and the test cannot pass.
- The comment's weight matrix (`[1,10]`) differs from the generator
  (`[1,1]`); `weights` and `abs_tol` are unused; tolerance 0.0001 effectively
  demands exact equality.
- Fix: seed numpy, un-ignore and commit the small compiled model, use a
  tolerance of one output quantization step, and delete the dead locals.

### 3.2 No hardware-free tests
- Cheap unit-test candidates: `copy_u32_bits` / `patch_chunk`,
  `parse_dfu_info`, `TpuEvent::from_bytes`, `LoadedParams::dequantize_output`,
  and `load_tflite_model` on the committed model.
- The two tests in `src/parser.rs` are `#[ignore]` and point at
  `assets/` and `../generated_tflite_ops`, neither of which exists.
- The `usb` cargo feature is empty and exists only to gate the hardware test;
  `#[ignore]` + `cargo test -- --ignored` (or an env var) is simpler.

### 3.3 Pre-commit and CI gaps
- Clippy runs without `--all-targets --all-features`, so these slipped
  through in `examples/dump_model.rs`: a duplicate/unreachable `INT64` match
  arm, `cargo fmt --check` failures, and INT8 tensors printed as `?`.
- The `cargo check` hook is redundant with clippy.
- Three unresolved rustdoc links (`[31:24]`, `[int]`, `[run_inference]`).
- No CI workflow.

### 3.4 Housekeeping
- `uv.lock` is untracked: commit it or add it to `.gitignore`.
- `src/parser.rs` is a magic-number disassembler (e.g. `word_index == 27`
  heuristic) that nothing in the runtime uses, yet the CHANGELOG advertises
  it. Move it to `examples/` or mark it experimental / `#[doc(hidden)]`.
- CLI flag parsing in `src/main.rs` is hand-rolled; `lexopt` or `clap` would
  be simpler. The default (no `--output`) print shows only `output[0]`.
- `src/main.rs` picks the default input size from the first input layer of
  *any* executable; use the execution executable.

## Suggested order of work

1. §1.1 pending-transfer growth and error-path drain.
2. §1.2 input length validation (Rust + Python).
3. §1.3 int8/uint8 domain handling.
4. §1.4 output placement by address.
5. §3.1 reproducible integration test + committed model.
6. §1.5, §1.6, §1.7, then the §2 items.
