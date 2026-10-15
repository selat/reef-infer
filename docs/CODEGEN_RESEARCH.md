# Research notes: generating Edge TPU programs ourselves

> The consolidated reference is [`EDGE_TPU.md`](EDGE_TPU.md). This file is
> the dated research log behind it.

Date: 2026-10-05. Goal: make reef-infer end-to-end, i.e. produce DarwiNN
instruction bitstreams for the Coral Edge TPU without `edgetpu_compiler`.

## Verdict

Realistic, and already demonstrated by others. Two public projects have done
most of the reverse engineering; one of them generates programs from scratch
and runs a whole transformer on the USB accelerator. We would build on their
findings rather than start from zero.

## Prior art

| Project | License | What it provides |
|---|---|---|
| [tinycoral](https://github.com/FrederikRogalski/tinycoral) | MIT | Decoded ISA with field layouts, encoders/decoders and builders for every instruction it emits; a code generator; a tinygrad backend; USB driver. Generated programs are byte-identical to `edgetpu_compiler` output where an equivalent exists. Emits FULLY_CONNECTED, CONV_2D, SOFTMAX, L2_NORMALIZATION, SUM/MEAN; chains of VALID stride-1 convs (odd kernels up to 5x5), 2x2 max pools, final linear layer. Runs all 6 layers of TinyStories-15M on-chip (1,815 tok/s at batch 256; 176.7 tok/s single token). Docs: `docs/ARCHITECTURE.md`, `docs/isa/*.md`, `NOTES.md`, `STORY.md`. |
| [libredgetpu](https://github.com/ricardodeazambuja/libredgetpu) | Apache 2.0 | Pure-Python driver. Does **not** generate instructions; patches weights into `edgetpu_compiler`-produced templates. `docs/HARDWARE_ANALYSIS.md` (~1400 lines) covers registers, DarwiNN executable format, SRAM limits (largest cached square matrix ~2896x2896), DMA/USB protocol, patent cross-references. |
| [edgetpuxray](https://github.com/geohot/edgetpuxray) | (check repo) | Earlier decompiler / register-map work both projects built on. |
| [libedgetpu](https://github.com/google-coral/libedgetpu) | Apache 2.0 | `executable.fbs` (what we already parse), CSR offsets, register bitfield headers, driver sequencing. Our `chip.rs` run-control registers (infeed, outfeed, avDataPop, parameterPop, ring bus, mesh bus) are the DMA-unit names from the patents. |

Google patents describing the architecture:

- [US9836691B1](https://patents.google.com/patent/US9836691B1/de) and
  [US20180197068A1](https://patents.google.com/patent/US20180197068A1/en),
  "Neural network instruction set architecture": TensorOp and DMAOp
  instructions, narrow/wide memory, ring bus, mesh bus, tile instruction
  buffers (mesh inbound/outbound, narrow-wide DMA, wide-narrow DMA, ring-bus
  DMA).
- [US20200005128A1](https://patents.google.com/patent/US20200005128A1/en),
  "Neural network accelerator with parameters resident on chip": the
  mechanism behind the PARAMETER_CACHING executable we already drive.
- libredgetpu also cites US20190050717A1, GB2558980A, US20210373895A1.

## Architecture facts (from tinycoral docs unless noted)

- **Not RISC-V.** The core is a custom design, internally "DarwiNN". The only
  conventional CPU on the chip is an **8051** microcontroller running the USB
  firmware we upload over DFU (`apex_latest_single_ep.bin`); `chip.rs` sets
  its clock bit. The RISC-V rumour most likely comes from libredgetpu
  advertising RISC-V *host* support, or from Sophgo SoCs that pair a TPU with
  RISC-V cores.
- **Instruction words:** 128-bit; variable-length instructions of 1-17 words;
  opcode in bits 6-11 of the first word. Matches the 16-byte words
  `src/parser.rs` already decodes.
- **Scalar core:** 32 registers, 8 predicates; ALU ops (ADD ... MOV with an
  immediate form); runs the program and issues DMAs.
- **Tiles:** 4x4 grid (16 PEs). Per tile: 192 KiB narrow memory
  (activations), 512 KiB wide memory (weights) -> 8 MiB weight storage total.
  Peak 4 TOPS int8 at ~500 MHz (libredgetpu: 64x64 systolic array at
  480 MHz).
- **Sync:** 19 sync counters per tile (AVDATA, PARAMETERS, MESH_IN/OUT,
  RING_*); a sync instruction waits until a counter reaches a value; every
  tile-dispatched instruction carries a sequence number.
- **Interconnect / DMA:** ring bus (scalar -> tiles), mesh bus (N/E/S/W
  neighbours); DMA units infeed, outfeed, avDataPop, parameterPop.
- **Requantization pipeline:** uint8 x uint8 -> int32, add bias, scale by a
  float32 multiplier, round half-to-even (found empirically), clamp, add zero
  point.
- **Still unknown publicly:** RTL, exact systolic organisation inside a tile,
  complete DMA descriptor format.
- **No signing or checksum** on bitstreams: any program the host sends runs.
  Bad programs stall the chip; recovery is re-init or replug.

## How it was reverse engineered

Differential analysis of `edgetpu_compiler` output: build many small models
varying one parameter at a time, diff the bitstreams, infer fields; use
byte-identical reproduction as the development oracle and the hardware as the
final oracle. Disassembling the compiler binary was not needed by the prior
projects; if considered, check the Coral software licence terms first.

## Proposed plan for reef-infer

1. **Decoder.** Replace `src/parser.rs` (magic-number heuristics) with a full
   decoder ported from tinycoral's field layouts (MIT, licence-compatible).
   Validate by round-tripping every compiler-produced bitstream we have
   (`tests/models/*_edgetpu.tflite`) and asserting decode -> encode is
   byte-identical.
   *Done 2026-10-08:* `src/isa/` decodes and encodes all 23 opcodes seen in
   86 bitstreams from 53 compiled models (17,942 instructions). Every
   bitstream round-trips byte for byte, and the decoded fields match
   tinycoral's Python decoder on all 17,901 instructions it covers. Run
   `cargo run --example isa_check -- <models>` for coverage and unexplained
   bits. Largest unexplained region: TensorOp bits 1878-1922 (set in ~40% of
   ops, 21 distinct values).
2. **Encoder + builders.** Emit single-op programs (FULLY_CONNECTED first,
   then CONV_2D) and verify byte-identical output against `edgetpu_compiler`
   on generated models. Keep the compiler as a test oracle in CI-less local
   tests, cached like tinycoral's `.compile/`.
3. **Scheduler / allocator (the hard part).** Tile memory allocation, DMA
   chunking, sync-counter sequencing for multi-layer graphs. We do not need
   to match the compiler's choices, only produce valid programs.
4. **Front end.** Lower TFLite ops (or a small internal graph IR) to the
   builders. Produce the same `Package`/`Executable` FlatBuffers so the
   existing runner needs no changes.

## Effort estimate

- Dense + conv + pool chains with tinycoral as reference: weeks to a few
  months.
- MobileNet-class coverage (depthwise conv, residual adds, quantization
  corner cases): many months.

## Open questions

- Licence of edgetpuxray (verify before borrowing code).

Answered 2026-10-08 from tinycoral's README and `NOTES.md`:

- **Parameter caching:** tinycoral emits stand-alone programs and keeps every
  weight matrix resident by placing weights itself. It does not generate
  PARAMETER_CACHING executables. Wide buffers are clobbered on every call, so
  it tracks a per-tile address floor across programs.
- **Multi-tile:** its placer shifts compute tiles to `[s, s+T)` while the
  input gather stays on tile 0, a layout `edgetpu_compiler` never emits. So
  multi-tile placement works for FC; conv is less clear.
- **Google at runtime:** none. `edgetpu_compiler` is only an offline
  byte-for-byte oracle during development.
- **Known generator limits:** 1x1 conv only up to 256 positions x 3840 bytes;
  output groups under 64 read wrong weights at >= 256 KiB (pad to 64).
- **Status:** Python, MIT, a handful of commits, single developer. Treat it as
  a reference, not a dependency.

## Removing Google entirely

| Piece | Status | Removable? |
|---|---|---|
| `edgetpu_compiler` | Not used at runtime | Yes, this is the codegen project. Golden bitstreams can be captured once and vendored as test fixtures. |
| TFLite / TensorFlow | Model input format and dev dependency for test models | Yes, with a PyTorch (`torch.export`) front end. |
| DarwiNN `executable.fbs` | Container the runner parses | Optional. It is only a schema (Apache 2.0); the runner could take an internal program struct instead. |
| `apex_latest_single_ep.bin` | 8051 USB firmware uploaded over DFU | Not realistically. No source exists; replacing it means writing 8051 firmware for undocumented hardware. It is Apache 2.0 and redistributable, so vendoring it is a licence formality, not a runtime dependency. |

## Instruction set and supported operations (from tinycoral `docs/isa/*.md`)

Two-level ISA: a scalar core runs the program and moves data; 16 tiles do the
arithmetic. Opcode is bits 6-11 of the first 128-bit word. All figures are
verified by byte-exact round-trip against compiler output in tinycoral.

### Scalar core

| Opcode | Mnemonic | Words | Role |
|---|---|---|---|
| 0x3e | start | 1 | Program header |
| 0x21 | halt | 1 | Terminator (tail is always `halt; nop x4; end`) |
| 0x3f | end | 1 | Final marker |
| 0x1a | sync | 1 | Tile fence/barrier; 12 known variants |
| 0x24 | scsync | 2 | Scalar fence; 10 known variants |
| 0x25 | pop | 2 | Host DMA -> staging buffer (activation stream, 8 B units, 128 KiB; parameter stream, 64 B units, 128 KiB) |
| 0x26 | infeed | 4 | Staging -> tiles over ring bus, destination tile bitmap |
| 0x27 | outfeed | 2 | Ring -> host (4-word descriptor: addr lo/hi, length, tag 1=input 2=param 3=output 4-7=interrupt) or ring -> scalar memory |
| 0x20 | ALU | 1 | ADD ... MOV (immediate form); 32 registers, 8 predicates |

Sync: 19 tile counters (AVDATA, PARAMETERS, PARTIAL_SUMS, MESH_{N,E,S,W}_{IN,OUT},
WIDE_TO_NARROW, WIDE_TO_SCALING, NARROW_TO_WIDE, RING_READ_{A,B}, RING_WRITE,
RING_PRODUCER_{A,B}); 9 scalar counters. Every tile-dispatched instruction
carries a global sequence number (bits 46-59 / [28,43) depending on class).

### Tile data movement (each carries its own loop nest: (increment, count) per level)

| Opcode | Mnemonic | Words | Role |
|---|---|---|---|
| 0x13 | narrowToWide | 7 | Narrow mem (4 B units) -> wide mem / ring FIFO (64 B units); 6-level narrow + 4-level wide TTU |
| 0x14 | wideToNarrow | 7 | Wide mem / ring FIFO -> narrow mem; modes: ring, sub-row select, bias/scaling |
| 0x10 | ringProducer | 4 | Multicast to 17-bit bitmap (bit t = tile t, bit 16 = scalar core/outfeed) |
| 0x11 / 0x12 | ringBusConsumer0/1 | 4 | Receive ring data |
| 0x15 / 0x16 / 0x17 / 0x18 | mesh S / W / N / E | 6 | To 4-neighbour tile in row-major 4x4 grid (t+4, t-1, t-4, t+1) |
| 0x19 | LUT load | 13 | 40 f32 coefficients + 7 breakpoints for LOGISTIC (8-segment quartic) |

### Tile compute

- **0x01 TensorOp**, 17 words (2176 bits): header; 8 main loop counters;
  four 8-deep tensor traversal units (input read, output write, parameter
  read, partial-sum read; each level = signed inc, count-1, 2-bit mode);
  datapath block (flags, sync slots, zero points w_zp/in_zp/out_zp, reduce
  mask, channel count, f32 multiplier, f32 clamp min/max).
- Narrow memory: 4-byte-aligned base, increments in 32-bit words. Wide
  memory: base in 64 B units, increments in 256 B rows, one row = 256 MACs
  per tile per step.
- Operation selected by `dp_mode` + `reduce_mask`:

| Operation | dp_mode | reduce_mask |
|---|---|---|
| Fully connected (64 outputs per pass, bias row in weights) | 3 | 0b101 |
| RELU / RELU6 / RELU_N1_TO_1 / QUANTIZE / ABS (requantize) | 1 | 0 |
| Copy / reformat pass before conv | 4 | 0 |
| Conv 3x3, split-K 1x1 (weight FIFO, K-chunked) | 4 | 0b1011 |
| Depthwise conv (per-channel weights) | 3 | 0b11 |
| Average pooling | 4 | 0b11 |
| ADD / SUB / MAX / MIN / MUL | 5 | 0b11 |

- **0x02 MAX_POOL_2D**: separate opcode, pooling window laid out in wide memory.
- ADD uses a 16-bit mode with rational weight approximation; MUL is a two-step
  multiply then pack with operand B streamed through a 1-4 row FIFO.
- Output stage (fixed): uint8 x uint8 -> int32, + bias, x f32 multiplier
  (`in_scale * w_scale / out_scale` computed as `F(F(x) * F(1/s_out))` in
  float32), round half-to-even, clamp (`255 - out_zp` / `-out_zp`, RELU sets
  min to 0), + out_zp. Float64 reproduces the compiler's constants only ~60%
  of the time.

### What this means for an op set

Everything in the public Edge TPU op list reduces to: dense, conv, depthwise,
pooling, elementwise arithmetic, a few activations, and reshapes expressed as
DMA loop nests. No general scalar math, no float path, tiles talk only to
neighbours or over the ring. Softmax, L2 norm and attention in tinycoral are
compositions of these with the host doing the rest.

Remaining public unknowns: per-dimension TTU mode bits, exact sync counter
semantics (set vs wait, `sync_val` values), wide address space range, some
header bits (`sync.b43`, `halt.code`, `end.length`).

## Compiler 16.0 corpus findings (2026-10-08)

Corpus: `tools/isa_corpus/gen.py` builds 158 single-op models (FC, conv,
depthwise, elementwise, pooling, activations, misc), each varying one
parameter around a base config, and compiles them with edgetpu_compiler
16.0.384591198 (156 compile; reshape and nearest resize fail). Analysis:
`tools/isa_corpus/analyze.py` over an `isa_check --dump`. Together with the
54 earlier models: 359 bitstreams from 210 models, all round-tripping.

**Compiler 16.0 emits a different FC op than 14.1.** 16.0 FC ops use
`cfg0 = 0x65`, `dp_mode = 1`; tinycoral's 14.1 table uses `cfg0 = 0xA5`,
`dp_mode = 3`. This is why tinycoral's FC generator does not reproduce our
16.0-compiled models. Both forms come from Google's compiler, so both should
be valid on hardware, but our models cannot serve as byte-exact references
for a 14.1-style generator.

**Decoded:**

| Field | Bits | Rule | Evidence |
|---|---|---|---|
| `wideToNarrow.tail_words` (signed) | 526-541 | `n_lim0 + 1 - (head_words_m1 + 1)` with a head, else 0. tinycoral's `head_tail16` is bits 4-7 of it | 1375/1375, asserted in `tests/isa_roundtrip.rs` |
| `op.row_a`, `op.row_m`, `op.row_c` (provisional) | 1886-1896, 1898-1899, 1910-1920 | 16.0 only; nonzero exactly in FC/conv/dwconv ops. `row_c = row_a` if `row_m = 1`, else `row_a - row_m` | 633/634 |

`row_a` behaves like a wide-memory row bound (a tile has 2048 rows of 256 B):
2047 for small convs, minus the input width for W = 48/64, minus `4*k^2` for
conv k >= 6 and `2*k^2` for depthwise k >= 5; FC gives 1-3. It does not equal
the op's `par_base` or `psum_base` row. Meaning unconfirmed; needs a hardware
experiment (perturb it and observe).

Second pass (same day), with program-level rules checked in
`tests/isa_roundtrip.rs::check_program`:

| Field | Bits | Rule | Evidence |
|---|---|---|---|
| `outfeed.loop_en`, `loop_row`, `loop_cfg` | 214, 188-198, 178-187 | Loopback outfeed: rows go to an on-chip ring of rows 1024..2047 instead of the host. First use is row 2046; each one advances `loop_row` by `d1_limit + 1`, wrapping to 1024. `loop_cfg` is 17 in 90/92 | ring rule 64/66 |
| `infeed.loop_en` (+ constants `loop_k168` = 1, `loop_k308` = 3, `loop_k315` = 3) | 13, 168, 308-309, 315-316 | The infeed that reads a loopback outfeed's rows back into all tiles: issued before it, same d0/d1/d2 limits | 92/92 pairs |
| `pop.buf_off` | 13-23 | Staging-buffer offset a parameter pop writes to; the next parameter infeed's `buf_off` equals it. Usually advances by `count_m1 + 1`, but the wrap is not understood (two misses differ by exactly 1024) | 39/39 pairs; advance 34/37 |
| `op.w_zp_hi` | 1862-1869 | In SUB ops, `w_zp \| w_zp_hi << 8` is a 16-bit operand weight: the numerator of `Fraction(s_b/s_a).limit_denominator(32767)` | 4/4 models |
| `op.row_cfg` | 2094-2096 | 1 exactly when `row_a`/`row_m`/`row_c` are used | 1803/1805 |

The loopback path matters for code generation. It is how 16.0 moves data
between tiles in depthwise and MobileNet programs: out of the tiles onto the
ring, then back in to every tile.

**Correlations, not yet decoded:**

- `narrowToWide` bits 589-756: four extra sync-like records at 589 + 42k.
  Each has a 5-bit id at +4 (20 or 4; 24 in one MobileNet op), a 2-bit level
  at +9, a 19-bit value at +13 (2048 or 4095) and bit +40 set. Seen in binary
  elementwise and MobileNet programs, with near-constant values.
- `mesh` bits 249-254 / 454-459: the same offset (249) in each half as
  `ringConsumer.gstride`. Values 34/50 for 3-level moves and 5/9 for 2-level
  moves; maybe 2 bits per level. No relation to the instruction's own fields.
- `op` bit 1525: set in 152 of the 157 copy-class ops (`dp_mode` 0, `cfg0` 0).
- `infeed` bits 353-368 and 391: SSDLite streamed-parameter infeeds only. The
  low 6 bits are always 0x25; the upper bits grow through the program.

- `wideToNarrow` bit 543 equals `skip_en0` in 1374/1375 instances.
- `ringConsumer` bits 286-287 and 319 accompany `aux_en0` / `aux_en1` (the
  caching relocation records) in the corpus, but only ~70-90% of the time in
  MobileNet/SSD programs.

**Still open:** the correlations above, `op` bits 1946-1948 (activation
ops, 5 values), `wideToNarrow` bits 643-645 and 490-493, and single-instance
bits in MobileNet programs. The rest likely needs targeted sweeps (multi-input
elementwise, stride-2 depthwise, streamed weights) or perturbation on hardware.

## First generated programs on hardware (2026-10-10)

Coral USB Accelerator, compiler 16.0 corpus as reference. Code:
`src/codegen/` (assembler, `elementwise`, `fc`), checks in
`examples/codegen_hw.rs`, perturbations in `examples/isa_probe.rs`, tests in
`tests/codegen.rs` (offline, byte equality when the corpus is present) and
`tests/codegen_hw.rs` (on device).

**Programs generated from Rust run correctly.** Every instruction is
encoded from named fields by `codegen::Asm`; no compiler bytes are copied.

| Program | vs compiler | On TPU vs host reference |
|---|---|---|
| Element-wise requant, 1x16x16x32 (RELU layout) | Byte-identical bitstream, relocations and output layout | 0 mismatches over 8192 values, 8 parameter sets including ones the compiler never emits (negative multiplier, nonzero `out_zp`, fractional clamps) |
| FC, PARAMETER_CACHING + EXECUTION_ONLY | Byte-identical bitstreams **and parameter blob** for K = N = 64 uint8 | 0 mismatches for random int8 weights, K in {8, 16, 24, 32, 40, 56, 64}, N in {16, 32, 48, 64}, with and without RELU |

**Runner bug found and fixed.** Output activations arrive in tile order. The
DarwiNN `OutputLayer.layout` (`OutputLayout`: tile-id maps, per-tile byte
offsets, local x/y offsets) tells the host how to gather them back into
y-x-z order, as libedgetpu does. `model.rs` did not parse it and the runner
returned raw tile order, so any output spread over several tiles came back
permuted (for 16x16x32: 128 B blocks transposed 4x4 within each 2 KiB). The
existing matmul test fits one tile and never showed it. `LayerInfo` now
carries the dims and layout, and `run_inference` relayouts each output layer.

**Numerics, verified bit-exactly:**

- Output stage: `sat_u8(round_half_even(clamp((x - in_zp) * mult, lo, hi)) + out_zp)`.
  The **clamp comes before rounding** (clamp_max 40.7 gives 41), which is
  why the compiler emits bounds one ulp inside the integer (130.99998 for
  131). Results above 255 or below 0 saturate.
- Ties round to even (exact x.5 cases at multiplier 0.5).
- FC per-channel multiplier is `F(F(s_in * s_w) * F(1 / s_out))`: 64/64
  channels match the compiler, `s_in * s_w / s_out` only 40/64. FC ops
  ignore `op.mult_bits` (setting it to 2.0 changes nothing).

**FC parameter blob (16.0):** `[f32 mult x N][i32 bias x N][i32 weight
zero point x N][uint8 weights, [K/4][N][4]]`. Weights are `W + 128` for int8
`W`. Without bias the bias block is dropped. Small N is padded to 16
channels. Parameter stream rules: units = N(12 + K)/64, `pop.rows_neg` =
32768 - floor(units / 2), `pop.b183` = units even, `infeed.f438` = N/16 - 1.
A one-unit (8 B) input transfer clears the pop/infeed loop fields (count,
limits, strides, `k157` / `k203`); encoding it as a count of one hangs.

**Fields decoded by perturbation:**

| Field | Finding |
|---|---|
| `op.row_a` / `row_c` | Wide-memory row of the per-channel **weight zero-point vector**. Moving the zero points to row 1 works iff `row_a = row_c = 1`; zeroing them or pointing elsewhere gives wrong results. FC: row 2, or row 1 without bias (compiler agrees) |
| Weight zero point | Per channel, arbitrary: storing channel `o` as `W + 60 + o` with zero point `60 + o` is exact |
| `op.cfg4` | Bias enable: `cfg4 = 0` equals the zero-bias result exactly; the compiler's no-bias FC sets 0 |
| `op.row_cfg` | Needed: clearing it alone gives wrong (saturated) output |

**Bits that hang the TPU when cleared** (all were "unexplained" or defaults):
`wideToNarrow` bit 543 (`rsv543`, set with `skip_en0`), `narrowToWide.tail_lvl`
(1 in FC/RELU copies), `wideToNarrow.tail_en870` in the FC input copy (the
RELU program emits 0 and runs with either), and the caching
`ringConsumer` bits 286-287 / 319 (`rsv286` = header rows - 1 with bias 2,
without 1). A hang is recoverable by reopening the device (chip re-init);
no replug was needed.

**14.1 FC form** (`cfg0 0xa5`, `dp_mode 3`, `reduce_mask 0b101`, no row
fields) on the 16.0 parameter layout runs but gives saturated garbage, so the
two forms differ in parameter layout too, not only in op bits.

**Still open:** `row_m`; `scsync.thr5` in the epilogue (16 after 16
outfeeds, 2 after one); the constant setup in the RELU program (mesh fills of
one-hot words and a copy op); the caching `ringConsumer` aux fields beyond
the rules above; K, N > 64 (multi-pass, multiple tiles).

## Loopback on hardware (2026-10-10)

`examples/loopback_probe.rs` on compiled dwconv models (C32 k3 unless noted;
C16 k3, C128 k3, C32 k7 for the row sweep). Each depthwise program caches
the weights in tile 0 (PARAMETER_CACHING infeed, `tiles = 0x1`) and every
inference broadcasts `C x (k^2 + 3)` bytes: ringProducer (tile 0, `to_c1`,
`dest 0x10000`) -> outfeed (`loop_en`, `loop_row 2046`) -> infeed
(`loop_en`, `tiles 0xffff`). Reference: the unmodified program. Each case
first runs the unmodified program with parameter blob A, then the perturbed
one with blob B (8012/8192 output bytes differ between A and B), so a
loopback that delivers nothing would return A's output.

| Change | Result |
|---|---|
| infeed `loop_en = 0`; `loop_k308 = 0`; `loop_k315 = 0`; all four (tinycoral) | hang |
| infeed `loop_k168 = 0` | wrong: channels 4-11, 16-31 (0-3, 12-15 exact) |
| infeed `tiles` 0x0001 / 0x00ff | hang |
| outfeed `loop_en = 0`; `loop_cfg = 0`; bit 120 = 0 | exact |
| outfeed `k173` 0 / 2 / 3; `f222` 0 / 0x7f | hang |
| producer `to_c1 = 0`; `dest = 0` | hang |
| producer `addr + 0x40`; `tile_mask = 0x2` | wrong on all channels |
| outfeed `loop_row` in [512 + rows, 1023] or [1536, 2047] | exact |
| `loop_row` elsewhere (0-511, 512 to 512 + rows - 1, 1024-1535) | hang |

Never stale: every non-hanging variant either matched B or was garbage
unrelated to A. Every hang recovered by reopening the device (34 in the widest sweep), no replug.

**Takeaways for codegen:** copy the infeed loopback constants exactly; the
outfeed's `loop_en` / `loop_cfg` decode is not what makes it a loopback
(`k173 = 1`, `f222 = 0x3f` are the candidates); any `loop_row` in the safe
range works, so the row is not an address the infeed reads.

**Still open:** why `loop_row` 0-(512 + rows) and 1024-1535 hang (a ring
pointer comparison?); what `loop_k168` selects; a loopback in a generated
program (experiment 3).

## Generated loopback on hardware (2026-10-10)

`elementwise::build_broadcast(rq, in, out, src)`: the identity/requant
program plus, before the output stage, a barrier, tile `src`'s ringProducer
-> loopback outfeed -> loopback infeed -> a ringConsumer on all 16 tiles,
and a second barrier. Checked by `examples/broadcast_hw.rs` and
`tests/codegen_hw.rs`: other tiles hold zeros, the host producers are
pointed at each received unit's wide address, and two input passes recover
which source byte landed where. **Bit-exact on all 16 tiles for sources 0,
3, 5, 6 and 15.**

Getting there, in order (each step was a hang until fixed):

1. **Epilogue.** The compiler waits on counter 6 for loopback rows and adds
   them to counter 5 (`thr5` = 16 + rows, `thr6` = rows, `smask = wait =
   0x60`, `sync counters 0x60000`; C8 dwconv 18 / 2, C32 22 / 6).
   `codegen::epilogue` takes the row count.
2. **Host producer `r0_val`.** Compiled dwconv uses `t + 6` after its 6-row
   loopback; ours now use `rows + t`. Fixing it alone did not stop the
   hang; whether it is needed was not tested.
3. **Consumer form.** A plain consumer (no `cbuf` / aux / `rsv286`) never
   completes. Bisected on the compiled program: clearing `aux_en0` or
   `rsv286` hangs, `aux_en1` / `rsv319` corrupt data, the consumer's `s_*`
   and the producer's `r0` / `r1` sync fields change nothing. No
   "no aux" combination ran, also with infeed `loop_k308` / `loop_k315`
   cleared or set to 1 or 2. Copying the C8 3x3 consumer verbatim (plus its
   infeed `count_m1 0x1f` / `count_neg 0x20`) ran.

Bisection pieces (`elementwise::part`, `PARTS=` in the example): barriers
alone run; consumer without producer hangs, as expected.

**Landing layout** (C8 consumer, `elementwise::bcast_unit_addr`): 24 units of
16 B arrive (source bytes 0-383; 384-511 never do). Units 0-2 -> 0x1fe8,
0x1fec, 0x1ff0; 3-11 -> 0x1fa0..0x1fc0; 12-14 -> 0x1ff4..0x1ffc; 15-23 ->
0x1fc4..0x1fe4. That is 3 aux + 9 slot units per 192 B, the depthwise
`k^2 + 3` split. A host producer at wide address A returns 16 B from A and
16 B from A + 4 (rest zero), so one wide address step of 4 is 16 B of
received data. Raw broadcast without this re-layout needs a consumer form we
have not found.

**Runner fix.** A hung program can announce an output (tag 3) and never
send it; `bulk_recv` then blocked forever (one probe sat for 8 minutes).
All in-loop USB transfers in `runner.rs` now time out after 2 s
(`DeviceError::Timeout`), so hangs are recoverable by reopening.

**Still open:** a consumer that lands the stream linearly; why only 384 of
512 B arrive (row size vs consumer slots); what `loop_k168` selects.

## CONV_2D generated (2026-10-10)

`codegen::conv::Conv`, checked by `examples/conv_check.rs` and
`tests/codegen.rs::conv_matches_compiler` (byte equality of both
executables, relocations, output layout and parameter blob, weights from
`tools/isa_corpus/conv_ref.py`) and `tests/codegen_hw.rs` (bit-exact on
the TPU against a host reference, random weights, ReLU, zero points 3-250).
**Byte-identical on 8 corpus models:** 16x16, Cin 16, same padding,
stride 1, k 1 (Cout 32) and k 3 (Cout 8, 16, 32; four quantizations, ReLU,
ReLU6). `examples/isa_emit.rs` prints a compiled bitstream as `emit` calls
to transcribe from.

**Program shape** (one EXECUTION_ONLY + one PARAMETER_CACHING, all models
with same padding and H, W multiples of 4: 3 TensorOps, one multicast):

1. Element-wise skeleton: prologue, one-hot constants, input over 4 rows of
   4 tiles (4x4 pixel patch per tile), two copy ops into the compute layout.
2. Halo exchange (k > 1): mesh moves north (0x17), south (0x15), then west
   (0x16) and east (0x18) over the full padded height, so corners follow.
   Each move is three instructions by tile row / column: receive only, send
   and receive, send and fill own halo with the input zero point. Mesh
   addresses are byte offsets in the padded patch, increments are words.
   Column moves carry sync flags 6 / 0xa (value 6 for 1-pixel halos at
   Cin >= 8; 5 at Cin <= 4; 7 for 2-3 pixel halos).
3. Weight multicast: tile 0's ringProducer sends the cached blob straight to
   `dest = 0xffff` (no scalar memory); every tile's ringConsumer (opcode
   0x12) puts the 3 header rows in its aux rows (0x1fe8..) and weight rows
   in slots of 8 address units below; the epilogue waits for the words on
   counter 6 (`thr6` = 3 + K/4) and not on counter 5.
4. One conv TensorOp per tile, then the element-wise output path.

**Rules found:**

- Address-generator increments apply from the last element of the inner
  loops (`inc = stride - span(inner)`); one helper (`walk`) produces every
  mesh and TensorOp walk.
- Narrow layout in bytes, `tb` = 16 Cin, `tp` = padded patch: compute input
  at 0 (padded), landing at `tp`, constants at `tp + 2 tb`, conv output at
  `tp`. The copy-op formulas (`codegen::copy_op`) fit element-wise and conv
  k = 1..7.
- k = 1: one pass, `loop0` = Cin/4 words. k = 3: one pass per 4 input
  channels (`loop2 = Cin/4 - 1`) over the 9 taps with partial sums
  (`psum_base` = slots - 64); the consumer refills its 9 slots each pass, so
  weight rows stream as `[Cin/4][ky][kx]` (confirmed by blob equality).
  k = 2, 4-5, 6-7 and Cin < 16 use other loop schemes.
- Cout < 16 is padded to 16 in the blob with zero multiplier, bias, weight
  zero point and weight bytes.
- **Clamp bounds** are `((q - zp) * s) * (1 / s)` in f32 (q = 0, 255):
  119/119 corpus conv, FC and depthwise models. FC now uses it too (it had
  a "one ulp towards zero" approximation that matched the one model).
- A tile output of one wide word (Cout <= 16) uses the circular
  single-row narrowToWide form; larger outputs use `w_lim0`.

**Hardware check of the halo:** with the reference's border padding set to
0 instead of the zero point, 1881 of 8192 values differ (the border
pixels); with the zero point all match. Interior tile edges, which depend
on the neighbour moves, are exact.

**Cin 32-128 (same day):** byte-identical on the Cin 32, 64, 128 corpus
models and bit-exact on the TPU (Cin 32/64/128 with Cout 8/16/32, including
combinations the corpus lacks). Three rules:

- **Input buffer.** One tile's input is `S = Cin / 16` wide words. S = 1
  lands in one slot at 0x2078 (the row's 4 words overwrite it in turn);
  S > 1 uses a circular buffer of S slots at `0x2078 - 8 (S - 1)` (0x2070,
  0x2060, 0x2040), refilled once per tile of the row (`w_lim1` / `cnt1` =
  3, `slots` = `s_cnt` = S, `s_id` 0x2b, mode 3). `s_val = -tb / 4` in both
  forms. Constants sit 4 below the buffer. Element-wise's input stage is
  the S = 2 circular form.
- **Header `sync_val` = 0x8000 x passes** (Cin / 4 for k = 3: 0x20000,
  0x40000, 0x80000, 0x100000). Our `sync_val` field was 19 bits; the
  "unexplained" `wideToNarrow` bits 643-645 are its top bits (now 22 bits,
  all corpus bitstreams still round-trip).
- **Mesh beats.** A pixel moves in beats of at most 16 words; Cin 128 (32
  words) adds the beats as the outermost loop (`(16, 1)` innermost, `(2,
  16)` outermost), and that level does not count in `in_mode` (2 x levels +
  1).

**Cin 1, 3, 4 (same day):** byte-identical on the three corpus models
(parameters included) and bit-exact on the TPU (Cin 1 / 3 / 4 with Cout 8 /
16 / 32).

- **Single pass.** With one 4-byte word per pixel the TensorOp takes all 9
  taps in one pass, as k = 1 does (`loop0 = k^2 Cin/4 - 1`, `dp_mode` 2,
  `reduce_mask` 1, no partial sums); the input walk is a row of k pixels
  contiguous, then k rows, then the 4 x 4 output pixels.
- **Walk rule refined:** a loop of count 1 takes the next loop's increment
  (1 when innermost); the earlier schemes are special cases.
- **Sub-word input.** For tb < 256 B a row of 4 tiles is one wide word, in
  a one-slot circular buffer (`w_inc0`, `wide_circ`, `cbuf`, mode 3); the
  narrow landing area is still a full word (`lb = max(tb, 256)`: copy 2
  reads `tp + lb`, constants at `tp + 2 lb`, `size64 = lb / 64`,
  `in_twait = lb / 64`, `s_val = -lb / 4`), the copy ops set `par_hmode` /
  `psum_hmode`, and there is no `sync counters 0x800` after the pop.
- **Cin 1 / 3 widening.** The host sends Cin bytes per pixel; the copy ops
  read them byte-wise (`in_hmode`, per-level `in_mode` 1, `par_mode` 2,
  `cfg1 = 0x6b + 0x100 Cin` in the second) into 4-byte pixels; the
  constants get a 64-word zero block and take two wide words at 0x2078,
  so the input buffer moves to 0x2070. The rest is the Cin 4 program with
  zero weights in the extra channels. Transcribed: with two samples, the
  input DMAs' `head_words_m1` is a per-Cin table and the parameter walk
  increment is -1 for Cin 3, 0 for Cin 1.
- Column-move sync value is 5 for 1-word pixels.

**Stride 2 (same day):** byte-identical on the k = 1 and k = 3 stride-2
corpus models and bit-exact on the TPU (k 1 Cin 16; k 3 with Cin 3 / 16 /
64, Cout 8 / 16 / 32, including Cin 3 with ReLU, which the corpus lacks).

- **Padding** is TensorFlow's same padding, `total = max((out - 1) s + k -
  in, 0)` with the extra pixel after: k = 3 stride 2 pads 1 pixel bottom /
  right, none top / left. The patch is then 5x5 and only the north and west
  mesh moves remain (10 mesh ops). Checked on the TPU: a reference padded
  on the other side gets ~98% of the outputs wrong.
- **Sync flags:** the west move then waits on 0xa alone, so 0xa is the
  north move's flag and 6 the south one's.
- **Walks:** the output pixels step by `s` pixels (`s cq`, `s rs`), 2x2
  per tile; partial sums are 4 words per output pixel (`psum_base` =
  slots - 4 npx); the output narrowToWide walks words x pixels x rows
  (`n_lim0` = Cout/4 - 1, `n_lim1` = `n_lim2` = 4/s - 1).
- **Walk rule, final:** every level's increment is its stride minus the
  inner span, including loops of count 1 (with their nominal stride: a
  word or a pixel step). The "count 1 copies the next increment" rule was a
  coincidence of the samples.
- The Cin 1 / 3 copy ops' output walk is the regular copy op's (rows one
  level up in the first copy), so it follows the patch size; Cin 3 stride
  2 (5x5 patch, not in the corpus) is bit-exact with it.

**Still open:** Cin 8 (18 weight rows, no slot count, other row fields),
Cout >= 64 (1 KiB per tile, a different narrow arrangement), k = 2 and
4-7, sizes other than 16x16, valid padding.

## Wider images without the compiler (2026-10-10)

Goal: support image sizes other than 16x16 from what we know, without
reading compiled programs of those sizes. Result: **width 32 and 64
(height 16) bit-exact on the TPU** for k 1 / 3, stride 1 / 2, Cin 16 / 32,
Cout 8-32 (`examples/conv_wide_hw.rs`, `tests/codegen_hw.rs`); 16x16
output stays byte-identical to the compiler.

**Design.** Everything verified so far assumed 16 tiles with a 4x4 pixel
patch each, and the input wideToNarrow's head / tail / skip fields cannot
be decoded from 16-wide samples (one tile's words and one image row's
words are both 4 Cin). So avoid them: a row of `16 m` pixels is
byte-for-byte a row of 16 *superpixels* of `m Cin` channels. The 16x16
loader, copy ops and halo exchange, verified for 32 / 64 / 128 channels,
then deliver each tile its 4 x 4m patch. The halo is whole superpixels (m
pixels instead of 1, a superset; costs memory, not correctness). Only the
conv op is new: real-pixel strides (`Cin / 4` words) inside the
superpixel row stride, `in_base` = the first window's origin (`left m -
pad` pixels in), `4 x 4m` output pixels per tile, and the output layout
to match.

**Hardware bisection.** The first run hung on every wide case. Separating
the new pieces with 16-wide cases (built with `Conv::build_unchecked`):

| Case | Result |
|---|---|
| 3x3, 16x16, Cin 32 (the loader / halo of a 16x32 Cin 16 image) | exact |
| 1x1, 16x16, Cout 64 (1 KiB of output per tile, 4 wide words at 0x2078) | exact; the compiler arranges this differently, ours works |
| 3x3 16x32 Cin 16 Cout 8; 3x3 stride 2; 1x1 16x32 | hang |

All wide cases hanging, even with 8 output pixels and `in_base` 0, pointed
at what they share: the input stage sized transfers from the real channel
count (`16 Cin` bytes per tile, `skip = Cin x column`) instead of the
superpixel one. Fixed, all wide cases are exact. Checks: 127-213 distinct
output values; `in_base` one pixel off makes 98% of outputs wrong.

**Recovery.** One hang left the chip unable to initialize (`Usb(Cancelled)`
on open, also for the matmul test); a USB reset through `nusb` then made
the device drop off the bus and it needed a replug. Not repeated: after
later hangs, reopening recovered.

**Heights 32 / 64 (same day), also without the compiler.** Bit-exact on
the TPU up to 64x64 (14 cases: k 1 / 3, stride 1 / 2, Cin 16 / 32, Cout
8-32, ReLU).

*First attempt, a loader of our own (dropped).* One strided infeed per
tile gathering its rows (`d0` a row segment, `d1` the rows at the image
row stride; the default `d1_stride = -0x3fff` decodes as a count-1 level
under the delta rule), a linear one-slot consumer and one wideToNarrow
for all tiles. It hung even at 16x16. Bisection from the working band
loader:

| Variant (16x16, k 1) | Result |
|---|---|
| per-tile infeeds of contiguous chunks (gather sizes, no strides) | hang |
| band loader, each band sent to 1 tile at a time (each band read 4x) | hang |
| band loader, bands to pairs of tiles (each band read 2x) | hang |
| band loader, only row 0 split per tile | hang |
| `k434` / `f438` of the 1-tile infeeds varied (1, 0) | hang |
| band loader, rows infed in reverse order (correct `buf_off`) | exact |
| band loader, each band in 4 quarter infeeds (16 infeeds) | exact |

So staging is random-access (`buf_off` honoured, any order) and the
number and size of infeeds are not limited, but **each staged byte can be
infed only once**: every variant that re-reads a range hangs. The
per-tile gather obeys that, so its hang is in its consumer /
wideToNarrow forms, which were guesses.

*Design used: stacked bands.* A 4-row band through the verified loader and
copy ops is a unit that deposits 4 image rows into the compute layout.
For `height / 4` rows per tile the program runs it once per band: the
image is popped once, band `h`'s infeeds read rows `(row x bands + h)` of
the staged image, the landing area is reused, and the copy ops write rows
`4 h ..` of the padded patch (`copy_op` takes the row offset). The copy
ops' `tp` term is only the nominal stride of a count-1 loop level, so
they carry over unchanged. The halo's vertical extents and the conv op's
output rows follow the patch's row count. Check: with the second copy's
row offset left at 0, exactly rows 3, 7, 11, ... come out wrong.

**Still open:** Cin < 16 at larger sizes (superpixel channel counts and
the sub-word input form untested), output above 2 KiB per tile (wide
staging above 0x207f untested), sizes not a multiple of 16.


## Large FC layers and the MNIST MLP (2026-10-14)

Corpus extended with the MNIST layer shapes (`gen.py`: K 784 x N 128 RELU,
128 x 10, and steps between). `codegen::fc` now covers K up to 1024
(multiple of 8) and N 4-64 or a multiple of 64 up to 1024, and is
byte-identical to 16.0 on all 28 supported corpus FC models, parameter
blobs included (`examples/fc_check.rs`, weights from `fc_ref.py`), and
bit-exact on the TPU for 784 x 128 RELU, 128 x 10, 128 x 64, 1000 x 64,
256 x 256, 64 x 1024 and 64 x 4 with random weights (`tests/codegen_hw.rs`).

**Large K: spread, then gather.** The compiler never loops over K in
passes. The input is infed 64 B per tile over `T = ceil(K / 64)` tiles,
then gathered over the ring into every compute tile, and one TensorOp
walks all K / 4 words.

- Infeed: one infeed per row of 4 tiles (256 B, `buf_off` 0x20 per row,
  `pop_wait` = `buf_off` + 1), each with its own ringConsumer. One row: a
  single slot at wide 0x2078, `s_val` -64. Several rows: 4 slots at
  0x2060 (`sdims`, `s_cnt` 4, `s_val` -256).
- Each tile keeps its 64 B with a wideToNarrow per column `c`: `skip_m1`
  = 16c - 1, `skip_end` = base + 16c - 1, `tail_f871` and `rsv543` set,
  `tail_en870` cleared (column 0: no skip, `tail_en870` set). Full tiles
  share a copy per column; a partial last tile gets its own.
- Staging lands at narrow word 0xc0 for one row, 0x180 for more (above
  the gathered input).
- Gather, per input tile `t`: wideToNarrow on the compute tiles to narrow
  `in_base + 64t` (`sync_val` 0x8000 + (t + 1) / 2, `sync_en1` for even
  t), ringConsumer `s_val` = 16t - 64, ringProducer on tile t with
  `r0_val` = t, narrowToWide of tile t's staging.
- Output after the input: `out_base` = K (K < 64: input at 64, output at 0).

**Large N: 64 channels per compute tile.** Tiles 0.. each run the same
TensorOp on their own parameter blob (the blobs are concatenated in the
stream: one ringConsumer and one infeed per tile, `buf_off` advancing by
the tile's 64 B units). Each tile outfeeds its channels; producers count
on after the gathers (`r0_val` = T + tile) and the epilogue's `thr5` is
the total of ring rows (T + compute tiles). The OutputLayout puts tile t's
channels at `t x row`.

**Parameter streams over 1024 units** (64 B): `d1_limit` full rows of
1024, `count_m1` the last row's units - 1 (K 2048 / 4096, N 1024).

**Small N.** Parameters are padded to 16 channels: up to the next multiple
of 4 the padded channels are computed (the first channel's multiplier,
zero bias and weights, weight zero point 128); beyond that everything is 0.
The op computes `ceil(N / 4)` words; the host gets N rounded up to 8 bytes
(a one-unit outfeed clears its loop like the one-unit input pop). N < 4
uses a different op form (`w_zp` set, no row fields): not supported.

**MNIST MLP** (`tools/mnist/train.py`: Flatten -> Dense 128 RELU -> Dense
10, int8). The new front end `graph::Graph` reads the CPU `.tflite`
(RESHAPE is free; the int32 SHAPE / STRIDED_SLICE / PACK feeding it are
skipped), maps int8 activations to uint8 (+128) and builds one program per
layer. **First model with no compiler involved: all 10000 test images on
the TPU give 97.35%, byte-identical to the host reference on every image**
(`examples/mnist.rs --device`), and the same as TFLite's CPU interpreter:
same top-1 on every image, 3 of 100000 logits differ by 1 (f32 multiplier
vs TFLite's fixed-point one). 3.8 ms per image, most of it reloading the
parameters (~100 KB) of both layers.

**Cached parameters do not coexist** (`examples/param_cache_hw.rs`): after
loading FC A (784 x 128) and then FC B (128 x 10), A runs wrong. Both blobs
start at wide address 0 of tile 0, so `Graph::run` reloads each layer's
parameters per inference. Placing layers at different wide rows or tiles
is the fix (the parameter placement item of the roadmap).

**Runner fixes found on the way.** The output relayout derived bytes per
element from `size_bytes / elements`, wrong when the output is padded to
twice its length (an FC with 4 channels sends 8 bytes; the compiler's own
model has the same layer): `LayerInfo.elem_bytes` now comes from the
DarwiNN `data_type`. And every `load_params` / `run_inference` submitted
a new event and interrupt read while the previous call's were still
pending; they piled up until usbfs failed with ENOMEM (errno 12) after
~1900 calls. The runner now tops up to one pending read.

## MaxPool, larger FCs and the MNIST CNN (2026-10-14)

**MAX_POOL_2D 2x2 stride 2** (`codegen::pool`), byte-identical to 16.0 on
the 5 corpus models (H, W 16 / 32; C 32 / 64 / 128; C 8 has one instruction
less, not covered) and bit-exact on the TPU, nonzero zero points included.
The program is the element-wise one up to the compute stage: the same band
loader (one staging row of `tw C / 64` slots, column skips of `tw C / 4`
words) and the two copy TensorOps, generalized to a `tw x th` patch (first
copy rows `0 ..= th - 2`, second the last row; the staging row is re-read
for every output row, stride 0). Then:

- a narrowToWide regroups the patch into wide words of 4 pixels x up to 64
  channels (levels: pixel, word, 4-pixel group, row, 64-channel block);
- the pool op is opcode 0x02, a TensorOp variant with window loops
  (`loop0` / `loop1` = 1 for 2x2) that reads the windows on its parameter
  path from those wide words (`par_hmode`, `par_mode*`) and writes outputs
  pixel by pixel; its datapath is an identity requantization `clamp(max -
  zp, -zp, 255 - zp) + zp` (`offset` = -zp);
- the output stage switches to the element-wise form (no circular buffer,
  producer `cnt1`) at 512 B per tile.

Wide memory, top down: the regrouped patch (`4 x words` units below
0x2080), one staging row (8 units per slot), 4 constant words. Every loop
increment follows the walk rule.

**FC above K = 1024**: still no passes. Each input tile takes 128 B (K <=
2048) or 256 B (K <= 4096), so 16 tiles hold the input; K 2049..3072 is not
in the corpus. Staging is `ceil(K / 256)` wide words at `0x2080 - 8 x
slots`; column c reads word `c ch / 256` with a skip of `(c ch mod 256) /
4` words; full rows' ringConsumers walk all slots with `grp` = words per
tile - 1 and `gstride` = 3 floor(K / 256) + 1 (fits 2048 / 3136 / 4096, not
decoded). Narrow staging sits at `round_up(K, 256) + 512` bytes, which also
gives the 0xc0 / 0x180 of smaller K. The parameter pop's `rows_neg`
saturates at 1024 rows.

**Conv Cout 64** (k 3, Cin 32 at 16x16, 1 KiB of output per tile):
byte-identical with no change beyond the check, parameters included. Cin 16
with Cout 64 is not: the compiler moves the narrow buffers.

**MNIST CNN** (`train.py --model cnn`: Conv 3x3 32 ReLU -> MaxPool -> Conv
3x3 64 ReLU -> MaxPool -> Dense 10, same padding). The front end maps it to
5 programs; the host pads 28 -> 32 and 14 -> 16 with the zero point and
crops after each (exact: same padding with stride 1 sees zero-point rows
either way, and a 2x2 pool's kept outputs never read the padding), and the
first conv's single channel goes in as 16 with zero weights. **All 10000
test images on the TPU: 98.66%, byte-identical to the host reference on
every image**, the same top-1 as TFLite CPU (4 of 100000 logits differ by
1); 7.9 ms per image, most of it USB round trips and parameter reloads for
5 programs. edgetpu_compiler rejects this model (Keras 3's Flatten has a
dynamic shape), so only its per-layer programs were compared.

## Element-wise ADD, wider 1x1 convs and ResNet-8 (2026-10-14)

**Element-wise ADD** (`codegen::add`), byte-identical to 16.0 on all 8
supported corpus models (16x16x32 / 16x16x64 / 32x32x16 / 32x32x64, scale
ratios 1/4 .. 4, with and without RELU) and bit-exact on the TPU, including
ResNet-8's arbitrary scale ratios. The program is two runs of the band
loader (now shared with the pool in `codegen::band`): A staged at narrow
`B` into the patch at 0, B staged at `2B` into the patch at `B` (B = bytes
per tile), then a constant setup and one TensorOp.

The arithmetic is an integer dot product per value: `out = clamp((wa a +
wb b + offset) mult) + out_zp`, with

- `(wa, wb)`: `s_small / s_big` (exact rational of the f32 scales) as the
  best fraction with denominator <= 32767 (Python's
  `Fraction.limit_denominator`), the larger scale getting the denominator.
  1 : 1 for nearly equal scales, 4 : 1 for exact powers, 32767 : 32766 or
  3157 : 3156 when the scales differ by 1e-5 / 3e-4 (8/8 corpus models; an
  f32 ratio gets one of them wrong). The weights are 16-bit halves of mesh
  fills from all four directions (opcodes 0x15-0x18) at narrow `2B`, moved
  to wide 0x2070 for the op's parameter path;
- `offset = -(wa za + wb zb)` and `mult = s_big * (1 / (w_big s_out))` in
  f32 (12/12 corpus models including the 1-D ones); clamps as FC / conv.

The add op reads word pairs of both patches (`in_hmode` 2, `in_mode*` 2;
levels: 2 words, the two patches, pixels, rows, channel words) and writes
over B's patch. Wide placement: with a 2-slot image row the constants sit
on top (0x207c) and the staging below, otherwise the staging on top.

**Runner: several input layers.** The compiled ADD has two named input
layers with their own relocations. The runner now gives each input layer
its own address in one buffer (layers back to back) and serves each input
request the slice it asks for; before, it sent the whole buffer for every
request.

**Conv 1x1 above 16 input channels**: the compiler splits Cin 32 into two
passes of 4 words (a 4-slot weight buffer refilled per pass) with partial
sums, the 3x3 multi-pass form; byte-identical for 32 -> 64 stride 2,
parameters included. Stride-2 3x3 with Cout 64 and 64 -> 64 needed no
change. The 32-wide convs differ from the compiler (it walks real pixels in
a wider patch where ours uses superpixels): both are correct on the TPU, so
`conv_check` compares them only with `--all-sizes`.

**Global average pool as an FC.** The compiler's 8x8x64 AVERAGE_POOL_2D is
a 127-instruction program that reduces across all 16 tiles over the mesh
(directions, slots and sync ids not decoded; one corpus model). The front
end instead lowers a global average pool to an FC (K = H W C, weight 127
from each pixel to its channel) that reproduces TFLite's int8 rounding
exactly: input and output zero point 128 (summing the raw int8 values) and
the smallest f32 multiplier at or above 1 / (127 n) for which the TPU's
round-half-even of every possible sum equals TFLite's `(sum +- n/2) / n`
(ties land just past .5 in magnitude; checked exhaustively at build time).

**ResNet-8** (`tools/cifar10/train.py`, the MLPerf Tiny model ending at
the logits, 20 epochs: 75.0% float, 75.2% int8). The front end now runs a
DAG (every activation kept by tensor id) and maps the model to 14 programs:
9 convs (the 3-channel input as 16 with zero weights, the 8x8 stage padded
to 16x16), 3 ADDs with fused RELU (8x8x64 padded to 16x16x64), the average
pool FC and the dense head. **All 10000 test images on the TPU: 75.22%,
byte-identical to the host reference on every image**, the same top-1 as
TFLite CPU on all of them (6 of 100000 logits differ by 1); 25.6 ms per
image, most of it 14 parameter loads and USB round trips per image.

## Speed against edgetpu_compiler (2026-10-14)

`examples/bench_compare.rs`: ResNet-8 (a retrained export, 68.9% int8,
with static shapes so that `edgetpu_compiler` accepts it: all 14 ops in one
subgraph, 115.75 KiB of parameters cached on chip, nothing streamed), the
same `.tflite` for both, same TPU, first 1000 CIFAR-10 test images.

| | edgetpu_compiler | ours |
|---|---|---|
| programs per image | 1 | 14 |
| per image | 0.96 ms (1041 images/s) | 32.4 ms (31 images/s) |
| parameter loading | 2.4 ms once | 10.6 ms per image (every layer reloads) |
| inference with parameters resident | 0.96 ms | 14.5 ms (sum of 14 calls) |
| accuracy / top-1 = TFLite CPU | 68.3% / 999 of 1000 | 68.4% / 1000 of 1000 |
| logits differing from TFLite CPU | 800 of 10000 | 26 of 10000 |

Our time is per-call overhead, not compute: every program sends its
instructions and moves its whole input and output over USB (the 64 x 10 FC
alone takes 0.34 ms, each 32x32 layer about 1.2 ms), and the compiled model
keeps activations on chip across all 14 layers. Of the 10.6 ms of reloads,
6.2 ms is the average-pool FC's 263 KB blob (the compiler's mesh reduction
needs no parameters). About 7 ms is host glue (padding / cropping) and
per-call setup. Numerically ours is closer to TFLite: our average pool
rounds as TFLite does, and the compiler's program differs from TFLite's
top-1 on one image.

Closing the gap, in order: per-layer parameter placement (-10.6 ms), a
native global average pool, then several layers per program with
activations kept on chip (the only way to approach 1 ms).

(Correction, 2026-10-15: the 32.4 ms included one host-reference pass per
image, about 6.5 ms; `bench_compare` read its clock after timing the
reference separately. `eval` measured the same code at 25.6 ms, and host
glue is under 0.1 ms, not 7 ms.)

## Parameter placement and a native average pool (2026-10-15)

**Placement.** Wide memory addresses are 64 B units, a 256 B row is 4 of
them (`row_a` 0x7ff is address 0x1ffc), and every generator keeps its own
wide buffers above 0x1e00 (conv weight slots from 0x1e60, staging from
0x2000). `Fc::build_at(base)` / `Conv::build_at(base)` move a layer's
cached parameters up by `base` units (a whole row):

- FC: caching ringConsumer `addr` 0xe + base, `aux_addr0` base,
  `aux_addr1` 8 + base; header wideToNarrow `wide_addr` base; TensorOp
  `par_base` 0xc + base, `row_a` / `row_c` 2 + base / 4.
- Conv: caching ringConsumer `addr` 2 + base; the multicast ringProducer
  reads from `addr` base. (In both, the caching consumer's `addr` is 2
  above where its rows land.)

`examples/param_cache_hw.rs`, on the TPU, each case loading layers one
after the other and then running all of them: two FCs (one on 2 tiles),
two convs, FC + conv + FC packed back to back, and bases 0x1000 / 0x1d74
are all exact after every load. Two layers at base 0 still overwrite each
other. **A parameter row takes 4 units** (256 B) even when it holds less:
packing an FC 1024 x 64 or a conv 64 -> 32 at one unit per row corrupts
it (2-3 units per row untested). So `wide_units` = 4 x (3 header rows +
weight rows) per tile, and `Graph::build` places the layers one after the
other from 0 (ResNet-8: 0x7a8 of 0x1e00 units, 122.5 KiB), `Graph::load` caches them
all once and `Graph::run` sends only instructions and activations.

**Global average pool** (`codegen::avg_pool`, ResNet-8's 8x8x64 head),
decoded from `pool_C64_H8_W8_opavg_padvalid_size8_stride8`. **Byte-identical
to 16.0** (output layout included; no parameters) and bit-exact on the TPU.
The program:

1. The band loader with a 2x2 patch per tile (staging at narrow 0, patch at
   0x1000). It has no `counters 0x800` sync after the input descriptor at
   256 B per tile; `band::load` now emits it only above 256 B (pool / add
   unchanged).
2. Column gather (mesh north, 0x17): row-0 tiles copy their patch to 0x1100
   through wide 0x2034 and receive the tiles below at 0x1200 / 0x1300 /
   0x1400. Tiles in between forward through a one-slot buffer at 0x1500;
   a forwarder `hop` tiles from the sender has sync records `s0_val` =
   256 (hop - 1) + 4, `s1_val` = 256 hop - 4 (ids 0xa / 0xe). The
   compiler orders the moves: all columns at distance 1, then distance 2's
   sends, then per column the rest.
3. Row gather (mesh west, 0x16): tile 0 spreads its own 1 KiB column block
   into an 8x8 image at narrow 0 (through wide 0x2034, rows of 2 pixels at
   a 512 B stride) and receives tiles 1-3's blocks at x = 2, 4, 6 of every
   row (forwarding at 0x1510, records 1 KiB (hop - 1) + 4 / 1 KiB hop - 4,
   ids 8 / 0x14).
4. A mesh fill puts the word 1 at 0x1520; the max pool's regroup moves the
   image into 16 wide words of 4 pixels at 0x2040 (plus undecoded
   narrowToWide bits 554-756, copied).
5. One TensorOp on tile 0 with the pool op's `cfg0` 0x127 and an 8x8 window
   (`loop0` / `loop1` 7): the image on the parameter path minus `w_zp`,
   times the input word 1, summed over the window; `mult` 1 / 64, `out_zp`
   = `w_zp` = the zero point. Tile 0 outfeeds 64 B.

The compiler's `mult` 1 / 64 rounds ties to even, where TFLite rounds the
raw sum / 64 half away from zero; that is most of its 800 differing logits.
`GlobalAvgPool::tflite` centres on 128 (int8) or 0 (uint8) and takes the
smallest f32 above 1 / 64 that reproduces TFLite's rounding for every sum
(`avg_pool::exact_multiplier`, also used by the FC lowering, which stays
for other shapes). ResNet-8's host reference is unchanged (same 334 of
100000 logits from TFLite as with the FC).

**Results** (ResNet-8, same TPU and images): all 10000 test images on the
TPU bit-exact with the host reference, 14.8 ms per image (`eval`; 25.6 ms
before on the same 1000 images). `bench_compare`: compiled 0.97 ms, ours
14.8 ms; parameters load once in 4.6 ms (2.2 ms for the compiled model);
the 14 calls alone take 14.7 ms (the average pool 0.55 ms instead of an FC
with a 263 KB blob), so what remains is per-call overhead: instructions and
activations over USB for every layer. MNIST MLP 3.8 -> 0.85 ms, CNN 5.4 ms,
both bit-exact on all 10000 images.

## Layers chained on chip in one call (2026-10-15)

Where a ResNet-8 inference goes (14 calls): 98 KB of instructions, 250 KB
of activations to the TPU and 150 KB back, against the compiled model's
25.5 KB, 3 KB and 16 B. Moving activations is most of the time, so the
layers have to hand them over on chip. `examples/chain_hw.rs`, on the
TPU:

1. **Splicing** (`codegen::chain::splice`): two complete conv programs as
   one bitstream, each with its own host input and output, run in one
   call, both exact. Each part keeps its prologue and epilogue; only the
   inner `halt` / nops / `end` go, the tile sequence numbers continue,
   and relocations move with their part. The first part's completion
   event mid-program is harmless (the runner waits for the outputs).
2. **conv -> conv on chip** (`conv::Io`): A's conv TensorOp writes its
   output straight into B's padded patch interior, at narrow 0x1000 on
   every tile, with a 3-level output walk (words, x, then y at the padded
   row stride). A skips its output stage; its epilogue waits for no
   outfeeds (`thr5` 0, `thr6` the weight words as before). B skips
   constants, copy ops and infeed, and its halo exchange and conv op read
   the patch at 0x1000. **Bit-exact**, also for 16 -> 32 -> 32 -> 16 with
   ReLU and the patches alternating between 0x1000 and 0x2000 (10 inputs).
   This works because every layer splits the image over the same 4x4 tile
   grid, so a conv's output pixels are already on the tile the next
   layer's patch needs.
3. **Speed:** those three convs take 0.81 ms as one chained call against
   2.22 ms as three calls.

Limits so far: 16x16, stride 1, conv only (`chain_program` refuses
others). Next for ResNet-8: chaining at 32x32 (superpixel patches,
4-row bands) and stride 2, add (two patches, the shortcut kept alive in
narrow memory), native 8x8 layers instead of host padding, and pool -> FC
from tile 0.

