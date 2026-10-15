# Edge TPU knowledge base

Everything this project knows about the Google Coral Edge TPU (USB
Accelerator), as of 2026-10-08. It covers the hardware, the file and USB
formats, the instruction set, what we decoded ourselves, what the chip can do
beyond Google's compiler, and what is still unknown.

- Dated research log with the reasoning behind each step:
  [`CODEGEN_RESEARCH.md`](CODEGEN_RESEARCH.md).
- Field-level ground truth: [`src/isa/layouts.rs`](../src/isa/layouts.rs).
  Each field we decoded carries its rule and evidence count in a comment.

Confidence labels used below:

- **Verified**: holds on every instance in our corpus, or was checked on
  hardware by tinycoral.
- **Strong**: holds on nearly all instances; exceptions are listed.
- **Provisional**: a consistent pattern whose meaning is a hypothesis.
- **Unknown**: observed but not explained.

Nothing in this document has been run on hardware by this project yet.

---

## 1. Hardware

| Part | Facts | Source |
|---|---|---|
| Core | Custom Google design, internally "DarwiNN" ("beagle" for this chip). Not RISC-V | tinycoral, libedgetpu |
| USB microcontroller | An 8051 runs the USB firmware (`apex_latest_single_ep.bin`), uploaded over DFU on every plug-in | `src/usb/dfu.rs` |
| Scalar core | Runs the program and issues DMAs. 32 registers, 8 predicates, integer and float32 ALU, exp2 and reciprocal units, hardware loops and branches | tinycoral |
| Scalar memory | 16 KiB, word addressed; the scalar core loads and stores it | tinycoral |
| Tiles | 16 tiles in a 4x4 grid (tile t at row t/4, column t%4) | tinycoral |
| Narrow memory | 192 KiB per tile, activations, 4-byte words | tinycoral |
| Wide memory | 512 KiB per tile = 2048 rows of 256 B, weights; 8 MiB total. One row = 256 MACs per tile per step | tinycoral, our row fields |
| Peak | ~4 TOPS int8 at ~480-500 MHz | libredgetpu, tinycoral |
| Interconnect | Ring bus (scalar core to tiles, multicast by 17-bit bitmap, bit 16 = scalar core) and mesh bus (4 neighbours) | patents, tinycoral |
| DMA units | infeed, outfeed, avDataPop, parameterPop | patents, `src/chip.rs` |
| Non-linear unit | Per tile; an 8-segment quartic spline whose coefficients are instruction immediates | tinycoral |
| Arithmetic | Tiles are uint8 x uint8 -> int32 only. Float exists only on the scalar core | tinycoral |
| Safety | No signing or checksum on programs. A bad program stalls the chip; recovery is re-init or replug (sometimes the firmware still answers control transfers) | tinycoral |
| Host link | Device-to-host USB ~55 MB/s; at batch 1 USB round trips dominate | tinycoral |

Largest cached square matrix is about 2896 x 2896 (libredgetpu).

## 2. Software stack and Google dependencies

| Piece | Role | Can we drop it? |
|---|---|---|
| `edgetpu_compiler` (x86-64 Linux only, last release 16.0.384591198) | Builds programs | Yes. This is the code generation project. We keep it only as an offline development oracle (a working copy lives in `~/projects/tpu/edgetpu_compiler.sh`) |
| TFLite / TensorFlow | Model input format, test-model generation | Yes, with a PyTorch (`torch.export`) front end |
| DarwiNN `executable.fbs` | Container our runner parses | Optional. It is only a schema (Apache 2.0) |
| `apex_latest_single_ep.bin` | 8051 USB firmware | Not realistically. No source; Apache 2.0 and redistributable, so vendoring it is a licence formality |
| libedgetpu | Google's runtime | Already replaced by reef-infer |

## 3. Model file format

A compiled model is a TFLite FlatBuffer with an `edgetpu-custom-op` operator
(`src/model.rs`). Its `custom_options` is a FlexBuffer map; key `"4"` holds a
DarwiNN `Package`, which holds a `MultiExecutable`, which holds serialized
`Executable`s.

- **Executable types.** `PARAMETER_CACHING` loads weights into wide memory
  once. `EXECUTION_ONLY` runs per inference. `STAND_ALONE` does both.
- **Executable contents.** Instruction bitstreams (chunks), field offsets
  (relocation bit positions for parameter, input, output and scratch base
  addresses, lower or upper 32 bits), parameters, input and output layers
  (size, zero point, dequantization factor, shape), DMA hints.
- **Zero points are in the uint8 domain.** A TFLite INT8 tensor with zp 0
  shows zp 128 (see `REVIEW.md` 1.3).
- Models with CPU ops besides the TPU op (for example SSD post-processing)
  load with `load_edgetpu_ops`; the runner still needs single-op models.

## 4. USB protocol

| Item | Value |
|---|---|
| Boot (DFU) ID | 1a6e:089a, until firmware upload |
| Application ID | 18d1:9302 |
| EP 0x01 OUT | Instructions, input, parameters. Single-endpoint mode prefixes an 8-byte header: u32 length, tag nibble |
| EP 0x81 IN | Output activations |
| EP 0x82 IN | 16-byte DMA-request events |
| EP 0x83 IN | Interrupts (phase completion) |
| Descriptor tags | 0 instructions, 1 input, 2 parameters, 3 output, 4-7 scalar-core interrupts 0-3 |

The device drives the exchange: the host sends the instruction bitstream,
then answers each event (tag 1: send input; tag 2: send parameters; tag 3:
read output) until an interrupt signals completion. The program itself emits
these requests as host DMA descriptors (4 words: address lo, address hi,
length, tag) pushed from scalar-core bundles.

Output activations arrive in tile order; each output layer's `OutputLayout`
(tile-id maps and per-tile offsets) says how to gather them into y-x-z order.
The runner does this since 2026-10-10 (before, multi-tile outputs came back
permuted).

Known runner gaps (from `REVIEW.md`): pending transfers grow per call, input
size is not validated, int8 outputs are read in the wrong domain by examples,
multi-output order follows arrival, fixed round caps. The runner also sends
the whole input on every tag-1 event, which breaks programs that request
input in pieces.

## 5. Instruction set

### 5.1 Encoding

- An instruction is 1-35 little-endian 128-bit words. Bit 0 is the lsb of
  byte 0 of word 0.
- Bits 6-11 of the first word are the opcode, which fixes the length.
- Common header: `gate` (bit 0), `pred_reg` (1-3), `pred_pol` (4). If `gate`
  is set, the instruction runs only when `p[pred_reg] == pred_pol`.
- Programs run `start ... halt, nop x4, end`. `start.length_bits` = 128 x
  (words between start and end). Verified on all 359 bitstreams.
- Every tile-dispatched instruction carries a sequence number.

### 5.2 Opcodes

All 23 opcodes seen in 359 compiled bitstreams decode fully
(`src/isa/`).

| Opcode | Name | Words | Role |
|---|---|---|---|
| 0x3e | start | 1 | Program header |
| 0x21 | halt | 1 | Terminator (code 4; 1 = continues in the next bitstream) |
| 0x3f | end | 1 | Final marker |
| 0x1a | sync | 1 | Tile fence, dispatched to the tiles |
| 0x24 | scsync | 2 | Scalar fence / scalar sync-counter op |
| 0x25 | pop | 2 | Host stream -> staging buffer (activations: 8 B units, 16384 units; parameters: 64 B units, 1024 units) |
| 0x26 | infeed | 4 | Staging buffer -> ring -> tiles in a bitmap |
| 0x27 | outfeed | 2 | Ring -> host, or ring -> scalar memory |
| 0x20 | scalar | 1 | Scalar-core bundle: ALU, loads/stores, special units, DMA descriptor pushes |
| 0x22 | branch | 1 | Relative branch while p0 (one form seen) |
| 0x23 | loop | 1 | Repeat the next `body` words `count` times |
| 0x01 | op | 17 | Tensor op (MAC, requant, conv, eltwise...) |
| 0x02 | op | 17 | MAX_POOL_2D |
| 0x03 | op | 17 | SUM/MEAN reduction |
| 0x10 | ringProducer | 4 | Memory -> ring, multicast to `dest` |
| 0x11, 0x12 | ringConsumer | 4 | Ring -> memory (consumer 0 / 1) |
| 0x13 | narrowToWide | 7 | Narrow -> wide memory or ring FIFO |
| 0x14 | wideToNarrow | 7 | Wide memory or ring FIFO -> narrow (mode 2: bias store) |
| 0x15-0x18 | mesh | 6 | Neighbour move south, west, north, east; can fill constants (edge padding). Convolution halo exchange decoded and reproduced (section 8) |
| 0x19 | nlu | 13 | Load the spline: 40 f32 coefficients, 7 breakpoints, mode |

Sync counters: 19 per tile (AVDATA, PARAMETERS, PARTIAL_SUMS, MESH in/out x4,
WIDE_TO_NARROW, WIDE_TO_SCALING, NARROW_TO_WIDE, RING_READ_A/B, RING_WRITE,
RING_PRODUCER_A/B) and 9 scalar (AVDATA_POP, PARAMETER_POP, AVDATA_INFEED,
PARAMETER_INFEED, SCALAR_INFEED, PRODUCER_A/B, RING_OUTFEED,
SCALAR_PIPELINE). Exact set/wait semantics are not fully known.

### 5.3 The tensor op (0x01)

17 words. Header and tile mask; 8 main loop counters; four 8-level address
generators (input read, output write, parameter read, partial-sum read; each
level is a signed increment, count - 1 and a 2-bit mode); a datapath block.

| Operation | dp_mode | reduce_mask | Notes |
|---|---|---|---|
| FC (compiler 14.1) | 3 | 0b101 | cfg0 0xA5; 64 outputs per pass |
| FC (compiler 16.0) | 1 | | cfg0 0x65, plus row fields (section 6) |
| RELU / RELU6 / QUANTIZE / ABS | 1 | 0 | Requantize |
| Copy / reformat | 4 | 0 | |
| Conv 3x3, split-K 1x1 | 4 | 0b1011 | Weight FIFO |
| Depthwise | 3 | 0b11 | Per-channel weights |
| Average pool | 4 | 0b11 | |
| ADD / SUB / MAX / MIN / MUL | 5 | 0b11 | MUL is two ops; ADD/SUB use 16-bit weights |

### 5.4 Numerics

Verified on hardware (tinycoral; stage order and saturation by us,
2026-10-10, bit-exact over all 256 inputs):

- uint8 x uint8 -> int32 (weights minus a per-channel zero point), + bias,
  x f32 multiplier, **clamp, then** round half to even, + output zero point,
  saturate to uint8.
- The multiplier is `in_scale * w_scale / out_scale`, computed as
  `F(F(in_scale * w_scale) * F(1/s_out))` in float32 (64/64 FC channels).
  Float64 reproduces the compiler's constants only ~60% of the time.
- Clamps are `[-out_zp, 255 - out_zp]`; fused RELU sets the minimum to 0.
  For uint8 FC the compiler moves both one ulp toward zero (130.99998), which
  is harmless because the clamp precedes rounding.
- LOGISTIC is a fixed 8-segment quartic spline (identical in every program).
- ADD/SUB weights are the best rational approximation of the scale ratio
  with terms <= 32767 (tinycoral for ADD; we confirmed the SUB encoding,
  section 6.1).
- The scalar core converts float to int rounding to nearest.

## 6. What we decoded (compiler 16.0 corpus)

Corpus: `tools/isa_corpus/gen.py` builds single-op models that vary one
parameter at a time (FC, conv, depthwise, elementwise, pooling, activations,
misc) and compiles them with 16.0. Together with 54 earlier models (including
MobileNet v2, SSD MobileNet v2, SSDLite): **359 bitstreams from 210 models,
about 41,500 instructions**, all decoding and re-encoding byte for byte.
Before we added our own fields, the Rust decoder agreed field for field with
tinycoral's on all 17,901 instructions of the 54 original models.

### 6.1 Fields

| Field | Bits | Meaning | Confidence |
|---|---|---|---|
| `wideToNarrow.tail_words` (signed) | 526-541 | Words in the innermost narrow row after the head: `n_lim0 + 1 - (head_words_m1 + 1)`, 0 without a head. tinycoral's `head_tail16` is bits 4-7 of it | Verified 1375/1375 |
| `outfeed.loop_en` | 214 | Loopback outfeed (section 6.3) | Verified 92/92 |
| `outfeed.loop_row` | 188-198 | Ring row written: starts at 2046, advances by `d1_limit + 1`, wraps from 2048 to 1024 | Strong 64/66 |
| `outfeed.loop_cfg` | 178-187 | 17 in 90/92 | Provisional |
| `infeed.loop_en` | 13 | Loopback infeed; with constants `loop_k168` = 1, `loop_k308` = 3, `loop_k315` = 3 | Verified 92/92 |
| `pop.buf_off` | 13-23 | Staging offset a parameter pop writes; the next parameter infeed reads the same `buf_off` | Verified 39/39; advance rule 34/37, wrap unclear |
| `op.w_zp_hi` | 1862-1869 | In SUB ops, `w_zp \| w_zp_hi << 8` is a 16-bit weight: numerator of `Fraction(s_b / s_a).limit_denominator(32767)` | Verified 4/4 models |
| `op.row_a`, `row_c` | 1886-1896, 1910-1920 | 16.0 only, in FC/conv/depthwise ops: wide-memory row of the per-channel weight zero points. `row_c = row_a` if `row_m = 1`, else `row_a - row_m` | **Verified on hardware** (FC: moving the zero points works iff these follow) |
| `op.row_m` | 1898-1899 | Unknown; 1 in FC | Provisional |
| `op.cfg4` | 2093 | Add bias (0 in no-bias FC; `cfg4 = 0` equals the zero-bias result on hardware) | **Verified on hardware** |
| `op.row_cfg` | 2094-2096 | 1 exactly when the row fields are used | Strong 1803/1805 |

`row_a` is 2047 for small convs and drops by the input width (W = 48, 64),
by `4k^2` for conv kernels k >= 6 and by `2k^2` for depthwise k >= 5. FC gives
1-3. It does not equal the op's parameter or partial-sum base row.

Bits that **hang the TPU when cleared** (2026-10-10): `wideToNarrow` bit 543
(set with `skip_en0`), `narrowToWide.tail_lvl`, `wideToNarrow.tail_en870` in
the FC input copy, and the caching `ringConsumer` bits 286-287 / 319. The FC
parameter blob and stream rules are in `CODEGEN_RESEARCH.md` and
`src/codegen/fc.rs`.

### 6.2 Program-level rules

Checked on every bitstream by `tests/isa_roundtrip.rs::check_program`:

- Each loopback outfeed pairs, in order, with an earlier loopback infeed with
  the same d0/d1/d2 limits.
- A parameter pop's `buf_off` equals the next parameter infeed's `buf_off`.
- `wideToNarrow.tail_words` follows its formula.

### 6.3 On-chip loopback

The compiler moves data from tiles into scalar memory and multicasts it back
to all 16 tiles without touching the host:

1. An infeed with `loop_en`, stream 0 (parameter path), `tiles = 0xffff`, is
   issued first and waits.
2. A ringProducer sends the data to the scalar core.
3. An outfeed with `loop_en` and the scalar-memory flag (bit 120) writes it to
   ring row `loop_row`.
4. The infeed delivers it to every tile.

In our corpus it carries **depthwise weights** every inference:
`C_out x (k^2 + 3)` bytes, rounded up to 64 B rows (32 channels, 3x3 -> 384
B; MobileNet's 960-channel layer -> 11,520 B).

tinycoral documents the outfeed half as undecoded and reports that a
scalar-memory infeed to all 16 tiles "delivers no data", cause unknown. Its
layout leaves `loop_en` and the three loopback constants at zero.

**On hardware** (2026-10-10, `examples/loopback_probe.rs`, compiled dwconv
C32 k3; the unmodified program is the reference and runs after a different
parameter blob, so stale data would show):

- The broadcast data is tile 0's: shifting the ringProducer's `addr` or
  sourcing tile 1 gives wrong output on every channel.
- Clearing infeed `loop_en`, `loop_k308` or `loop_k315`, or tinycoral's
  form (all four cleared), **hangs**. No variant returned stale data, so
  tinycoral's failure is not reproduced as such; the missing bits are
  required, but the failure mode differs.
- Infeed `loop_k168 = 0` runs but corrupts channels 4-11 and 16-31
  (0-3 and 12-15 stay exact).
- Infeed `tiles` other than `0xffff` (0x0001, 0x00ff) hangs.
- Outfeed `loop_en`, `loop_cfg` and bit 120 (`d3_stride` lsb) **change
  nothing**: the result is exact with each cleared. `k173` (1 here, 3 on
  host outfeeds) and `f222` (0x3f, host 0x7f) hang for any other value, as
  do ringProducer `to_c1 = 0` and `dest = 0`. So the loopback outfeed is
  identified by `k173` / `f222` / the producer, not by `loop_en`.
- `loop_row` does not select what the infeed reads (any working value gives
  the exact result) but must lie in [512 + rows, 1023] or [1536, 2047];
  everything else hangs. Measured on 3, 6, 24 and 26 rows. The compiler's
  2046 is inside; the gap below 512 + rows and in 1024-1535 is unexplained.
- The receiving ringConsumer needs the compiler's aux form: clearing
  `aux_en0` or `rsv286` hangs (`aux_en1` / `rsv319` only corrupt data),
  and no "plain" consumer combination ran, even with matching infeed
  constants. Its geometry tracks the kernel (`slots = k^2`, `cnt0 = k^2 - 1`,
  `cnt2 = C/4 - 1`, `rsv286 = 2`).

**Generated loopback** (`codegen::elementwise::build_broadcast`, checked by
`tests/codegen_hw.rs` and `examples/broadcast_hw.rs`): our own program sends
any tile's data through scalar memory to all 16 tiles, bit-exact on every
tile for sources 0, 3, 5, 6, 15. With the C8 3x3 consumer copied verbatim, the first
384 B of the source's 512 B arrive as 24 units of 16 B. Per 192 B, 3 units
go to the aux rows (0x1fe8 + 4 per unit) and 9 to the slots (0x1fa0 + 4 per
unit), so each wide address step of 4 holds 16 B here. Two program-level
rules came out of it:

- The epilogue waits on counter 6 for the loopback rows, and they count
  towards counter 5: `smask = wait = 0x60`, `thr5` = outfeeds + rows,
  `thr6` = rows, then `sync counters = 0x60000` (compiled C8: 18 / 2, C32:
  22 / 6). Without it the program hangs.
- The host producers' `r0_val` (id 0x51) indexes the rows sent to the scalar
  core, with the loopback rows first: `rows + t`, not `t` (compiled dwconv:
  6..=21).

### 6.4 Compiler 14.1 vs 16.0

- 16.0 FC ops use `cfg0 0x65`, `dp_mode 1` and set the row fields; 14.1 uses
  `cfg0 0xA5`, `dp_mode 3`, and none of its 3,400 FC ops set the row fields.
- The 14.1 op bits on a 16.0 parameter blob give saturated garbage on
  hardware, so the two forms also differ in parameter layout.
- So tinycoral's FC generator (14.1-exact) does not reproduce our 16.0
  models; the difference is structural (an extra transfer stage, different
  padding). Both are Google's output, so both should run.
- tinycoral's FC generator matches its own 14.1 table on all 355 entries
  (run here).

### 6.5 Correlated but not decoded

| Region | Observation |
|---|---|
| `narrowToWide` 589-756 | Four 42-bit sync-like records at 589 + 42k: 5-bit id at +4 (20, 4, 24), 2-bit level at +9, 19-bit value at +13 (2048, 4095), bit +40 set. Binary elementwise and MobileNet programs |
| `mesh` 249-254, 454-459 | Same offset as `ringConsumer.gstride` in each half. 34/50 for 3-level moves, 5/9 for 2-level moves; maybe 2 bits per level |
| `op` 1525 | Set in 152 of 157 copy-class ops (dp_mode 0, cfg0 0) |
| `infeed` 353-368, 391 | SSDLite streamed-parameter infeeds; low 6 bits always 0x25, upper bits grow through the program |
| `wideToNarrow` 543 | Equals `skip_en0` in 1514/1515 |
| `ringConsumer` 286-287, 319 | Accompany `aux_en0` / `aux_en1` (caching relocation) in the corpus, only 70-90% in MobileNet/SSD |

### 6.6 Still unknown

`op` 1946-1948 (activation ops, 5 values), `wideToNarrow` 490-493 (643-645 are
the top bits of `sync_val`, 2026-10-10),
single-instance bits in MobileNet programs, TTU mode bits per level, exact
sync counter semantics, `sync.b43`, `halt.code`. Most need targeted sweeps
(stride-2 depthwise, multi-input elementwise, streamed weights) or
perturbation on the device.

## 7. Beyond the official compiler

**Shown on hardware by tinycoral:**

- Transformer attention and whole layers on chip (softmax on the scalar core
  in float32). TinyStories-15M: 176.7 tok/s at batch 1, one program per
  token, 2.5x its matmul-only path.
- Batched matmul (the compiler refuses): 1,815 tok/s at batch 256.
- Tile-shifted placement that keeps 19 programs' weights resident at once.
- Freely fused programs (FFN: conv, conv, logistic, mul, mul, conv) and an
  on-chip argmax.

**Plausible from the ISA, untested:**

- Custom activations by loading our own spline coefficients (the compiler
  only loads logistic).
- Small control flow on the scalar core (top-k, post-processing, iteration).
- Activations surviving between programs (a KV cache on chip).
- Transposes and patch extraction as DMA loop nests.
- One-hop broadcast via the loopback (section 6.3): shared weights, scalar
  core in the loop, re-layout between layers.

## 8. Code generation status and plan

The model-by-model plan (which ops in which order) is in `ROADMAP.md`.

| Step | Status |
|---|---|
| Decoder / encoder in Rust (`src/isa/`) | Done; all bitstreams round-trip |
| Corpus tooling (`tools/isa_corpus/`, `examples/isa_check.rs`) | Done |
| Generators in Rust (`src/codegen/`): element-wise requant (1x16x16x32), FC (K <= 64, N <= 64) | Done; byte-identical to 16.0 (FC parameters included), bit-exact on device |
| FC K up to 1024 (input over 16 tiles, gathered over the ring), N 4-64 or 64 per tile up to 1024 | Done; byte-identical to 16.0 on 28 corpus models (parameters included), bit-exact on device |
| TFLite front end (`graph::Graph`): a DAG of CONV_2D, MAX_POOL_2D, ADD, global AVERAGE_POOL_2D (8x8x64 native, else lowered to an FC), FC, RESHAPE, int8 I/O; layers run through the host, which pads to the generators' sizes and crops; parameters cached once per layer at its own wide address | Done: MNIST MLP (97.35%), CNN (98.66%) and CIFAR-10 ResNet-8 (75.22%) on the TPU over 10000 images each, bit-exact with the host reference, same top-1 as TFLite CPU |
| Element-wise ADD (`codegen::add`): two input layers, requantized (integer weights from the scale ratio), fused RELU | Done; byte-identical to 16.0 on 8 corpus models, bit-exact on device (arbitrary scale ratios too) |
| Conv 1x1 Cin 32 -> Cout 64 (two passes of 16 channels), stride-2 Cout 64 | Done; byte-identical to 16.0 (parameters included), bit-exact on device |
| MAX_POOL_2D 2x2 stride 2 (`codegen::pool`): H, W 16 / 32, C 32 / 64 / 128 | Done; byte-identical to 16.0 on 5 corpus models, bit-exact on device |
| Global AVERAGE_POOL_2D 8x8x64 (`codegen::avg_pool`): mesh reduction into tile 0, no parameters, TFLite's rounding | Done; byte-identical to 16.0 (compiler form), bit-exact on device |
| Parameter placement (`build_at`, `Graph::build`): each layer's cached parameters at their own wide address | Done; coexistence verified on device (`examples/param_cache_hw.rs`) |
| Several layers in one call (`codegen::chain::splice`); conv and residual add chains with the activations (shortcuts included) in narrow memory (`chain::program`, `conv::Io`, `add::Io`): 16x16, 32x32, 32x32 stride 2 -> 16x16, 16x16 stride 2 -> native 8x8, ending in the global average pool and an FC on its output; `Graph::run` calls each chain once | Done; bit-exact on device (`examples/chain_hw.rs`, `examples/residual_hw.rs`; ResNet-8 in one call, 1.93 ms vs the compiler's 1.00 ms) |
| FC K 2048 / 3073-4096 (128 / 256 B per input tile); conv Cout 64 (Cin >= 32, 16x16) | Done; byte-identical to 16.0 (parameters included), bit-exact on device |
| On-chip loopback broadcast (`elementwise::build_broadcast`) | Done; bit-exact on device; landing layout is the compiler's depthwise one |
| CONV_2D (`codegen::conv`): same padding, stride 1 or 2, 16x16; k 1 (Cin 16, Cout 32) and k 3 (Cin 1, 3, 4, 16-128; Cout 8 / 16 / 32) | Done; byte-identical to 16.0 on 16 corpus models (parameters included), bit-exact on device |
| Conv width and height 16 / 32 / 64, Cin >= 16: superpixel columns, 4-row bands, derived without the compiler | Done; bit-exact on device up to 64x64 (not compared with 16.0) |
| Conv: Cout 64 for Cin 16, Cout 128+, Cin < 16 at larger sizes (stopgap: zero-weight channels), Cin 8, other k, valid padding, sizes not a multiple of 16 (stopgap: host padding) | Next |
| Scheduler/allocator for multi-layer graphs | Weeks to months |
| PyTorch front end (`torch.export`, int8) | After the above |

Our runner is compatible with generated programs: it is event-driven and
tinycoral programs need no relocation. Generated programs can be handed to it
as in-memory executables.

**Hardware experiments:**

1. ~~Perturb `row_a` / `row_m` / `row_c`~~: done on FC (2026-10-10); conv not yet.
2. ~~Clear the loopback bits on a compiled depthwise model~~: done
   (2026-10-10, section 6.3): hangs, no stale data.
3. ~~Minimal loopback: tile 0 -> scalar memory -> all tiles -> host~~: done
   (2026-10-10, section 6.3), bit-exact on all 16 tiles.
4. Load custom spline coefficients and compare with a host model.

The fastest end-to-end path for PyTorch today is PyTorch -> ONNX -> onnx2tf
-> int8 TFLite -> `edgetpu_compiler` -> reef-infer. It depends on Google's
compiler and gaps in the runner (shapes not exposed, int8 domain, output
order).

## 9. Tools

```bash
# Decode, round-trip and summarize models; --dump writes per-bitstream JSON
cargo run --example isa_check -- [--disasm] [--dump DIR] model_edgetpu.tflite...

# Hardware-free round-trip and rule checks
REEF_ISA_MODELS=~/projects/tpu:~/projects/pycoral-test:target/isa_corpus \
  cargo test --test isa_roundtrip

# Generated programs on the TPU vs host reference and the compiler
.venv/bin/python tools/isa_corpus/fc_ref.py   # FC weights for the parameter comparison
cargo run --features usb --example codegen_hw
cargo test --test codegen; cargo test --features usb --test codegen_hw

# Perturbation experiments on hardware (filter by name substring)
cargo run --features usb --example isa_probe [-- 'fc row']

# Loopback perturbations on a compiled dwconv (ROWS=a,b,... sweeps loop_row)
cargo run --release --features usb --example loopback_probe -- [model] [filter]

# Generated loopback broadcast, every tile checked (PARTS=hex bisects pieces)
cargo run --release --features usb --example broadcast_hw -- [src...]

# A compiled bitstream as codegen::Asm::emit calls (transcription start)
cargo run --example isa_emit -- model_edgetpu.tflite [executable]

# Generated conv vs every supported corpus model, field by field
.venv/bin/python tools/isa_corpus/conv_ref.py   # weights for the parameter comparison
cargo run --example conv_check -- [filter]

# Generated FC vs every supported corpus model, field by field
.venv/bin/python tools/isa_corpus/fc_ref.py   # weights for the parameter comparison
cargo run --example fc_check -- [filter]

# Generated element-wise add vs every supported corpus model, field by field
cargo run --example add_check -- [filter]

# CIFAR-10 ResNet-8 end to end (train + export, then host reference / TPU);
# --export-only redoes the int8 export from the saved resnet8.keras
.venv/bin/python tools/cifar10/train.py [--export-only]
cargo run --release --example eval -- --dir target/cifar10 --model resnet8 [--limit N] [--device]

# Speed / accuracy against edgetpu_compiler on the same .tflite and TPU
(cd target/cifar10 && ~/projects/tpu/edgetpu_compiler.sh -s resnet8.tflite)
cargo run --release --example bench_compare -- [--dir target/cifar10] [--model resnet8] [--limit N]

# Generated max pools and the global average pool vs the corpus, field by field
cargo run --example pool_check -- [filter]

# MNIST end to end (train + export, then host reference / TPU)
.venv/bin/python tools/mnist/train.py [--model mlp|cnn]
cargo run --release --example eval -- [--dir target/mnist] [--model mlp|cnn] [--limit N] [--device]
cargo run --release --example param_cache_hw   # layers' cached parameters at their own wide addresses coexist?
cargo run --release --example chain_hw   # several layers in one call, activations on chip
cargo run --release --example residual_hw   # ResNet-8 stages 1-2 (convs + adds) in one call

# Field diff of two compiled models
cargo run --example isa_diff -- a_edgetpu.tflite b_edgetpu.tflite

# Build the corpus with the 16.0 compiler
EDGETPU_COMPILER=~/projects/tpu/edgetpu_compiler.sh .venv/bin/python tools/isa_corpus/gen.py

# Reserved-bit regions, per-region tables, formula search
python3 tools/isa_corpus/analyze.py target/isa_corpus_dump [--region op:1886 --width 37 --fields w_zp] [--relate]
```

## 10. Sources

- tinycoral (MIT): https://github.com/FrederikRogalski/tinycoral. Our field
  layouts are ported from it (attribution in `NOTICE`).
- libredgetpu (Apache 2.0): https://github.com/ricardodeazambuja/libredgetpu
- edgetpuxray: https://github.com/geohot/edgetpuxray (licence unchecked)
- libedgetpu (Apache 2.0): https://github.com/google-coral/libedgetpu
- Patents: US9836691B1, US20180197068A1 (ISA), US20200005128A1 (parameter
  caching), US20190050717A1, GB2558980A, US20210373895A1.
