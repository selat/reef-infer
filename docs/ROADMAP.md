# Roadmap: from single ops to whole models without `edgetpu_compiler`

Status as of 2026-10-14 and the order to work in. Each milestone is a model
family that becomes runnable end to end; each item lists what it needs,
what we already know, what is unknown, and how it gets verified.
`EDGE_TPU.md` section 8 has the per-generator status, `CODEGEN_RESEARCH.md`
the findings.

## Where we are

| Generator | Coverage | Verified |
|---|---|---|
| Element-wise requant (`elementwise`) | 1x16x16x32 | byte-identical, on device |
| FC (`fc`) | K <= 2048 or 3073-4096 (multiple of 8, input spread over tiles and gathered), N 4-64 or 64 per tile up to 1024 | byte-identical (31 corpus models), on device |
| MAX_POOL_2D (`pool`) | 2x2 stride 2 valid; H, W 16 / 32; C 32 / 64 / 128 | byte-identical (5 corpus models), on device |
| Front end (`graph`) | CPU int8 `.tflite`: CONV_2D, MAX_POOL_2D, FC, RESHAPE; host pads / crops between layers | MNIST MLP (97.35%) and CNN (98.66%) on device, bit-exact with the host reference |
| CONV_2D (`conv`) | same padding; k 1 / 3; stride 1 / 2; 16x16 with Cin 1-128 (k 3), up to 64x64 with Cin >= 16; Cout 8-32 | byte-identical at 16x16, on device up to 64x64 |
| Loopback broadcast | tile -> scalar memory -> all tiles | on device |

What carries over to every new op: the instruction encoder, the walk rule
(increment = stride - span of inner loops), the tile/patch model with halo
exchange, the band loader, weight multicast, partial sums, the output path,
the clamp formula, and the hardware bisection workflow (`build_unchecked`,
host references, deliberately broken references as negative checks).

## How each item gets verified

1. Host reference in Rust, bit-exact on the TPU (always).
2. Byte equality with the compiler where the corpus has the case (learning
   tool; not required once a part is understood).
3. Per milestone: the whole model against TFLite's CPU interpreter on a test
   set (`.venv`), bit-exact per layer and top-1 accuracy end to end.

Sizes: S = a day or less, M = a few days, L = a week or more.

## Milestone 0: plumbing (needed by every model)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| TFLite front end | read the CPU `.tflite` (schema in `schema_v3_generated.rs`): tensors, weights, per-channel scales, zero points, op list | **done** for FC + RESHAPE (`graph::Graph`); other ops map as their generators arrive | M |
| int8 <-> uint8 | TFLite int8 I/O vs our uint8: +128 on input, weights already stored as `W + 128` | **done** (`graph::i8_to_u8`) | S |
| Host-side ops | flatten / reshape (NHWC, free), argmax, softmax, dequantize | reshape done; the rest when a model needs them | S |
| Layer chaining | run layer programs in sequence, output of one as input of the next (through the host first) | runner can do it; **unknown: whether two models' PARAMETER_CACHING blobs overwrite each other in tile 0's wide memory** **answered: they overwrite** (`examples/param_cache_hw.rs`); `Graph::run` reloads each layer's parameters per inference until layers get separate placements | S + experiment |
| Model runner API | `Model::from_tflite(path)` -> list of programs + host glue, one `run(input) -> output` | **done** as `Graph::from_tflite` / `build` / `run` | S |

## Milestone 1: MNIST MLP (Flatten -> Dense 128 ReLU -> Dense 10)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| **FC, large K** | K = 784, 128 | **done**: no passes; the compiler spreads the input 64 B per tile and gathers it over the ring into the compute tiles (byte-identical, K up to 1024) | M |
| **FC, large N** | N = 128 | **done**: 64 channels per compute tile, one parameter blob each | M |
| FC, small N | N = 10 | **done**: padded to 16 as the compiler does (live channels to a multiple of 4) | S |
| End-to-end test | MNIST test set vs TFLite CPU | `tools/mnist/train.py` + `examples/mnist.rs`: **done**: TPU 97.35% on all 10000, bit-exact with the host reference, same top-1 as TFLite CPU | S |

First model with no compiler involved anywhere.

## Milestone 2: MNIST CNN (Conv 3x3 32 -> MaxPool -> Conv 3x3 64 -> MaxPool -> Dense)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
**Done (2026-10-14):** all 10000 test images on the TPU, 98.66%,
bit-exact with the host reference, same top-1 as TFLite CPU.

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| **MaxPool 2x2 stride 2** | after each conv | **done** (`codegen::pool`): the element-wise loader, a regrouping into wide words, and a pool op (opcode 0x02); byte-identical on 5 corpus models | M |
| Cin 1 on larger images | first layer, 28x28x1 | **stopgap**: fed as Cin 16 with zero weights (the verified Cin 16 path at 32x32); the sub-word input form at larger sizes is still open | M |
| Sizes not a multiple of 16 | 28x28, 14x14, 7x7 | **stopgap done**: host pads with the zero point and crops (`graph::Step`); proper: tiles with unequal patches | S stopgap / L proper |
| Conv Cout 64 at k 3 | second conv | **done** for Cin >= 32 at 16x16 (byte-identical); Cin 16 moves the compiler's narrow buffers | S-M |
| FC K = 3136 | after flatten | **done**: no passes either; 128 / 256 B per input tile above K 1024 (K 2049..3072 not derived) | S |

## Milestone 3: small CNNs (CIFAR-10 style, ResNet-8 / -20)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| **Element-wise add (two inputs)** | residual connections | second input stream; corpus has add / sub / mul / min / max | M |
| Conv Cout 64-256 | wider layers | compiler uses `loop3` 64-channel blocks above 64 | M |
| Conv Cin 8 | odd widths in small nets | the 18-weight-row scheme seen in the corpus | S-M |
| Valid padding | many Keras models | corpus models use 12-38 TensorOps (edge tiles special-cased); our own: no halo fill, smaller output | M |
| Global average pool / MEAN | classifier heads | corpus `misc_opmean`, `pool_..._opavg` | M |
| Output > 2 KiB per tile | larger feature maps | wide staging above 0x207f untested; or produce in chunks | S-M |
| Images > 64x64 (32x32 done) | 96-128 px inputs | patch memory per tile grows; needs a capacity model (experiment: largest patch that runs) | M |

## Milestone 4: MobileNet V1 / V2 (the Edge TPU's home ground)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| **Depthwise conv 3x3, stride 1 / 2** | every block | weights through the on-chip loopback (decoded and reproduced), consumer aux / slot layout known; TensorOp mode new; corpus has 20 dwconv models | L |
| 1x1 conv, any Cin / Cout | pointwise layers | generalize the k 1 scheme beyond Cin 16 / Cout 32 | M |
| ReLU6 | V1 / V2 activations | a clamp; already supported by the output stage | S |
| 224x224 inputs | real resolution | spatial tiling in several passes per layer (patch does not fit), streamed weights (corpus SSDLite has streamed-parameter infeeds) | L |
| **Keep activations on chip between layers** | performance (host round-trips per layer are slow) | needs a memory planner: layer i's output layout = layer i+1's input layout, no host hop; the scheduler / allocator from section 8 | L |
| Concat, pad | V2 / Inception-style graphs | corpus `misc_opconcat`, `misc_oppad`; often a DMA re-layout | M |

## Milestone 5: detection and beyond (SSD-MobileNet, EfficientNet-Lite)

| Item | Need | Known / unknown | Size |
|---|---|---|---|
| Logistic, tanh, hard-swish | activations | the compiler loads spline coefficients; hardware experiment 4 (custom splines) is the open step | M-L |
| Softmax on chip | heads (optional; host can do it) | tinycoral ran softmax on the scalar core in float32 | M |
| Resize (nearest / bilinear) | FPN-style heads | corpus `misc_opresize_*` | M |
| Reshape / transpose | general graphs | DMA loop nests; corpus `misc_opreshape`, `misc_optranspose` | M |
| SSD post-processing | box decode, NMS | on the host (as libedgetpu does) | S |
| PyTorch front end | `torch.export` int8 -> our op list | after the TFLite front end proves the mapping | L |

## Cross-cutting, in order of when they start to matter

1. **Parameter cache placement** (milestone 0 experiment): needed as soon
   as two layers exist.
2. **Memory capacity model** (milestone 3): narrow / wide / partial-sum
   sizes per tile, from hardware experiments (largest patch, largest
   partial-sum area, wide addresses above 0x207f). Turns size checks from
   "verified list" into "fits".
3. **Planner instead of per-shape code** (milestone 3-4): choose loop
   scheme and buffer placement from the shape and the capacity model;
   generators become emitters for a plan.
4. **On-chip chaining** (milestone 4): the biggest performance step;
   depends on 2 and 3.

## Not planned

- Byte equality with the compiler for new shapes: kept as a learning tool
  where the corpus has the case, not a goal.
- Training, float models, or ops the Edge TPU does not accelerate (they
  stay on the host, as with libedgetpu).
