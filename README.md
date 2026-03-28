# reef-infer

A pure-Rust, open-source runtime for the [Google Coral Edge TPU](https://coral.ai/products/accelerator) USB Accelerator. Communicates directly with the device via userspace USB, handling firmware upload, chip initialization, weight loading, and inference.

Also provides **Python bindings** via PyO3/maturin.

## Rust Library & CLI

Build the library and CLI binary:

```bash
cargo build --release
```

Run inference from the command line:

```bash
./target/release/reef-infer model_edgetpu.tflite --input input.bin --output output.bin
```

Run a latency benchmark:

```bash
./target/release/reef-infer model_edgetpu.tflite --bench 1000
```

## Python Bindings

Install into the current virtualenv for development:

```bash
maturin develop --features extension-module
```

Build a distributable wheel:

```bash
maturin build --features extension-module
```

Usage:

```python
import reef_infer

model = reef_infer.load_model("model_edgetpu.tflite")
device = reef_infer.open_device()
loaded = device.load_params(model)
output = device.run_inference(loaded, input_bytes)
```

See `examples/` for more — image classification, benchmarking, and a live webcam demo.

## Integration Tests

### Generating test models

The test models are int8-quantized TFLite models created with TensorFlow/Keras, then compiled for the Edge TPU.

Generate the base models:

```bash
python tests/gen_models.py
```

This outputs two models into `tests/models/`:
- **matmul.tflite** — small matrix multiply for precision testing
- **conv2d_bench.tflite** — 6-layer Conv2D network for throughput benchmarking

Compile them for the Edge TPU:

```bash
edgetpu_compiler -s tests/models/matmul.tflite
edgetpu_compiler -s tests/models/conv2d_bench.tflite
```

### Running tests

Tests require a physical Edge TPU plugged in:

```bash
cargo test --features usb
```
