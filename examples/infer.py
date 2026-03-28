"""
Example: run a single inference on an Edge TPU model.

Usage:
    python examples/infer.py <model_edgetpu.tflite> [input.bin]

If no input file is given, a zero-filled buffer of the right size is used.
"""

import sys
import numpy as np
import reef_infer


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python infer.py <model_edgetpu.tflite> [input.bin]")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2] if len(sys.argv) > 2 else None

    # ── Load model ──────────────────────────────────────────────────────────
    model = reef_infer.load_model(model_path)
    print(model)

    # ── Prepare input ───────────────────────────────────────────────────────
    if input_path:
        with open(input_path, "rb") as f:
            input_bytes = f.read()
        print(f"Loaded {len(input_bytes)} bytes from {input_path}")
    else:
        # All-zeros int8 input — replace with real data for meaningful results.
        input_bytes = bytes(model.input_size)
        print(f"Using {model.input_size} zero bytes as input")

    # ── Open device and load weights ─────────────────────────────────────────
    print("Opening Edge TPU device...")
    dev = reef_infer.open_device()

    print("Loading parameters onto chip (one-time cost)...")
    params = dev.load_params(model)

    # ── Run inference ────────────────────────────────────────────────────────
    print("Running inference...")
    output = dev.run_inference(params, input_bytes)

    # ── Decode output ────────────────────────────────────────────────────────
    # The output is raw int8 bytes. Convert to a numpy array for easy use.
    out_array = np.frombuffer(output, dtype=np.int8)
    top_class = int(np.argmax(out_array))
    print(f"Output: {len(output)} bytes")
    print(f"Top class index: {top_class}  (raw score: {out_array[top_class]})")


if __name__ == "__main__":
    main()
