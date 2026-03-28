"""
Classify an image using a quantized MobileNet model on the Edge TPU.

Usage:
    python examples/classify.py <model_edgetpu.tflite> <labels.txt> <image>

Example:
    python examples/classify.py \
        tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
        tmp/labels_mobilenet_quant_v1_224.txt \
        photo.jpg
"""

import sys

import numpy as np
from PIL import Image

import reef_infer


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python classify.py <model_edgetpu.tflite> <labels.txt> <image>")
        sys.exit(1)

    model_path, labels_path, image_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # ── Load labels ─────────────────────────────────────────────────────────
    with open(labels_path) as f:
        labels = [line.strip() for line in f]

    # ── Load model ──────────────────────────────────────────────────────────
    model = reef_infer.load_model(model_path)
    print(model)

    # Infer spatial dims from input size: input_size = H * W * C (uint8)
    # MobileNet v1 expects 224x224x3
    input_bytes = model.input_size
    channels = 3
    spatial = int((input_bytes // channels) ** 0.5)
    print(f"Input: {spatial}x{spatial}x{channels} ({input_bytes} bytes)")

    # ── Preprocess image ────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB").resize((spatial, spatial))
    img_array = np.asarray(img, dtype=np.uint8)  # shape [H, W, 3], range [0, 255]

    # MobileNet quant models typically use uint8 input with zp=0, scale~0.00784
    # The raw pixel values [0,255] map directly to the quantized input range.
    input_data = img_array.tobytes()

    # ── Run inference ───────────────────────────────────────────────────────
    print("Opening Edge TPU device...")
    dev = reef_infer.open_device()

    print("Loading parameters...")
    params = dev.load_params(model)

    print("Running inference...")
    output = dev.run_inference(params, input_data)

    # ── Dequantize and decode ───────────────────────────────────────────────
    raw = np.frombuffer(output[: model.output_size], dtype=np.uint8)
    scores = (raw.astype(np.float32) - model.output_zero_point) * model.output_scale

    top5_idx = np.argsort(scores)[::-1][:5]
    print("\nTop 5 predictions:")
    for rank, idx in enumerate(top5_idx, 1):
        label = labels[idx] if idx < len(labels) else f"class {idx}"
        print(f"  {rank}. {label:40s}  score={scores[idx]:.4f}  (raw={raw[idx]})")


if __name__ == "__main__":
    main()
