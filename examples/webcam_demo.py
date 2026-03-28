"""
Webcam demo: run MobileNet on live camera frames and print top-5 classes.

Usage:
    python examples/webcam_demo.py <model_edgetpu.tflite> <labels.txt> [--camera N]

Example:
    python examples/webcam_demo.py \\
        tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite \\
        tmp/labels_mobilenet_quant_v1_224.txt \\
        --camera 0

Controls:
    q / Esc               Quit
"""

import sys
import time

import cv2
import numpy as np
import reef_infer


# ── helpers ───────────────────────────────────────────────────────────────────


def load_labels(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f]


def preprocess(frame_bgr: np.ndarray, size: int) -> bytes:
    """Resize to (size, size), convert BGR→RGB, return raw uint8 bytes."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8).tobytes()


def dequantize(raw: bytes, size: int, scale: float, zero_point: int) -> np.ndarray:
    """Convert raw uint8 output to float scores."""
    arr = np.frombuffer(raw[:size], dtype=np.uint8).astype(np.float32)
    return (arr - zero_point) * scale


def top_k(scores: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


def clear_lines(n: int) -> None:
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")  # move up + clear line


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "Usage: python webcam_demo.py <model_edgetpu.tflite> <labels.txt> [--camera N]"
        )
        sys.exit(1)

    model_path = args[0]
    labels_path = args[1]
    camera_idx = 0

    i = 2
    while i < len(args):
        if args[i] == "--camera" and i + 1 < len(args):
            camera_idx = int(args[i + 1])
            i += 2
        else:
            i += 1

    labels = load_labels(labels_path)

    # ── load model + device ───────────────────────────────────────────────
    model = reef_infer.load_model(model_path)
    print(model)

    channels = 3
    spatial = int((model.input_size // channels) ** 0.5)
    print(f"Input: {spatial}x{spatial}x{channels} ({model.input_size} bytes)")

    print("Opening Edge TPU device...")
    dev = reef_infer.open_device()

    print("Loading parameters...")
    params = dev.load_params(model)

    # ── open camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {camera_idx}")
        sys.exit(1)

    print(f"\nRunning on camera {camera_idx} — press q to quit\n")

    TOP_K = 5
    prev_time = time.perf_counter()
    printed_lines = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break

        # ── inference ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        input_bytes = preprocess(frame, spatial)
        output_bytes = dev.run_inference(params, input_bytes)
        inference_ms = (time.perf_counter() - t0) * 1e3

        fps = 1.0 / (time.perf_counter() - prev_time)
        prev_time = time.perf_counter()

        scores = dequantize(
            output_bytes, model.output_size, model.output_scale, model.output_zero_point
        )
        results = top_k(scores, TOP_K)

        # ── print top-5, overwriting previous output ───────────────────────
        if printed_lines:
            clear_lines(printed_lines)

        lines = []
        lines.append(f"FPS: {fps:5.1f}   inference: {inference_ms:.1f} ms")
        lines.append("─" * 40)
        for rank, (cls_idx, score) in enumerate(results, 1):
            label = labels[cls_idx] if cls_idx < len(labels) else f"class {cls_idx}"
            bar = "█" * int(max(0.0, score) * 20)
            lines.append(f"  {rank}. {label:<30s}  {score:+.3f}  {bar}")
        print("\n".join(lines))
        printed_lines = len(lines)

        # ── key handling (non-blocking) ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
