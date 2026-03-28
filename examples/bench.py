"""
Example: benchmark inference latency on an Edge TPU model.

Usage:
    python examples/bench.py <model_edgetpu.tflite> [--reps N] [--warmup N]
"""

import sys
import time
import reef_infer


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python bench.py <model_edgetpu.tflite> [--reps N] [--warmup N]")
        sys.exit(1)

    model_path = args[0]
    reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 50
    warmup = int(args[args.index("--warmup") + 1]) if "--warmup" in args else 3

    model = reef_infer.load_model(model_path)
    print(model)

    input_bytes = bytes(model.input_size)

    dev = reef_infer.open_device()
    params = dev.load_params(model)

    print(f"Warmup ({warmup} reps)...")
    for _ in range(warmup):
        dev.run_inference(params, input_bytes)

    print(f"Benchmarking ({reps} reps)...")
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        dev.run_inference(params, input_bytes)
        times.append(time.perf_counter() - t0)
        print(f"  rep {i + 1:>3}/{reps}: {times[-1] * 1e3:.2f} ms", end="\r")

    times_ms = [t * 1e3 for t in times]
    print(
        f"\nmin {min(times_ms):.2f} ms  "
        f"avg {sum(times_ms) / len(times_ms):.2f} ms  "
        f"max {max(times_ms):.2f} ms"
    )


if __name__ == "__main__":
    main()
