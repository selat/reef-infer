"""Generate int8-quantised .tflite test models (no TensorFlow required).

Usage:
    python tests/gen_models.py [output_dir]

Then compile for Edge TPU:
    edgetpu_compiler -s <model>.tflite
"""

from pathlib import Path

import numpy as np
import keras
import tensorflow as tf


def build_matmul_tflite(weights: np.ndarray) -> bytes:
    """Return raw .tflite bytes for: y = W @ x, int8 via Keras + full integer quantization.

    weights: Nx2 float matrix. Input is 2x1, output is Nx1.
    """

    n = weights.shape[0]
    assert weights.shape == (n, 2), f"expected Nx2 matrix, got {weights.shape}"

    inp = keras.Input(shape=(2,), name="input")
    dense = keras.layers.Dense(n, use_bias=False, name="matmul")
    out = dense(inp)
    model = keras.Model(inputs=inp, outputs=out)

    # Keras Dense stores weights as [input_dim, output_dim] = [2, N]
    dense.set_weights([weights.T.astype(np.float32)])

    def representative_dataset():
        for _ in range(200):
            yield [np.random.randint(-128, 128, size=(1, 2)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def make_dotproduct_keras_quant(output_dir: Path, dim: int = 8) -> None:
    """y = dot([1,2,...,dim], x) — int8 via Keras quantization."""
    weights = np.array(
        [
            [0, 0],
            [-128, 127],
            [-10, 10],
            [-1, 1],
        ],
        dtype=np.float32,
    )
    data = build_matmul_tflite(weights)
    out_path = output_dir / "dotproduct_keras_quant.tflite"
    out_path.write_bytes(data)
    print(f"wrote {out_path} ({len(data)} bytes)")


def make_matmul(output_dir: Path) -> None:
    """Generate Nx2 matmul model for precision testing.

    y = W @ x where W is 8x2, x is 2x1, output is 8x1.
    Then compile for Edge TPU:
        edgetpu_compiler -s matmul.tflite
    """
    weights = np.array(
        [
            [1, 0],  # identity on first element
            [0, 1],  # identity on second element
            [1, 1],  # addition
            [1, -1],  # subtraction
            [3, 7],  # scaling
            [-5, 10],  # mixed signs
            [50, -30],  # larger values
            [100, 100],  # large same direction
        ],
        dtype=np.float32,
    )
    data = build_matmul_tflite(weights)
    out_path = output_dir / "matmul.tflite"
    out_path.write_bytes(data)
    print(f"wrote {out_path} ({len(data)} bytes)")
    print_tflite_info(data, "matmul")


def build_conv2d_benchmark_tflite(
    input_size: int,
    channels: list[int],
    kernel_size: int,
) -> tuple[bytes, int]:
    """Build a multi-layer Conv2D model for TPU TOPS benchmarking.

    Architecture: chain of Conv2D(kernel_size x kernel_size) layers with ReLU,
    no pooling (keeps spatial dims via 'same' padding).

    Args:
        input_size: spatial H=W of the input feature map.
        channels: list of output channels per layer, e.g. [64, 64, 128, 128].
        kernel_size: convolution kernel size.

    Returns:
        (tflite_bytes, total_mac_ops)

    Op count per Conv2D layer (no bias):
        MACs = H_out * W_out * C_out * C_in * K * K
        OPs  = 2 * MACs   (one multiply + one accumulate per MAC)

    With 'same' padding and stride=1: H_out = H_in, W_out = W_in.
    """

    # Build the model
    inp = keras.Input(shape=(input_size, input_size, 3), name="input")
    x = inp
    prev_c = 3  # input channels (RGB)
    total_macs = 0

    for i, c_out in enumerate(channels):
        x = keras.layers.Conv2D(
            c_out,
            kernel_size,
            padding="same",
            use_bias=False,
            activation="relu",
            name=f"conv{i}",
        )(x)
        # MACs for this layer: H * W * C_out * C_in * K * K
        layer_macs = (
            input_size * input_size * c_out * prev_c * kernel_size * kernel_size
        )
        total_macs += layer_macs
        prev_c = c_out

    # Global average pooling to collapse spatial dims → [batch, C_out]
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    model = keras.Model(inputs=inp, outputs=x)

    total_ops = 2 * total_macs

    print(
        f"Conv2D benchmark: input={input_size}x{input_size}x3, "
        f"layers={channels}, kernel={kernel_size}x{kernel_size}"
    )
    print(f"  Total MACs:  {total_macs:,}")
    print(f"  Total OPs:   {total_ops:,}  ({total_ops / 1e9:.3f} G-OPs)")

    # Full int8 quantization
    def representative_dataset():
        for _ in range(100):
            yield [
                np.random.randint(0, 256, size=(1, input_size, input_size, 3)).astype(
                    np.float32
                )
            ]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert(), total_ops


def make_conv2d_benchmark(output_dir: Path) -> None:
    """Generate a multi-layer Conv2D model for TPU TOPS estimation.

    Architecture: 4 Conv2D layers on 56x56x3 input.
      conv0: 56x56x3  -> 56x56x64,   9x9 kernel  →  MACs = 56*56*64*3*9*9     =  48,771,072
      conv1-5: 56x56x64 -> 56x56x64,   9x9 kernel  →  MACs = 56*56*64*64*9*9    = 1,040,449,536
      GAP:   56x56x64-> 64
      ──────────────────────────────────────────────────────────
      Total MACs:   1,040,449,536 * 6 + 48,771,072 = 1,290,670,848
      Total OPs:  14,663,835,648  (~14.66 G-OPs)

    TOPS = total_ops / inference_time_seconds / 1e12

    Then compile for Edge TPU:
        edgetpu_compiler -s conv2d_bench.tflite
    """
    data, total_ops = build_conv2d_benchmark_tflite(
        input_size=56,
        channels=[64, 64, 64, 64, 64, 64],
        kernel_size=9,
    )
    out_path = output_dir / "conv2d_bench.tflite"
    out_path.write_bytes(data)
    print(f"wrote {out_path} ({len(data)} bytes)")
    print("To estimate TOPS: run inference N times, measure total time T seconds")
    print(f"  TOPS = N * {total_ops} / T / 1e12")
    print_tflite_info(data, "conv2d_bench")


def print_tflite_info(data: bytes, label: str) -> None:
    """Print tensor and operator dtype info from a .tflite flatbuffer."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()

    print(f"\n── {label} ──")
    print("Tensors:")
    for t in interp.get_tensor_details():
        print(
            f"  [{t['index']}] {t['name']:20s}  shape={t['shape']}  dtype={t['dtype'].__name__}  quant={t['quantization_parameters']}"
        )

    print("Inputs:")
    for t in interp.get_input_details():
        print(f"  [{t['index']}] {t['name']:20s}  dtype={t['dtype'].__name__}")

    print("Outputs:")
    for t in interp.get_output_details():
        print(f"  [{t['index']}] {t['name']:20s}  dtype={t['dtype'].__name__}")


def run_tflite_model(data: bytes, input_vec: np.ndarray, label: str) -> None:
    """Run a .tflite model on a single input vector and print the output."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()

    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    x = input_vec.astype(inp_detail["dtype"]).reshape(inp_detail["shape"])
    interp.set_tensor(inp_detail["index"], x)
    interp.invoke()
    y = interp.get_tensor(out_detail["index"])
    print(f"  {label}: input={x.flatten()} -> output={y.flatten()}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_matmul(output_dir)
    make_conv2d_benchmark(output_dir)
