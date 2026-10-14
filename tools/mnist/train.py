"""Train and int8-quantize an MNIST model, and export the test set with
TFLite CPU outputs as the reference for `examples/eval.rs`.

- `mlp`: Flatten -> Dense 128 ReLU -> Dense 10.
- `cnn`: Conv 3x3 32 ReLU -> MaxPool 2 -> Conv 3x3 64 ReLU -> MaxPool 2 ->
  Flatten -> Dense 10 (same padding).

    .venv/bin/python tools/mnist/train.py [--model mlp|cnn] [--out target/mnist] [--epochs 5]

Writes to `--out`:
- `<model>.tflite`: the int8 model (int8 input / output, no softmax).
- `test_images.bin`: 10000 x 784 uint8 pixels; `test_labels.bin`: 10000 labels.
- `<model>_tflite_logits.bin`: 10000 x 10 int8 logits from the TFLite
  interpreter.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="target/mnist")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
keras = tf.keras
L = keras.layers
if a.model == "mlp":
    layers = [keras.Input((28, 28)), L.Flatten(), L.Dense(128, activation="relu")]
else:
    x_train = x_train[..., None]
    layers = [
        keras.Input((28, 28, 1)),
        L.Conv2D(32, 3, padding="same", activation="relu"),
        L.MaxPooling2D(2),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.MaxPooling2D(2),
        L.Flatten(),
    ]
model = keras.Sequential(layers + [L.Dense(10)])
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=a.epochs, batch_size=128, verbose=2)
x_eval = x_test.astype("float32") / 255.0
if a.model == "cnn":
    x_eval = x_eval[..., None]
_, acc = model.evaluate(x_eval, y_test, verbose=0)
print(f"float accuracy: {acc:.4f}")


def rep():
    for i in range(0, 2000, 10):
        yield [x_train[i : i + 1]]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
(out / f"{a.model}.tflite").write_bytes(tfl)

it = tf.lite.Interpreter(model_content=tfl)
it.allocate_tensors()
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
s, zp = inp["quantization"]
logits = np.zeros((len(x_test), 10), np.int8)
for i, img in enumerate(x_test):
    q = np.clip(np.round(img.astype("float32") / 255.0 / s + zp), -128, 127)
    it.set_tensor(inp["index"], q.astype(np.int8).reshape(inp["shape"]))
    it.invoke()
    logits[i] = it.get_tensor(outp["index"])[0]
print(f"int8 TFLite accuracy: {(logits.argmax(1) == y_test).mean():.4f}")
print(f"input quantization: scale {s}, zero point {zp}")
x_test.astype(np.uint8).tofile(out / "test_images.bin")
y_test.astype(np.uint8).tofile(out / "test_labels.bin")
logits.tofile(out / f"{a.model}_tflite_logits.bin")
