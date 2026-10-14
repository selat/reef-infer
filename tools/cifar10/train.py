"""Train and int8-quantize ResNet-8 on CIFAR-10 (the MLPerf Tiny image
classification model), and export the test set with TFLite CPU outputs as
the reference for `examples/cifar10.rs`.

    .venv/bin/python tools/cifar10/train.py [--out target/cifar10] [--epochs 20]

Model (MLPerf Tiny `resnet_v1_eembc`, ending at the logits instead of a
softmax): Conv 3x3 16 -> 3 residual stacks of 16 / 32 / 64 filters (the
second and third downsample with stride 2 and a 1x1 stride-2 shortcut) ->
AveragePooling 8x8 -> Dense 10. Batch norm folds into the convs.

Writes to `--out`:
- `resnet8.tflite`: the int8 model (int8 input / output).
- `test_images.bin`: 10000 x 32 x 32 x 3 uint8 pixels; `test_labels.bin`.
- `resnet8_tflite_logits.bin`: 10000 x 10 int8 logits from TFLite.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="target/cifar10")
ap.add_argument("--epochs", type=int, default=20)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train[:, 0], y_test[:, 0]
keras = tf.keras
L = keras.layers


def block(x, filters, stride):
    reg = keras.regularizers.l2(1e-4)
    y = L.Conv2D(filters, 3, strides=stride, padding="same", kernel_regularizer=reg)(x)
    y = L.BatchNormalization()(y)
    y = L.Activation("relu")(y)
    y = L.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(y)
    y = L.BatchNormalization()(y)
    if stride != 1:
        x = L.Conv2D(
            filters, 1, strides=stride, padding="same", kernel_regularizer=reg
        )(x)
    return L.Activation("relu")(L.Add()([x, y]))


inp = keras.Input((32, 32, 3))
x = L.Conv2D(16, 3, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
x = L.BatchNormalization()(x)
x = L.Activation("relu")(x)
x = block(x, 16, 1)
x = block(x, 32, 2)
x = block(x, 64, 2)
x = L.AveragePooling2D(8)(x)
x = L.Flatten()(x)
model = keras.Model(inp, L.Dense(10)(x))
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

x_tr = x_train.astype("float32") / 255.0
x_te = x_test.astype("float32") / 255.0
aug = keras.Sequential([L.RandomFlip("horizontal"), L.RandomTranslation(0.1, 0.1)])
ds = (
    tf.data.Dataset.from_tensor_slices((x_tr, y_train))
    .shuffle(50000, seed=0)
    .batch(64)
    .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
model.fit(ds, epochs=a.epochs, verbose=2)
_, acc = model.evaluate(x_te, y_test, verbose=0)
print(f"float accuracy: {acc:.4f}")


def rep():
    for i in range(0, 5000, 25):
        yield [x_tr[i : i + 1]]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
(out / "resnet8.tflite").write_bytes(tfl)

it = tf.lite.Interpreter(model_content=tfl)
it.allocate_tensors()
inp_d, out_d = it.get_input_details()[0], it.get_output_details()[0]
s, zp = inp_d["quantization"]
logits = np.zeros((len(x_test), 10), np.int8)
for i, img in enumerate(x_te):
    q = np.clip(np.round(img / s + zp), -128, 127)
    it.set_tensor(inp_d["index"], q.astype(np.int8)[None])
    it.invoke()
    logits[i] = it.get_tensor(out_d["index"])[0]
print(f"int8 TFLite accuracy: {(logits.argmax(1) == y_test).mean():.4f}")
print(f"input quantization: scale {s}, zero point {zp}")
x_test.astype(np.uint8).tofile(out / "test_images.bin")
y_test.astype(np.uint8).tofile(out / "test_labels.bin")
logits.tofile(out / "resnet8_tflite_logits.bin")
