"""Export the weights of every corpus CONV_2D model, so the parameter blobs
`codegen::conv` builds can be compared with the compiler's.

    .venv/bin/python tools/isa_corpus/conv_ref.py [--corpus target/isa_corpus]

Writes `<model dir>/conv_ref.json`: weights ([Cout][kh][kw][Cin] int8,
row-major), w_scale, bias. Activation quantization comes from the compiled
model.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="target/isa_corpus")
args = ap.parse_args()

n = 0
for d in sorted((Path(args.corpus) / "conv").iterdir()):
    it = tf.lite.Interpreter(str(d / "model.tflite"))
    it.allocate_tensors()
    ts = it.get_tensor_details()
    w = next(
        t
        for t in ts
        if t["dtype"] == tf.int8.as_numpy_dtype
        and len(t["shape"]) == 4
        and t["name"].endswith("/convolution")
    )
    b = next(t for t in ts if t["dtype"] == tf.int32.as_numpy_dtype)
    ref = dict(
        weights=it.get_tensor(w["index"]).flatten().tolist(),
        w_scale=w["quantization_parameters"]["scales"].tolist(),
        bias=it.get_tensor(b["index"]).tolist(),
    )
    (d / "conv_ref.json").write_text(json.dumps(ref))
    n += 1
print(f"wrote {n} conv_ref.json")
