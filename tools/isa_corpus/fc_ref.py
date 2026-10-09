"""Export the weights of the corpus FC model that `examples/codegen_hw.rs`
reproduces byte for byte (parameters included).

    .venv/bin/python tools/isa_corpus/fc_ref.py [--corpus target/isa_corpus]

Writes `<corpus>/fc_ref.json`: k, n, weights ([N][K] int8, row-major),
w_scale, bias, in_scale, in_zp, out_scale, out_zp.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="target/isa_corpus")
args = ap.parse_args()
model = (
    Path(args.corpus)
    / "fc/fc_K64_N64_actnone_biasTrue_in_range4p0_iouint8/model.tflite"
)

it = tf.lite.Interpreter(str(model))
it.allocate_tensors()
by_name = {t["name"]: t for t in it.get_tensor_details()}


def find(pred):
    return next(t for t in by_name.values() if pred(t))


inp = it.get_input_details()[0]
out = it.get_output_details()[0]
w = find(lambda t: t["name"].endswith("/MatMul"))
b = find(lambda t: t["name"].endswith("/BiasAdd"))
n, k = w["shape"]
ref = dict(
    k=int(k),
    n=int(n),
    weights=it.get_tensor(w["index"]).flatten().tolist(),
    w_scale=w["quantization_parameters"]["scales"].tolist(),
    bias=it.get_tensor(b["index"]).tolist(),
    in_scale=float(inp["quantization"][0]),
    in_zp=int(inp["quantization"][1]),
    out_scale=float(out["quantization"][0]),
    out_zp=int(out["quantization"][1]),
)
dst = Path(args.corpus) / "fc_ref.json"
dst.write_text(json.dumps(ref))
print(f"wrote {dst}")
