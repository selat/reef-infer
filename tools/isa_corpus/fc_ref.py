"""Export the weights of the corpus FC models, so the parameter blobs
`codegen::fc` builds can be compared with the compiler's.

    .venv/bin/python tools/isa_corpus/fc_ref.py [--corpus target/isa_corpus]

Writes `<model dir>/fc_ref.json` for every FC model with a bias, and
`<corpus>/fc_ref.json` for the model `examples/codegen_hw.rs` reproduces
byte for byte (parameters included): k, n, weights ([N][K] int8,
row-major), w_scale, bias, in_scale, in_zp, out_scale, out_zp.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="target/isa_corpus")
args = ap.parse_args()


def export(model):
    it = tf.lite.Interpreter(str(model))
    it.allocate_tensors()
    ts = it.get_tensor_details()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    w = next(
        t
        for t in ts
        if t["dtype"] == tf.int8.as_numpy_dtype
        and len(t["shape"]) == 2
        and t["index"] != inp["index"]
        and t["index"] != out["index"]
    )
    b = next(t for t in ts if t["dtype"] == tf.int32.as_numpy_dtype)
    n, k = w["shape"]
    return dict(
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


corpus = Path(args.corpus)
n = 0
for d in sorted((corpus / "fc").iterdir()):
    if "biasTrue" not in d.name:
        continue
    (d / "fc_ref.json").write_text(json.dumps(export(d / "model.tflite")))
    n += 1
print(f"wrote {n} fc_ref.json")
ref = corpus / "fc/fc_K64_N64_actnone_biasTrue_in_range4p0_iouint8/fc_ref.json"
(corpus / "fc_ref.json").write_text(ref.read_text())
