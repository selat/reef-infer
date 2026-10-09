"""Generate and compile a corpus of small Edge TPU models for ISA analysis.

Each model varies one parameter around a base configuration, so differences
in the compiled bitstreams can be attributed to that parameter. Every model
directory holds `model.tflite`, `model_edgetpu.tflite` and `params.json`.

    .venv/bin/python tools/isa_corpus/gen.py [--out target/isa_corpus] [--jobs 24] [--only fc,conv]

Requires TensorFlow (dev dependency) and an `edgetpu_compiler`; set
EDGETPU_COMPILER to its path (default: `edgetpu_compiler` on PATH).
"""

import argparse
import zlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def sweep(base, **axes):
    """Base config plus one variant per value of each axis."""
    out = [dict(base)]
    for k, vals in axes.items():
        for v in vals:
            if base.get(k) != v:
                out.append({**base, k: v})
    return out


def configs():
    c = []
    fc = dict(N=64, K=64, bias=True, act="none", io="int8", in_range=4.0)
    for p in sweep(
        fc,
        N=[1, 4, 8, 16, 32, 48, 63, 65, 100, 128, 256, 500, 1024, 1500, 2048],
        K=[4, 8, 16, 32, 63, 65, 100, 128, 256, 1000, 2048, 4096],
        bias=[False],
        act=["relu", "relu6"],
        io=["uint8"],
        in_range=[0.5, 1.0, 16.0, 100.0],
    ):
        c.append(("fc", p))
    conv = dict(
        H=16, W=16, Cin=16, Cout=32, k=3, stride=1, pad="same", act="none", in_range=4.0
    )
    for p in sweep(
        conv,
        H=[4, 8, 9, 15, 32, 64],
        W=[4, 8, 9, 15, 32, 64],
        Cin=[1, 3, 4, 8, 32, 64, 128],
        Cout=[8, 16, 64, 128, 256],
        k=[1, 2, 5, 7],
        stride=[2],
        pad=["valid"],
        act=["relu", "relu6"],
        in_range=[0.5, 16.0],
    ):
        c.append(("conv", p))
    for k in (1, 3, 5):
        for s in (1, 2):
            for pad in ("same", "valid"):
                c.append(("conv", {**conv, "k": k, "stride": s, "pad": pad}))
    for k in range(
        1, 8
    ):  # grid: the op datapath region (bits 1886-1920) moves with k and W
        for W in (16, 32, 48, 64):
            c.append(("conv", {**conv, "k": k, "W": W}))
    dw = dict(
        H=16, W=16, C=32, k=3, stride=1, pad="same", mult=1, act="none", in_range=4.0
    )
    for k in (3, 5, 7):
        for W in (16, 32, 64):
            c.append(("dwconv", {**dw, "k": k, "W": W}))
    for p in sweep(
        dw,
        H=[8, 32],
        C=[8, 16, 64, 128],
        k=[5],
        stride=[2],
        pad=["valid"],
        mult=[2],
        act=["relu6"],
        in_range=[16.0],
    ):
        c.append(("dwconv", p))
    ew = dict(shape=[16, 16, 32], op="add", a_range=4.0, b_range=4.0, act="none")
    for p in sweep(
        ew,
        shape=[[64], [1024], [8, 8, 8], [32, 32, 64]],
        op=["mul", "sub", "max", "min"],
        b_range=[1.0, 2.0, 16.0],
        act=["relu"],
    ):
        c.append(("eltwise", p))
    pool = dict(H=16, W=16, C=32, op="max", size=2, stride=2, pad="valid")
    for p in sweep(
        pool, op=["avg"], size=[3], stride=[1], pad=["same"], C=[8, 128], H=[32]
    ):
        c.append(("pool", p))
    act = dict(shape=[16, 16, 32], op="logistic", in_range=8.0)
    for p in sweep(
        act,
        op=["tanh", "relu", "relu6", "leaky_relu", "hard_swish", "softmax"],
        shape=[[1024], [8, 8, 64]],
        in_range=[2.0, 32.0],
    ):
        c.append(("act", p))
    misc = dict(op="concat", shape=[8, 8, 16])
    for p in sweep(
        misc,
        op=["reshape", "pad", "mean", "resize_nearest", "resize_bilinear", "transpose"],
    ):
        c.append(("misc", p))
    return c


def name_of(family, p):
    return (
        family
        + "_"
        + "_".join(
            f"{k}{'x'.join(map(str, v)) if isinstance(v, list) else v}"
            for k, v in sorted(p.items())
        ).replace(".", "p")
    )


def build(family, p):
    import numpy as np
    import tensorflow as tf

    keras = tf.keras
    L = keras.layers
    rng = np.random.default_rng(zlib.crc32(name_of(family, p).encode()))
    act = {"none": None, "relu": "relu", "relu6": tf.nn.relu6}.get(p.get("act", "none"))
    ins = []
    if family == "fc":
        x = keras.Input((p["K"],))
        ins = [x]
        y = L.Dense(p["N"], use_bias=p["bias"], activation=act)(x)
        shapes = [(p["K"],)]
        ranges = [p["in_range"]]
    elif family == "conv":
        x = keras.Input((p["H"], p["W"], p["Cin"]))
        ins = [x]
        y = L.Conv2D(
            p["Cout"], p["k"], strides=p["stride"], padding=p["pad"], activation=act
        )(x)
        shapes = [(p["H"], p["W"], p["Cin"])]
        ranges = [p["in_range"]]
    elif family == "dwconv":
        x = keras.Input((p["H"], p["W"], p["C"]))
        ins = [x]
        y = L.DepthwiseConv2D(
            p["k"],
            strides=p["stride"],
            padding=p["pad"],
            depth_multiplier=p["mult"],
            activation=act,
        )(x)
        shapes = [(p["H"], p["W"], p["C"])]
        ranges = [p["in_range"]]
    elif family == "eltwise":
        a = keras.Input(tuple(p["shape"]))
        b = keras.Input(tuple(p["shape"]))
        ins = [a, b]
        f = {
            "add": tf.add,
            "mul": tf.multiply,
            "sub": tf.subtract,
            "max": tf.maximum,
            "min": tf.minimum,
        }[p["op"]]
        y = L.Lambda(lambda t: f(t[0], t[1]))([a, b])
        if act:
            y = L.Activation(act)(y)
        shapes = [tuple(p["shape"])] * 2
        ranges = [p["a_range"], p["b_range"]]
    elif family == "pool":
        x = keras.Input((p["H"], p["W"], p["C"]))
        ins = [x]
        P = L.MaxPooling2D if p["op"] == "max" else L.AveragePooling2D
        y = P(p["size"], strides=p["stride"], padding=p["pad"])(x)
        shapes = [(p["H"], p["W"], p["C"])]
        ranges = [4.0]
    elif family == "act":
        x = keras.Input(tuple(p["shape"]))
        ins = [x]
        f = {
            "logistic": tf.sigmoid,
            "tanh": tf.tanh,
            "relu": tf.nn.relu,
            "relu6": tf.nn.relu6,
            "leaky_relu": lambda t: tf.nn.leaky_relu(t, 0.1),
            "hard_swish": lambda t: t * tf.nn.relu6(t + 3) / 6,
            "softmax": tf.nn.softmax,
        }[p["op"]]
        y = L.Lambda(f)(x)
        shapes = [tuple(p["shape"])]
        ranges = [p["in_range"]]
    elif family == "misc":
        x = keras.Input(tuple(p["shape"]))
        ins = [x]
        op = p["op"]
        if op == "concat":
            y = L.Concatenate()([x, L.Lambda(lambda t: t * 0.5)(x)])
        elif op == "reshape":
            y = L.Reshape((-1, p["shape"][-1] * 2))(x)
        elif op == "pad":
            y = L.ZeroPadding2D(1)(x)
        elif op == "mean":
            y = L.GlobalAveragePooling2D()(x)
        elif op == "resize_nearest":
            y = L.UpSampling2D(2, interpolation="nearest")(x)
        elif op == "resize_bilinear":
            y = L.UpSampling2D(2, interpolation="bilinear")(x)
        elif op == "transpose":
            y = L.Permute((2, 1, 3))(x)
        shapes = [tuple(p["shape"])]
        ranges = [4.0]
    model = keras.Model(ins, y)
    for w in model.weights:
        w.assign(rng.normal(0, 0.5, w.shape).astype("float32"))

    def rep():
        for _ in range(64):
            yield [
                rng.uniform(-r, r, (1, *s)).astype("float32")
                for s, r in zip(shapes, ranges)
            ]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    io = tf.uint8 if p.get("io") == "uint8" else tf.int8
    conv.inference_input_type = io
    conv.inference_output_type = io
    return conv.convert()


def job(family, p, out, compiler):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    d = Path(out) / family / name_of(family, p)
    if (d / "model_edgetpu.tflite").exists():
        return d, "cached"
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps({"family": family, **p}))
    try:
        (d / "model.tflite").write_bytes(build(family, p))
    except Exception as e:
        return d, f"build failed: {type(e).__name__}: {str(e)[:200]}"
    r = subprocess.run(
        [compiler, "-s", "-o", str(d), str(d / "model.tflite")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    (d / "compile.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or not (d / "model_edgetpu.tflite").exists():
        return d, "compile failed"
    return d, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="target/isa_corpus")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) * 3 // 4))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    compiler = os.environ.get("EDGETPU_COMPILER") or shutil.which("edgetpu_compiler")
    if not compiler:
        sys.exit("set EDGETPU_COMPILER to the edgetpu_compiler binary or wrapper")
    cs = [(f, p) for f, p in configs() if not a.only or f in a.only.split(",")]
    seen, uniq = set(), []
    for f, p in cs:
        if name_of(f, p) not in seen:
            seen.add(name_of(f, p))
            uniq.append((f, p))
    print(f"{len(uniq)} models -> {a.out}", flush=True)
    status = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(job, f, p, a.out, compiler) for f, p in uniq]
        for fu in as_completed(futs):
            d, s = fu.result()
            status[s.split(":")[0]] = status.get(s.split(":")[0], 0) + 1
            if s not in ("ok", "cached"):
                print(f"{d.name}: {s}", flush=True)
    print(status)


if __name__ == "__main__":
    main()
