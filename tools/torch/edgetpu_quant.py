"""PT2E quantization for the Edge TPU, and export of the quantized graph
for `reef_infer::graph::Graph::from_torch`.

`EdgeTpuQuantizer` annotates an exported graph with the int8 scheme the
TPU computes: activations int8 per tensor (asymmetric), weights int8 per
output channel (symmetric, -127..127), biases int32 at scale
`s_in * s_w`; a ReLU after a conv / linear / add is fused (the output is
quantized after it); pools, flatten and reshapes keep their input's
quantization; a channel concat and all its inputs share one (the first
input's), so it only moves bytes.
"""

import torch
from torch.fx import Node
from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)

aten = torch.ops.aten

ACT = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)
WEIGHT = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
)

COMPUTE = (aten.conv2d.default, aten.linear.default, aten.add.Tensor)
RELU = (aten.relu.default, aten.relu_.default)
SHARED = (
    aten.adaptive_avg_pool2d.default,
    aten.mean.dim,
    aten.max_pool2d.default,
    aten.flatten.using_ints,
    aten.view.default,
    aten.reshape.default,
)


def _bias_qparams(obs):
    act, w = obs
    s_act, _ = act.calculate_qparams()
    s_w, _ = w.calculate_qparams()
    s = (s_act * s_w).to(torch.float32)
    return s, torch.zeros_like(s, dtype=torch.int32)


def _annotate(node, inputs, output):
    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=inputs, output_qspec=output, _annotated=True
    )


class EdgeTpuQuantizer(Quantizer):
    def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in COMPUTE:
                users = list(node.users)
                relu = users[0] if len(users) == 1 and users[0].target in RELU else None
                if node.target == aten.add.Tensor:
                    inputs = {a: ACT for a in node.args[:2]}
                else:
                    x, w = node.args[0], node.args[1]
                    inputs = {x: ACT, w: WEIGHT}
                    if len(node.args) > 2 and isinstance(node.args[2], Node):
                        inputs[node.args[2]] = DerivedQuantizationSpec(
                            derived_from=[(x, node), (w, node)],
                            derive_qparams_fn=_bias_qparams,
                            dtype=torch.int32,
                            quant_min=-(2**31),
                            quant_max=2**31 - 1,
                            qscheme=torch.per_channel_symmetric,
                            ch_axis=0,
                        )
                if relu is None:
                    _annotate(node, inputs, ACT)
                else:
                    _annotate(node, inputs, None)
                    _annotate(relu, {}, ACT)
            elif node.target in SHARED:
                x = node.args[0]
                _annotate(node, {x: ACT}, SharedQuantizationSpec((x, node)))
            elif node.target == aten.cat.default:
                xs = node.args[0]
                share = SharedQuantizationSpec(xs[0])
                for x in xs[1:]:
                    # the other producers quantize their outputs as the first
                    a = x.meta.get("quantization_annotation")
                    if a is not None:
                        a.output_qspec = share
                _annotate(node, {x: share for x in xs}, share)
        return gm

    def validate(self, gm: torch.fx.GraphModule) -> None:
        pass


# ── Export: the converted graph (`convert_pt2e`) as a JSON graph.
#
# {"format": "reef-graph", "version": 1,
#  "input": {"tensor": 0, "shape": [H, W, C], "scale": s, "zero_point": zp},
#  "output": t,
#  "ops": [{"op": ..., "inputs": [t, ...], "output": t, "shape": [H, W, C],
#           "scale": s, "zero_point": zp, ...op fields}]}
#
# Tensors are numbered (0 the graph input); shapes are NHWC without the
# batch; zero points are int8 (-128..127). Ops:
# - "conv": kernel, stride, pad (before, PyTorch's `padding`), relu;
#   weights int8 [Cout][k][k][Cin], weight_scales, bias int32 (scale
#   s_in * s_w).
# - "fc": relu; weights int8 [N][K] (K in the NHWC flatten order),
#   weight_scales, bias int32.
# - "add": relu (two inputs).
# - "global_avg_pool": the mean over the image (same quantization in and
#   out); "max_pool": kernel, stride.
# - "reshape": same data and quantization (flatten / view).
# - "concat": the inputs' channels one after the other (all of them
#   quantized as the output).

QD = torch.ops.quantized_decomposed


def _nhwc(shape):
    s = list(shape)[1:]  # drop the batch
    if len(s) == 3:
        return [s[1], s[2], s[0]]
    if len(s) == 1:
        return [1, 1, s[0]]
    raise ValueError(f"unsupported activation shape {list(shape)}")


def export_graph(gm: torch.fx.GraphModule) -> dict:
    """The JSON graph of a converted (int8) model with one input."""
    tensors = {}  # quantize node -> (id, scale, zp, NCHW shape)
    chw = {}  # tensor id -> (C, H, W) it was flattened from
    params = {}  # dequantize_per_channel node -> (values, scales)
    ops = []
    graph_in = None
    output = None

    def attr(n):
        return getattr(gm, n.target)

    def src(n):
        """The tensor a dequantize node reads."""
        if n.op != "call_function" or n.target != QD.dequantize_per_tensor.default:
            raise ValueError(f"{n.name}: input not a dequantized int8 tensor")
        return tensors[n.args[0]]

    def quantized_output(n):
        """(relu, quantize node) after a compute node."""
        users = list(n.users)
        relu = len(users) == 1 and users[0].target in RELU
        if relu:
            users = list(users[0].users)
        if len(users) != 1 or users[0].target != QD.quantize_per_tensor.default:
            raise ValueError(f"{n.name}: output not quantized to int8")
        return relu, users[0]

    def new_tensor(q, shape):
        tensors[q] = (len(tensors), float(q.args[1]), int(q.args[2]), list(shape))
        return tensors[q]

    def op(kind, n, ins, shape=None, **fields):
        """Appends op `kind` of node `n` reading tensors `ins`; its output
        is the int8 tensor after `n` (and a fused ReLU)."""
        relu, q = quantized_output(n)
        if relu and kind not in ("conv", "fc", "add"):
            raise ValueError(f"{n.name}: a ReLU after {kind} is not fused")
        t = new_tensor(q, shape or n.meta["val"].shape)
        d = {
            "op": kind,
            "inputs": [i[0] for i in ins],
            "output": t[0],
            "shape": _nhwc(t[3]),
            "scale": t[1],
            "zero_point": t[2],
        }
        if kind in ("conv", "fc", "add"):
            d["relu"] = relu
        d.update(fields)
        ops.append(d)
        return t

    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target == QD.dequantize_per_channel.default:
            v, s = attr(n.args[0]), attr(n.args[1])
            if (attr(n.args[2]) != 0).any() or n.args[3] != 0:
                raise ValueError(f"{n.name}: need symmetric per-output-channel scales")
            params[n] = (v, s)
        elif n.op == "placeholder":
            # (besides the export's shape guards)
            users = [u for u in n.users if u.op != "call_module"]
            if graph_in is not None or len(users) != 1:
                raise ValueError("need one graph input, quantized once")
            graph_in = new_tensor(users[0], n.meta["val"].shape)
        elif n.op == "call_function" and n.target in (
            aten.conv2d.default,
            aten.linear.default,
        ):
            x = src(n.args[0])
            w, ws = params[n.args[1]]
            b = (
                params[n.args[2]][0]
                if len(n.args) > 2 and n.args[2] is not None
                else None
            )
            cout = w.shape[0]
            bias = (
                b if b is not None else torch.zeros(cout, dtype=torch.int32)
            ).tolist()
            if n.target == aten.conv2d.default:
                a = list(n.args) + [None] * (7 - len(n.args))
                stride, pad, dil, groups = (
                    a[3] or [1, 1],
                    a[4] or [0, 0],
                    a[5] or [1, 1],
                    a[6] or 1,
                )
                k = w.shape[2]
                if (
                    w.shape[2] != w.shape[3]
                    or stride[0] != stride[1]
                    or pad[0] != pad[1]
                    or list(dil) != [1, 1]
                    or groups != 1
                ):
                    raise ValueError(
                        f"{n.name}: need square kernels, strides, padding; no dilation / groups"
                    )
                op(
                    "conv",
                    n,
                    [x],
                    kernel=k,
                    stride=stride[0],
                    pad=pad[0],
                    weights=w.permute(0, 2, 3, 1).flatten().tolist(),
                    weight_scales=ws.tolist(),
                    bias=bias,
                )
            else:
                if x[0] in chw:
                    # K flattened from C, H, W; the TPU's tensor is H, W, C
                    c, h, ww = chw[x[0]]
                    w = w.reshape(cout, c, h, ww).permute(0, 2, 3, 1).reshape(cout, -1)
                op(
                    "fc",
                    n,
                    [x],
                    weights=w.flatten().tolist(),
                    weight_scales=ws.tolist(),
                    bias=bias,
                )
        elif n.op == "call_function" and n.target == aten.add.Tensor:
            op("add", n, [src(n.args[0]), src(n.args[1])])
        elif n.op == "call_function" and n.target == aten.cat.default:
            dim = n.args[1] if len(n.args) > 1 else 0
            if len(n.meta["val"].shape) != 4 or dim % 4 != 1:
                raise ValueError(f"{n.name}: only channel concats")
            ins = [src(x) for x in n.args[0]]
            t = op("concat", n, ins)
            if any((i[1], i[2]) != (t[1], t[2]) for i in ins):
                raise ValueError(f"{n.name}: inputs quantized unlike the output")
        elif n.op == "call_function" and n.target in (
            aten.adaptive_avg_pool2d.default,
            aten.mean.dim,
        ):
            x = src(n.args[0])
            out = list(n.meta["val"].shape)
            if out[2:] not in ([1, 1], []) or (
                n.target == aten.mean.dim and sorted(d % 4 for d in n.args[1]) != [2, 3]
            ):
                raise ValueError(f"{n.name}: only global average pools")
            op("global_avg_pool", n, [x], shape=out if len(out) == 4 else out + [1, 1])
        elif n.op == "call_function" and n.target == aten.max_pool2d.default:
            x = src(n.args[0])
            a = list(n.args) + [None] * (6 - len(n.args))
            k, s = a[1], a[2] or a[1]
            if k[0] != k[1] or s[0] != s[1] or (a[3] and any(a[3])):
                raise ValueError(f"{n.name}: need square, unpadded max pools")
            op("max_pool", n, [x], kernel=k[0], stride=s[0])
        elif n.op == "call_function" and n.target in (
            aten.flatten.using_ints,
            aten.view.default,
            aten.reshape.default,
        ):
            x = src(n.args[0])
            t = op("reshape", n, [x])
            if len(x[3]) == 4 and len(t[3]) == 2 and x[3][2] * x[3][3] > 1:
                chw[t[0]] = tuple(x[3][1:])
        elif n.op == "output":
            (res,) = n.args[0] if isinstance(n.args[0], (list, tuple)) else (n.args[0],)
            if res.target == QD.dequantize_per_tensor.default:
                res = res.args[0]
            output = tensors[res][0]
        elif (
            n.op == "call_function"
            and n.target
            in (
                QD.quantize_per_tensor.default,
                QD.dequantize_per_tensor.default,
            )
            or n.target in RELU
            or n.op in ("get_attr", "call_module")
        ):
            continue  # handled with the op they belong to, or constants / guards
        else:
            raise ValueError(f"{n.name}: {n.target} not supported")
    _, s, zp, shape = graph_in
    return {
        "format": "reef-graph",
        "version": 1,
        "input": {"tensor": 0, "shape": _nhwc(shape), "scale": s, "zero_point": zp},
        "output": output,
        "ops": ops,
    }
