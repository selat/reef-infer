"""ResNet-8 on CIFAR-10 in PyTorch, int8-quantized with PT2E
(`torch.export`, `edgetpu_quant.EdgeTpuQuantizer`) and exported as a
JSON graph for `reef_infer::graph::Graph::from_torch`.

    .venv/bin/python tools/torch/resnet8.py [--out target/cifar10] [--epochs 20] [--export-only]

The same network as `tools/cifar10/train.py` (MLPerf Tiny ResNet-8), in
PyTorch's usual form: 3x3 convs with `padding=1` (on stride 2 that is
1 pixel before and none after, where TensorFlow's "same" pads after),
batch norm folded into the convs before quantization, an adaptive
average pool and a linear layer. Input: pixels / 255, NCHW.

Writes to `--out`:
- `resnet8_torch.pt`: the trained float weights.
- `resnet8_torch.json`: the int8 graph.
- `resnet8_torch_logits.bin`: 10000 x 10 int8 logits of the quantized
  PyTorch model (its q / dq reference graph) on the test set.
- `test_images.bin` / `test_labels.bin` as `tools/cifar10/train.py`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.insert(0, str(Path(__file__).parent))
from edgetpu_quant import EdgeTpuQuantizer, export_graph

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="target/cifar10")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--export-only", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


def conv_bn(cin, cout, k, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
    )


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = conv_bn(cin, cout, 3, stride)
        self.b = conv_bn(cout, cout, 3, 1)
        self.short = nn.Conv2d(cin, cout, 1, stride) if stride != 1 else None

    def forward(self, x):
        y = self.b(F.relu(self.a(x)))
        return F.relu(y + (x if self.short is None else self.short(x)))


class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv_bn(3, 16, 3, 1)
        self.blocks = nn.Sequential(
            Block(16, 16, 1), Block(16, 32, 2), Block(32, 64, 2)
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.blocks(F.relu(self.stem(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def fold_bn(m):
    """The eval model with every conv + batch norm pair as one conv."""
    for name, child in m.named_children():
        if (
            isinstance(child, nn.Sequential)
            and len(child) == 2
            and isinstance(child[1], nn.BatchNorm2d)
        ):
            setattr(m, name, fuse_conv_bn_eval(child[0], child[1]))
        else:
            fold_bn(child)
    return m


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train[:, 0], y_test[:, 0]


def to_t(x):
    return torch.from_numpy(x.astype(np.float32) / 255.0).permute(0, 3, 1, 2)


x_tr, x_te = to_t(x_train), to_t(x_test)
yt_tr = torch.from_numpy(y_train.astype(np.int64))

model = ResNet8()
if a.export_only:
    model.load_state_dict(torch.load(out / "resnet8_torch.pt"))
else:
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for epoch in range(a.epochs):
        model.train()
        t0, perm = time.time(), torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 64):
            idx = perm[i : i + 64]
            x = x_tr[idx]
            # random horizontal flip and translation by up to 4 pixels
            flip = torch.rand(len(x)) < 0.5
            x[flip] = x[flip].flip(3)
            dy, dx = np.random.randint(-4, 5, 2)
            x = torch.roll(F.pad(x, (4, 4, 4, 4)), (int(dy), int(dx)), (2, 3))[
                :, :, 4:36, 4:36
            ]
            loss = F.cross_entropy(model(x), yt_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1).numpy() == y_test).mean()
        print(
            f"epoch {epoch + 1}: loss {loss.item():.3f}, test {acc:.4f}, {time.time() - t0:.0f} s",
            flush=True,
        )
    torch.save(model.state_dict(), out / "resnet8_torch.pt")

model.eval()
with torch.no_grad():
    print(f"float accuracy: {(model(x_te).argmax(1).numpy() == y_test).mean():.4f}")

# ── int8: export with a batch of 1, calibrate, convert, export the graph
folded = fold_bn(model)
gm = torch.export.export(folded, (x_tr[:1],)).module()
prepared = prepare_pt2e(gm, EdgeTpuQuantizer())
with torch.no_grad():
    for i in range(0, 5000, 10):
        prepared(x_tr[i : i + 1])
quantized = convert_pt2e(prepared)
g = export_graph(quantized)
(out / "resnet8_torch.json").write_text(json.dumps(g))

out_op = next(o for o in g["ops"] if o["output"] == g["output"])
s, zp = out_op["scale"], out_op["zero_point"]
logits = np.zeros((len(x_te), 10), np.int8)
with torch.no_grad():
    for i in range(len(x_te)):
        y = quantized(x_te[i : i + 1])[0].numpy()
        logits[i] = np.clip(np.round(y / s) + zp, -128, 127)
print(f"int8 PyTorch accuracy: {(logits.argmax(1) == y_test).mean():.4f}")
print(
    f"input quantization: scale {g['input']['scale']}, zero point {g['input']['zero_point']}"
)
x_test.astype(np.uint8).tofile(out / "test_images.bin")
y_test.astype(np.uint8).tofile(out / "test_labels.bin")
logits.tofile(out / "resnet8_torch_logits.bin")
