"""Analyse unexplained (reserved) instruction bits across a dumped corpus.

    cargo run --example isa_check -- --dump target/isa_corpus_dump $(find target/isa_corpus -name model_edgetpu.tflite)
    python3 tools/isa_corpus/analyze.py target/isa_corpus_dump [--region op:1878] [--fields w_zp,in_zp]

Without --region: one line per reserved region (adjacent reserved chunks
merged) with how often it is nonzero, which bits ever toggle, and the
contiguous runs of toggling bits as candidate fields.
With --region LAYOUT:LO: every instance's candidate sub-fields next to the
model parameters and the named fields given in --fields.
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def load(dump):
    out = []
    for jf in glob.glob(os.path.join(dump, "*.json")):
        doc = json.load(open(jf))
        pj = os.path.join(os.path.dirname(doc.get("path", "")), "params.json")
        params = (
            json.load(open(pj))
            if os.path.exists(pj)
            else {"family": os.path.basename(doc.get("path", "?"))}
        )
        prev = None
        for i, ins in enumerate(doc["instrs"]):
            if ins.get("layout"):
                out.append(dict(doc=doc, params=params, ins=ins, idx=i, prev=prev))
            prev = ins
    return out


def regions(rsv):
    """merge adjacent reserved chunks: {lo: (width, value)}"""
    items = sorted((lo, w, v) for lo, w, v in rsv.values())
    merged = []
    for lo, w, v in items:
        if merged and merged[-1][0] + merged[-1][1] == lo:
            plo, pw, pv = merged[-1]
            merged[-1] = (plo, pw + w, pv | (v << pw))
        else:
            merged.append((lo, w, v))
    return merged


def runs(mask):
    """contiguous runs of set bits: [(bit, width)]"""
    out, b = [], 0
    while mask >> b:
        if mask >> b & 1:
            s = b
            while mask >> b & 1:
                b += 1
            out.append((s, b - s))
        else:
            b += 1
    return out


def bit_regions(recs):
    """per (layout, absolute bit): instance count; regions from bit positions so chunk merging can't miss values"""
    toggles = defaultdict(Counter)
    totals = Counter()
    for r in recs:
        lay = r["ins"]["layout"]
        totals[lay] += 1
        for lo, w, v in r["ins"].get("rsv", {}).values():
            for b in range(w):
                if v >> b & 1:
                    toggles[lay][lo + b] += 1
    return toggles, totals


def summary(recs):
    toggles, totals = bit_regions(recs)
    for lay in sorted(toggles):
        bits = sorted(toggles[lay])
        groups, cur = [], [bits[0]]
        for b in bits[1:]:
            if b - cur[-1] <= 8:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        for g in groups:
            lo, hi = g[0], g[-1]
            vals = Counter()
            for r in recs:
                if r["ins"]["layout"] != lay:
                    continue
                v = extract(r, lo, hi - lo + 1)
                if v:
                    vals[v] += 1
            fams = Counter(
                r["params"].get("family")
                for r in recs
                if r["ins"]["layout"] == lay and extract(r, lo, hi - lo + 1)
            )
            print(
                f"{lay:13} bits {lo}..{hi} ({hi - lo + 1}b)  nonzero {sum(vals.values())}/{totals[lay]}  distinct {len(vals)}"
                f"  set bits {[b - lo for b in g]}  families {dict(fams)}"
            )


def extract(r, lo, width):
    v = 0
    for rlo, w, x in r["ins"].get("rsv", {}).values():
        for b in range(w):
            if x >> b & 1 and lo <= rlo + b < lo + width:
                v |= 1 << (rlo + b - lo)
    return v


def detail(recs, layout, lo, width, fields, limit):
    sel = [r for r in recs if r["ins"]["layout"] == layout]
    mask = 0
    for r in sel:
        mask |= extract(r, lo, width)
    subs = runs(mask)
    print(
        f"{layout} bits {lo}..{lo + width - 1}: candidate fields (bit offset, width) {subs}"
    )
    rows = Counter()
    for r in sel:
        v = extract(r, lo, width)
        sub = tuple(v >> b & ((1 << w) - 1) for b, w in subs)
        p = {k: v for k, v in r["params"].items()}
        fam = p.pop("family", "?")
        f = tuple(r["ins"]["fields"].get(n) for n in fields)
        rows[
            (fam, json.dumps(p, sort_keys=True), r["doc"]["exec_type"][:4], sub, f)
        ] += 1
    print(
        f"{'family':8} {'exec':4} {str([f'{lo + b}+{w}' for b, w in subs]):40} {str(fields):30} params"
    )
    for (fam, p, ex, sub, f), n in sorted(rows.items())[:limit]:
        print(f"{fam:8} {ex:4} {str([hex(x) for x in sub]):40} {str(f):30} {p} x{n}")


def relate(recs, layout, lo, width, top=10):
    """rank formulas v == a*f + b, f >> k, f1 + f2, (f1+1)*(f2+1) over the instruction's named fields by how many
    instances (zero and nonzero) they predict exactly"""
    sel = [r for r in recs if r["ins"]["layout"] == layout]
    vals = [extract(r, lo, width) for r in sel]
    nz = [i for i, v in enumerate(vals) if v]
    if not nz:
        print("never set")
        return
    names = sorted({k for r in sel for k in r["ins"]["fields"] if k != "opcode"})
    cols = {k: [r["ins"]["fields"].get(k, 0) for r in sel] for k in names}
    N = len(sel)

    def score(pred):
        allhit = sum(p == v for p, v in zip(pred, vals))
        nzhit = sum(pred[i] == vals[i] for i in nz)
        return allhit, nzhit

    res = []
    for k in names:
        c = cols[k]
        for a_ in (1, 2, 4, 8, 16, 64, -1):
            for b_ in range(-4, 5):
                res.append((*score([a_ * x + b_ for x in c]), f"{a_}*{k}{b_:+d}"))
        for sh in range(1, 8):
            res.append((*score([x >> sh for x in c]), f"{k}>>{sh}"))
        res.append((*score([int(x != 0) for x in c]), f"[{k} != 0]"))
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            cx, cy = cols[x], cols[y]
            res.append((*score([p + q for p, q in zip(cx, cy)]), f"{x}+{y}"))
            res.append(
                (
                    *score([(p + 1) * (q + 1) for p, q in zip(cx, cy)]),
                    f"({x}+1)*({y}+1)",
                )
            )
    res = [r for r in res if r[1] > 0]
    res.sort(key=lambda r: (r[0], r[1]), reverse=True)
    print(
        f"{layout} bits {lo}..{lo + width - 1}: {len(nz)} nonzero of {N}, distinct {len(set(vals[i] for i in nz))}"
    )
    for allhit, nzhit, h in res[:top]:
        print(f"  all {allhit:5}/{N}  nonzero {nzhit:4}/{len(nz)}  {h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--region")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--fields", default="")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument(
        "--relate", action="store_true", help="search simple formulas over named fields"
    )
    a = ap.parse_args()
    recs = load(a.dump)
    if not a.region:
        summary(recs)
        return
    lay, lo = a.region.split(":")
    lo = int(lo)
    width = a.width
    if not width:
        toggles, _ = bit_regions(recs)
        hi = lo
        while any(b in toggles[lay] for b in range(hi + 1, hi + 9)):
            hi = max(b for b in range(hi + 1, hi + 9) if b in toggles[lay])
        width = hi - lo + 1
    if a.relate:
        relate(recs, lay, lo, width)
        return
    detail(recs, lay, lo, width, [f for f in a.fields.split(",") if f], a.limit)


if __name__ == "__main__":
    main()
