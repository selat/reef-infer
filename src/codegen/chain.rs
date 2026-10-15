//! Several layer programs as one program (one instruction stream, one
//! call).
//!
//! [`splice`] joins complete bitstreams: each keeps its own prologue and
//! epilogue, the inner `halt` / `end` framing is dropped, the tile
//! instructions' sequence numbers continue from one part to the next, and
//! relocations move with their part.
//!
//! [`program`] chains layers with the activations kept on chip: convs
//! and residual adds as a small graph ([`Node`]), possibly ending in a
//! global average pool and an FC on its output. Every intermediate
//! tensor is stored once, in a padded patch in narrow memory ([`plan`]),
//! from the layer that writes it ([`conv::Io::output_to`],
//! [`add::Io::output_to`]) until the last layer that reads it, so a
//! shortcut stays on chip across the convs of its block. Only the first
//! layer's input comes from the host and only the last one's output goes
//! back. [`convs`] is the plain sequence.

use super::add::{self, Add};
use super::avg_pool::GlobalAvgPool;
use super::conv::{self, Conv};
use super::fc::Fc;
use super::*;
use crate::isa;

type Error = Box<dyn std::error::Error>;

/// Joins `parts` (each a complete program from [`Asm::finish`]) into one.
/// Relocation names are kept: give the parts' layers distinct names
/// ([`rename`]) when they would collide.
pub fn splice(parts: &[ChunkInfo]) -> Result<ChunkInfo, Error> {
    let mut body: Vec<Vec<u128>> = Vec::new();
    let mut relocs = Vec::new();
    let mut seq = 0;
    for (i, part) in parts.iter().enumerate() {
        let instrs = isa::split(&part.bitstream)?;
        // `start` first; `halt`, 4 nops and `end` last
        let n = instrs.len();
        if n < 7 || instrs[0].opcode() != OP_START || instrs[n - 1].opcode() != OP_END {
            return Err(format!("part {i} is not a complete program").into());
        }
        let keep_tail = i + 1 == parts.len();
        let last = if keep_tail { n } else { n - 6 };
        // bit offset of this part's instruction 1 in the joined stream
        let words_before: usize = 1 + body.iter().map(|w| w.len()).sum::<usize>();
        let first_word = instrs[0].words.len();
        for r in &part.field_offsets {
            relocs.push(OffsetInfo {
                offset_bit: r.offset_bit + ((words_before - first_word) * 128) as i32,
                desc: r.desc,
                position: r.position,
                name: r.name.clone(),
            });
        }
        let mut part_max = None;
        for ins in &instrs[1..last] {
            let mut d = ins.decode()?;
            match d.get("seq") {
                Some(s) => {
                    d.set("seq", seq + s)?;
                    part_max = Some(part_max.unwrap_or(0).max(s));
                    body.push(d.encode()?);
                }
                None => body.push(ins.words.clone()),
            }
        }
        seq += part_max.map_or(0, |m| m + 1);
    }
    let inner = (body.iter().map(|w| w.len()).sum::<usize>() - 1) as i64 * 128;
    let start = isa::layout(OP_START)
        .unwrap()
        .encode(&[("opcode", OP_START as i64), ("length_bits", inner)])?;
    let bitstream = isa::assemble(std::iter::once(&start[..]).chain(body.iter().map(|w| &w[..])));
    Ok(ChunkInfo {
        bitstream,
        field_offsets: relocs,
    })
}

/// Renames a part's input / output relocations (`from` -> `to`).
pub fn rename(chunk: &mut ChunkInfo, from: &str, to: &str) {
    for r in &mut chunk.field_offsets {
        if r.name == from {
            r.name = to.to_string();
        }
    }
}

/// Narrow memory per tile (bytes).
pub const NARROW_BYTES: i64 = 192 * 1024;

/// A layer of a chain: a conv or FC with its parameters cached at a wide
/// base, an add, or a global average pool. A pool's output (on tile 0) is
/// read only by an FC, and an FC (reading a pool's) comes last.
#[derive(Debug, Clone, Copy)]
pub enum Op<'a> {
    Conv(&'a Conv, i64),
    Add(&'a Add),
    AvgPool(&'a GlobalAvgPool),
    Fc(&'a Fc, i64),
}

/// A chain layer and the tensors it reads: 0 is the chain's input, `k + 1`
/// layer `k`'s output (an add's inputs in its A, B order).
#[derive(Debug, Clone)]
pub struct Node<'a> {
    pub op: Op<'a>,
    pub inputs: Vec<usize>,
}

/// Where a chain keeps its tensors: per tensor (indexed as
/// [`Node::inputs`]) the narrow byte address and patch geometry (`None`
/// for the chain's input, which the first conv lands itself, and its
/// output, which goes to the host), and the adds' scratch.
#[derive(Debug, Clone)]
pub struct Plan {
    pub at: Vec<Option<(i64, Patch)>>,
    pub scratch: i64,
}

/// The tensor's patch padded for one more reader: the same interior,
/// the larger padding on every side.
fn union(p: Option<Patch>, q: Patch) -> Result<Patch, Error> {
    let Some(p) = p else { return Ok(q) };
    if p.cin != q.cin || p.rows != q.rows || p.cols != q.cols {
        return Err(format!("chain: a tensor is read as {p:?} and as {q:?}").into());
    }
    Ok(Patch {
        top: p.top.max(q.top),
        bottom: p.bottom.max(q.bottom),
        left: p.left.max(q.left),
        right: p.right.max(q.right),
        ..p
    })
}

/// Lays out `nodes`' tensors in narrow memory. Each intermediate tensor
/// gets one patch padded for all its readers (an add's two inputs share
/// one geometry), at the lowest 256 B aligned address free while it is
/// live (from its writer to its last reader, both included). Everything
/// lies above every layer's own buffers ([`Conv::narrow_end`], the adds'
/// scratch).
pub fn plan(nodes: &[Node]) -> Result<Plan, Error> {
    let n = nodes.len();
    let up = |x: i64| (x + 255) / 256 * 256;
    match nodes.first() {
        Some(Node {
            op: Op::Conv(..),
            inputs,
        }) if inputs[..] == [0] => {}
        _ => return Err("chain: the first layer must be a conv of the chain input".into()),
    }
    let mut last_use = vec![None; n + 1];
    for (k, node) in nodes.iter().enumerate() {
        let want = match node.op {
            Op::Conv(..) => 1,
            Op::Add(_) => 2,
            Op::AvgPool(_) => 1,
            Op::Fc(..) if k + 1 == n => 1,
            Op::Fc(..) => return Err("chain: an FC must come last".into()),
        };
        if node.inputs.len() != want || node.inputs.iter().any(|&t| t > k) {
            return Err(format!("chain: layer {k} reads {:?}", node.inputs).into());
        }
        if k > 0 && node.inputs.contains(&0) {
            return Err("chain: only the first layer reads the chain input".into());
        }
        for &t in &node.inputs {
            last_use[t] = Some(k);
            // a pool's output (tile 0 only) feeds an FC and nothing else
            let pooled = t > 0 && matches!(nodes[t - 1].op, Op::AvgPool(_));
            if pooled != matches!(node.op, Op::Fc(..)) {
                return Err(format!("chain: layer {k} reads layer {}'s output", t - 1).into());
            }
        }
    }
    if let Some(t) = (1..n).find(|&t| last_use[t].is_none()) {
        return Err(format!("chain: layer {}'s output is never read", t - 1).into());
    }
    // patches: every reader's, then an add's two inputs alike
    let mut patch: Vec<Option<Patch>> = vec![None; n + 1];
    for node in &nodes[1..] {
        for &t in &node.inputs {
            let q = match node.op {
                Op::Conv(c, _) => c.input_patch(),
                Op::Add(a) => a.patch(),
                Op::AvgPool(p) => p.input_patch(),
                Op::Fc(f, _) => f.input_patch(),
            };
            patch[t] = Some(union(patch[t], q)?);
        }
    }
    loop {
        let mut changed = false;
        for node in nodes {
            if let (Op::Add(_), [x, y]) = (node.op, &node.inputs[..]) {
                let u = union(patch[*x], patch[*y].unwrap())?;
                changed |= patch[*x] != Some(u) || patch[*y] != Some(u);
                (patch[*x], patch[*y]) = (Some(u), Some(u));
            }
        }
        if !changed {
            break;
        }
    }
    // the layers' own buffers, then the adds' scratch
    let own = nodes
        .iter()
        .filter_map(|l| match l.op {
            Op::Conv(c, _) => Some(c.narrow_end()),
            Op::AvgPool(p) => Some(p.narrow_end()),
            Op::Fc(f, _) => Some(f.narrow_end()),
            Op::Add(_) => None,
        })
        .max()
        .unwrap_or(0);
    let scratch = up(own);
    let floor = scratch
        + nodes
            .iter()
            .enumerate()
            .filter_map(|(k, l)| match l.op {
                Op::Add(a) => Some(up(a.scratch_bytes(k + 1 == n))),
                Op::Conv(..) | Op::AvgPool(_) | Op::Fc(..) => None,
            })
            .max()
            .unwrap_or(0);
    // first fit, in the order the tensors are written
    let mut at: Vec<Option<(i64, Patch)>> = vec![None; n + 1];
    for t in 1..n {
        let (p, live) = (patch[t].unwrap(), (t - 1, last_use[t].unwrap()));
        let mut taken: Vec<(i64, i64)> = (1..t)
            .filter(|&u| last_use[u].unwrap() >= live.0)
            .map(|u| {
                let (a, q) = at[u].unwrap();
                (a, a + up(q.tp()))
            })
            .collect();
        taken.sort();
        let mut addr = floor;
        for (lo, hi) in taken {
            if addr + up(p.tp()) <= lo {
                break;
            }
            addr = addr.max(hi);
        }
        if addr + p.tp() > NARROW_BYTES {
            return Err(format!(
                "chain: tensor {t} ({} B) does not fit narrow memory above {addr:#x}",
                p.tp()
            )
            .into());
        }
        at[t] = Some((addr, p));
    }
    Ok(Plan { at, scratch })
}

/// One program running `nodes` in order with the activations on chip
/// ([`plan`]). Its input layer is the first conv's (`input`), its output
/// layer the last layer's (`output`).
pub fn program(nodes: &[Node]) -> Result<ChunkInfo, Error> {
    let plan = plan(nodes)?;
    let n = nodes.len();
    let parts = nodes
        .iter()
        .enumerate()
        .map(|(k, node)| {
            let output_to = (k + 1 < n).then(|| plan.at[k + 1].unwrap());
            match node.op {
                Op::Conv(c, base) => c.chain_program(
                    base,
                    conv::Io {
                        input_at: plan.at[node.inputs[0]],
                        output_to,
                    },
                ),
                Op::Add(a) => {
                    let ((ia, p), (ib, _)) = (
                        plan.at[node.inputs[0]].unwrap(),
                        plan.at[node.inputs[1]].unwrap(),
                    );
                    a.chain_program(add::Io {
                        a: ia,
                        b: ib,
                        patch: p,
                        output_to,
                        scratch: plan.scratch,
                    })
                }
                Op::AvgPool(p) => p.chain_program(plan.at[node.inputs[0]].unwrap(), output_to),
                Op::Fc(f, base) => f.chain_program(base, plan.at[node.inputs[0]].unwrap().0),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    splice(&parts)
}

/// [`program`] for convs one after the other, each with its parameters
/// cached at its wide base.
pub fn convs(layers: &[(&Conv, i64)]) -> Result<ChunkInfo, Error> {
    let nodes: Vec<Node> = layers
        .iter()
        .enumerate()
        .map(|(k, &(c, base))| Node {
            op: Op::Conv(c, base),
            inputs: vec![k],
        })
        .collect();
    program(&nodes)
}
