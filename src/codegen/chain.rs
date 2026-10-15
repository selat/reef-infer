//! Several layer programs as one program (one instruction stream, one
//! call).
//!
//! [`splice`] joins complete bitstreams: each keeps its own prologue and
//! epilogue, the inner `halt` / `end` framing is dropped, the tile
//! instructions' sequence numbers continue from one part to the next, and
//! relocations move with their part.
//!
//! [`convs`] chains convolutions with the activations kept on chip: each
//! conv writes its output into the next one's padded patch in narrow
//! memory ([`conv::Io`]), two patch buffers in turn above every layer's
//! own buffers. Only the first layer's input comes from the host and only
//! the last one's output goes back.

use super::conv::{self, Conv};
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

/// The two narrow byte addresses the patches of a conv chain alternate
/// between: above every layer's own buffers, each big enough for the
/// largest patch.
pub fn conv_buffers(layers: &[&Conv]) -> Result<[i64; 2], Error> {
    let up = |n: i64| (n + 255) / 256 * 256;
    let end = layers.iter().map(|c| c.narrow_end()).max().unwrap_or(0);
    let patch = layers
        .iter()
        .skip(1)
        .map(|c| c.input_patch().tp())
        .max()
        .unwrap_or(0);
    let bufs = [up(end), up(end) + up(patch)];
    if bufs[1] + patch > NARROW_BYTES {
        return Err(format!(
            "conv chain: patches at {bufs:#x?} of {patch} B do not fit narrow memory"
        )
        .into());
    }
    Ok(bufs)
}

/// One program running `layers` (each with its parameters cached at its
/// wide base) in order, the activations staying on chip. Its input layer is
/// the first conv's (`input`), its output the last one's (`output`).
pub fn convs(layers: &[(&Conv, i64)]) -> Result<ChunkInfo, Error> {
    let convs: Vec<&Conv> = layers.iter().map(|l| l.0).collect();
    let buf = conv_buffers(&convs)?;
    let n = layers.len();
    let parts = layers
        .iter()
        .enumerate()
        .map(|(i, &(c, base))| {
            let io = conv::Io {
                input_at: (i > 0).then(|| buf[(i + 1) % 2]),
                output_to: (i + 1 < n).then(|| (buf[i % 2], convs[i + 1].input_patch())),
            };
            c.chain_program(base, io)
        })
        .collect::<Result<Vec<_>, _>>()?;
    splice(&parts)
}
