//! Several layer programs as one program (one instruction stream, one
//! call).
//!
//! [`splice`] joins complete bitstreams: each keeps its own prologue and
//! epilogue, the inner `halt` / `end` framing is dropped, the tile
//! instructions' sequence numbers continue from one part to the next, and
//! relocations move with their part.

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
