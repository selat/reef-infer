//! DarwiNN (Edge TPU) instruction encoder and decoder.
//!
//! An instruction is one or more little-endian 128-bit words. Bit 0 is the
//! least significant bit of byte 0 of word 0, and word `j` holds bits
//! `[128j, 128j + 128)`. Bits 6-11 of the first word are the opcode, which
//! fixes the instruction length.
//!
//! Each known opcode has a [`Layout`]: a list of named bit fields. Bits that no
//! named field covers become reserved fields (`rsv<lo>`), so
//! `encode(decode(words)) == words` holds for every bit pattern, and nonzero
//! reserved bits mark what the layouts do not explain yet.
//!
//! Field layouts are ported from tinycoral (MIT, Copyright 2026 Frederik
//! Rogalski, <https://github.com/FrederikRogalski/tinycoral>, `coral/isa/`),
//! which reverse engineered them from `edgetpu_compiler` output.

mod layouts;

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

pub use layouts::{SCALAR_COUNTERS, TILE_COUNTERS};

/// Size of one instruction word in bytes.
pub const WORD_BYTES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsaError {
    /// The bitstream length is not a multiple of 16 bytes.
    Unaligned { len: usize },
    /// The bitstream ends inside an instruction.
    Truncated {
        word_index: usize,
        opcode: u8,
        need: usize,
        have: usize,
    },
    /// The words given to a layout have the wrong count.
    WrongLength {
        layout: &'static str,
        expected: usize,
        got: usize,
    },
    /// The opcode does not belong to the layout.
    WrongOpcode { layout: &'static str, opcode: u8 },
    /// No layout has a field with this name.
    UnknownField { layout: &'static str, field: String },
    /// The value does not fit the field width.
    Overflow {
        layout: &'static str,
        field: String,
        value: i64,
        width: u32,
        signed: bool,
    },
    /// No layout is known for this opcode.
    UnknownOpcode { opcode: u8 },
}

impl fmt::Display for IsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsaError::Unaligned { len } => {
                write!(f, "bitstream length {len} is not a multiple of 16")
            }
            IsaError::Truncated {
                word_index,
                opcode,
                need,
                have,
            } => write!(
                f,
                "instruction at word {word_index} (opcode {opcode:#04x}) needs {need} words, only {have} left"
            ),
            IsaError::WrongLength {
                layout,
                expected,
                got,
            } => {
                write!(f, "{layout}: expected {expected} words, got {got}")
            }
            IsaError::WrongOpcode { layout, opcode } => write!(f, "{layout}: opcode {opcode:#04x}"),
            IsaError::UnknownField { layout, field } => {
                write!(f, "{layout}: unknown field {field}")
            }
            IsaError::Overflow {
                layout,
                field,
                value,
                width,
                signed,
            } => write!(
                f,
                "{layout}.{field}={value} does not fit {width} {} bits",
                if *signed { "signed" } else { "unsigned" }
            ),
            IsaError::UnknownOpcode { opcode } => write!(f, "no layout for opcode {opcode:#04x}"),
        }
    }
}

impl std::error::Error for IsaError {}

/// The opcode of an instruction's first word.
pub fn opcode(first_word: u128) -> u8 {
    ((first_word >> 6) & 0x3f) as u8
}

/// Instruction length in words for an opcode. Unknown opcodes count as one word.
///
/// Lengths for 0x3e 0x1a 0x24 0x20 0x25 0x14 0x11 0x26 0x17 0x13 0x01 0x27 0x10
/// 0x21 0x3f were verified on hardware by tinycoral; 0x02 0x03 0x15 0x18 0x19
/// 0x22 0x23 follow from decoded layouts; the rest are statistical.
pub fn instr_len(op: u8) -> usize {
    match op {
        0x00 => 14,
        0x01..=0x03 => 17,
        0x05 => 4,
        0x06 => 3,
        0x07 => 7,
        0x0b => 2,
        0x0c => 7,
        0x10..=0x12 => 4,
        0x13 | 0x14 => 7,
        0x15..=0x18 => 6,
        0x19 => 13,
        0x1c | 0x1d => 5,
        0x1e => 4,
        0x24 | 0x25 | 0x27 => 2,
        0x26 => 4,
        0x29 => 3,
        0x2a => 15,
        0x30 => 11,
        0x32 => 15,
        0x34 => 2,
        0x39 => 35,
        0x3c => 12,
        _ => 1,
    }
}

/// One bit field of a layout.
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub lo: u32,
    pub width: u32,
    pub signed: bool,
}

impl Field {
    pub fn is_reserved(&self) -> bool {
        self.name.starts_with("rsv")
    }
}

/// The field layout of a fixed-length instruction.
#[derive(Debug)]
pub struct Layout {
    pub name: &'static str,
    pub opcodes: &'static [u8],
    pub nwords: usize,
    pub fields: Vec<Field>,
    index: HashMap<String, usize>,
    defaults: Vec<(usize, i64)>,
}

/// Reserved gaps are split into chunks of at most this many bits so every
/// value fits an `i64`.
const RSV_CHUNK: u32 = 32;

impl Layout {
    /// Builds a layout from `(name, lo, width, signed)` fields. Gaps become
    /// reserved fields. Panics on overlapping or duplicate fields, since
    /// layouts are static tables.
    fn new(
        name: &'static str,
        opcodes: &'static [u8],
        nwords: usize,
        mut named: Vec<(String, u32, u32, bool)>,
        defaults: &[(&str, i64)],
    ) -> Layout {
        named.sort_by_key(|f| f.1);
        let end = 128 * nwords as u32;
        let mut fields = Vec::new();
        let mut pos = 0u32;
        let push_rsv = |fields: &mut Vec<Field>, from: u32, to: u32| {
            let mut p = from;
            while p < to {
                let w = (to - p).min(RSV_CHUNK);
                fields.push(Field {
                    name: format!("rsv{p}"),
                    lo: p,
                    width: w,
                    signed: false,
                });
                p += w;
            }
        };
        for (n, lo, w, s) in named {
            assert!(
                lo >= pos,
                "{name}: field {n} at bit {lo} overlaps (pos {pos})"
            );
            assert!(w > 0 && w <= 64, "{name}: field {n} width {w}");
            assert!(lo + w <= end, "{name}: field {n} past the end");
            push_rsv(&mut fields, pos, lo);
            fields.push(Field {
                name: n,
                lo,
                width: w,
                signed: s,
            });
            pos = lo + w;
        }
        push_rsv(&mut fields, pos, end);
        let index: HashMap<String, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        assert_eq!(index.len(), fields.len(), "{name}: duplicate field names");
        let mut layout = Layout {
            name,
            opcodes,
            nwords,
            fields,
            index,
            defaults: Vec::new(),
        };
        let mut d: Vec<(&str, i64)> = Vec::new();
        if layout.index.contains_key("opcode") {
            d.push(("opcode", opcodes[0] as i64));
        }
        d.extend_from_slice(defaults);
        layout.defaults = d.iter().map(|(n, v)| (layout.index[*n], *v)).collect();
        layout
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.index.get(name).map(|&i| &self.fields[i])
    }

    /// Decodes `words` into one value per field, in `self.fields` order.
    pub fn decode(&self, words: &[u128]) -> Result<Decoded<'_>, IsaError> {
        if words.len() != self.nwords {
            return Err(IsaError::WrongLength {
                layout: self.name,
                expected: self.nwords,
                got: words.len(),
            });
        }
        let op = opcode(words[0]);
        if !self.opcodes.contains(&op) {
            return Err(IsaError::WrongOpcode {
                layout: self.name,
                opcode: op,
            });
        }
        let values = self
            .fields
            .iter()
            .map(|f| {
                let raw = get_bits(words, f.lo, f.width);
                if f.signed && f.width < 64 && raw >> (f.width - 1) & 1 == 1 {
                    (raw as i64) - (1i64 << f.width)
                } else {
                    raw as i64
                }
            })
            .collect();
        Ok(Decoded {
            layout: self,
            values,
        })
    }

    /// Encodes the given fields on top of the layout's defaults. Fields not
    /// named are zero (or their default).
    pub fn encode(&self, kv: &[(&str, i64)]) -> Result<Vec<u128>, IsaError> {
        let mut values = vec![0i64; self.fields.len()];
        for &(i, v) in &self.defaults {
            values[i] = v;
        }
        for &(n, v) in kv {
            let i = *self.index.get(n).ok_or_else(|| IsaError::UnknownField {
                layout: self.name,
                field: n.to_string(),
            })?;
            values[i] = v;
        }
        self.encode_values(&values)
    }

    fn encode_values(&self, values: &[i64]) -> Result<Vec<u128>, IsaError> {
        let mut words = vec![0u128; self.nwords];
        for (f, &v) in self.fields.iter().zip(values) {
            let fits = if f.signed {
                let half = 1i128 << (f.width - 1);
                (-half..half).contains(&(v as i128))
            } else {
                v >= 0 && (v as i128) < (1i128 << f.width)
            };
            if !fits {
                return Err(IsaError::Overflow {
                    layout: self.name,
                    field: f.name.clone(),
                    value: v,
                    width: f.width,
                    signed: f.signed,
                });
            }
            set_bits(&mut words, f.lo, f.width, v as u64);
        }
        let op = opcode(words[0]);
        if !self.opcodes.contains(&op) {
            return Err(IsaError::WrongOpcode {
                layout: self.name,
                opcode: op,
            });
        }
        Ok(words)
    }
}

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn get_bits(words: &[u128], lo: u32, width: u32) -> u64 {
    let (wi, off) = ((lo / 128) as usize, lo % 128);
    let mut v = words[wi] >> off;
    if off + width > 128 {
        v |= words[wi + 1] << (128 - off);
    }
    (v & mask(width)) as u64
}

fn set_bits(words: &mut [u128], lo: u32, width: u32, value: u64) {
    let v = value as u128 & mask(width);
    let (wi, off) = ((lo / 128) as usize, lo % 128);
    words[wi] |= v << off;
    if off + width > 128 {
        words[wi + 1] |= v >> (128 - off);
    }
}

/// A decoded instruction: one value per field of its layout.
#[derive(Debug, Clone)]
pub struct Decoded<'a> {
    pub layout: &'a Layout,
    pub values: Vec<i64>,
}

impl Decoded<'_> {
    pub fn get(&self, name: &str) -> Option<i64> {
        self.layout.index.get(name).map(|&i| self.values[i])
    }

    pub fn set(&mut self, name: &str, value: i64) -> Result<(), IsaError> {
        let i = *self
            .layout
            .index
            .get(name)
            .ok_or_else(|| IsaError::UnknownField {
                layout: self.layout.name,
                field: name.to_string(),
            })?;
        self.values[i] = value;
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u128>, IsaError> {
        self.layout.encode_values(&self.values)
    }

    /// Named (non-reserved) fields with their values.
    pub fn named(&self) -> impl Iterator<Item = (&str, i64)> {
        self.layout
            .fields
            .iter()
            .zip(&self.values)
            .filter(|(f, _)| !f.is_reserved())
            .map(|(f, &v)| (f.name.as_str(), v))
    }

    /// Reserved fields that are nonzero: bits the layout does not explain.
    pub fn unexplained(&self) -> impl Iterator<Item = (&Field, i64)> {
        self.layout
            .fields
            .iter()
            .zip(&self.values)
            .filter(|(f, v)| f.is_reserved() && **v != 0)
            .map(|(f, &v)| (f, v))
    }
}

impl fmt::Display for Decoded<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.layout.name)?;
        for (n, v) in self.named() {
            if v != 0 && n != "opcode" {
                write!(f, " {n}={v:#x}")?;
            }
        }
        for (fl, v) in self.unexplained() {
            write!(f, " !{}={v:#x}", fl.name)?;
        }
        Ok(())
    }
}

struct Registry {
    layouts: Vec<Layout>,
    by_opcode: HashMap<u8, usize>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| {
        let layouts = layouts::all();
        let mut by_opcode = HashMap::new();
        for (i, l) in layouts.iter().enumerate() {
            for &op in l.opcodes {
                assert_eq!(
                    l.nwords,
                    instr_len(op),
                    "{}: length table mismatch for {op:#x}",
                    l.name
                );
                assert!(
                    by_opcode.insert(op, i).is_none(),
                    "opcode {op:#x} registered twice"
                );
            }
        }
        Registry { layouts, by_opcode }
    })
}

/// The layout of an opcode, if decoded.
pub fn layout(op: u8) -> Option<&'static Layout> {
    let r = registry();
    r.by_opcode.get(&op).map(|&i| &r.layouts[i])
}

/// Every known layout.
pub fn layouts() -> &'static [Layout] {
    &registry().layouts
}

/// One instruction cut from a bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instr {
    /// Index of the first word in the bitstream.
    pub word_index: usize,
    pub words: Vec<u128>,
}

impl Instr {
    pub fn opcode(&self) -> u8 {
        opcode(self.words[0])
    }

    pub fn decode(&self) -> Result<Decoded<'static>, IsaError> {
        let op = self.opcode();
        layout(op)
            .ok_or(IsaError::UnknownOpcode { opcode: op })?
            .decode(&self.words)
    }
}

/// Reads a bitstream into 128-bit words.
pub fn words(bytes: &[u8]) -> Result<Vec<u128>, IsaError> {
    if bytes.len() % WORD_BYTES != 0 {
        return Err(IsaError::Unaligned { len: bytes.len() });
    }
    Ok(bytes
        .chunks_exact(WORD_BYTES)
        .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Cuts a bitstream into instructions using the opcode length table.
pub fn split(bytes: &[u8]) -> Result<Vec<Instr>, IsaError> {
    let ws = words(bytes)?;
    let mut out = Vec::new();
    let mut i = 0;
    while i < ws.len() {
        let op = opcode(ws[i]);
        let n = instr_len(op);
        if i + n > ws.len() {
            return Err(IsaError::Truncated {
                word_index: i,
                opcode: op,
                need: n,
                have: ws.len() - i,
            });
        }
        out.push(Instr {
            word_index: i,
            words: ws[i..i + n].to_vec(),
        });
        i += n;
    }
    Ok(out)
}

/// Serializes instructions back into a bitstream.
pub fn assemble<'a>(instrs: impl IntoIterator<Item = &'a [u128]>) -> Vec<u8> {
    instrs
        .into_iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// One line per instruction: word index, opcode, and decoded fields.
pub fn disassemble(bytes: &[u8]) -> Result<String, IsaError> {
    use std::fmt::Write;
    let mut s = String::new();
    for ins in split(bytes)? {
        let _ = match ins.decode() {
            Ok(d) => writeln!(s, "{:5}  {:02x}  {d}", ins.word_index, ins.opcode()),
            Err(_) => writeln!(
                s,
                "{:5}  {:02x}  ?? {}",
                ins.word_index,
                ins.opcode(),
                ins.words
                    .iter()
                    .map(|w| format!("{w:032x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        };
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_build_and_cover_every_bit() {
        for l in layouts() {
            let total: u32 = l.fields.iter().map(|f| f.width).sum();
            assert_eq!(total, 128 * l.nwords as u32, "{}", l.name);
            assert_eq!(
                opcode(l.encode(&[]).unwrap()[0]),
                l.opcodes[0],
                "{}",
                l.name
            );
        }
    }

    #[test]
    fn signed_fields_round_trip() {
        let l = layout(0x14).unwrap();
        let w = l
            .encode(&[("seq", 7), ("n_inc2", -5), ("sync_dec", -1)])
            .unwrap();
        let d = l.decode(&w).unwrap();
        assert_eq!(d.get("n_inc2"), Some(-5));
        assert_eq!(d.get("sync_dec"), Some(-1));
        assert_eq!(d.get("n_inc0"), Some(1)); // default
        assert_eq!(d.encode().unwrap(), w);
    }

    #[test]
    fn overflow_is_rejected() {
        let l = layout(0x1a).unwrap();
        assert!(matches!(
            l.encode(&[("tiles", 1 << 16)]),
            Err(IsaError::Overflow { .. })
        ));
        assert!(matches!(
            l.encode(&[("bogus", 1)]),
            Err(IsaError::UnknownField { .. })
        ));
    }

    #[test]
    fn end_encodes_like_the_compiler() {
        // The final word the old parser called HALT_WORD is `end` with length 128.
        let end = layout(0x3f)
            .unwrap()
            .encode(&[("length_bits", 128)])
            .unwrap();
        let mut expected = [0u8; 16];
        expected[..4].copy_from_slice(&[0xC0, 0x0F, 0x00, 0x04]);
        assert_eq!(end[0].to_le_bytes(), expected);
    }

    #[test]
    fn split_rejects_truncation() {
        let op = layout(0x01).unwrap().encode(&[]).unwrap();
        let bytes = assemble([&op[..3]]);
        assert!(matches!(split(&bytes), Err(IsaError::Truncated { .. })));
    }
}
