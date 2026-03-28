use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Instruction decoder
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error),
    InvalidHeader,
    InvalidSize { expected: usize, got: usize },
    MissingStartWord,
    MissingHaltWord,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "I/O error: {e}"),
            ParseError::InvalidHeader => write!(f, "invalid file header"),
            ParseError::InvalidSize { expected, got } => {
                write!(f, "invalid size: expected {expected} bytes, got {got}")
            }
            ParseError::MissingStartWord => write!(f, "missing start sentinel at word 3"),
            ParseError::MissingHaltWord => write!(f, "missing halt word at end"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::Io(e)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Register {
    ParamLo,
    ParamHi,
    ScratchLo,
    ScratchHi,
    ActHi,
    ActLo,
    ClearHi,
    ClearLo,
}

impl Register {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x1b => Some(Register::ParamLo),
            0x0b => Some(Register::ParamHi),
            0x3b => Some(Register::ScratchLo),
            0x2b => Some(Register::ScratchHi),
            0xab => Some(Register::ActHi),
            0xbb => Some(Register::ActLo),
            0x4b => Some(Register::ClearHi),
            0x5b => Some(Register::ClearLo),
            _ => None,
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::ParamLo => write!(f, "r_param_lo"),
            Register::ParamHi => write!(f, "r_param_hi"),
            Register::ScratchLo => write!(f, "r_scratch_lo"),
            Register::ScratchHi => write!(f, "r_scratch_hi"),
            Register::ActHi => write!(f, "r_act_hi"),
            Register::ActLo => write!(f, "r_act_lo"),
            Register::ClearHi => write!(f, "r_clear_hi"),
            Register::ClearLo => write!(f, "r_clear_lo"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmaKind {
    Infeed,
    Generic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputSkeleton {
    Single,
    Vector,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Header {
        n_words: u16,
    },
    Movi {
        reg: Register,
        imm32: u32,
    },
    DmaSize {
        kind: DmaKind,
        bytes: u32,
    },
    Fence {
        seq: u8,
    },
    FencePair,
    TileConfig {
        multi_tile: bool,
    },
    MacCtrl {
        skeleton: OutputSkeleton,
    },
    /// Stride-carrying MAC config word: hi64(this word) + lo64(next word) = 8×i16 slots.
    MacConfig {
        slots: [i16; 8],
        input_dim: u32,
    },
    StartSentinel,
    Halt,
    Unknown {
        raw: [u8; 16],
    },
}

pub struct Program {
    pub words: Vec<[u8; 16]>,
    pub instructions: Vec<Instruction>,
}

pub struct ModelParams {
    pub input_dim: u32,
    pub dma_input_bytes: u32,
    pub output_skeleton: OutputSkeleton,
}

const START_WORD: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0x07,
];
const HALT_WORD: [u8; 16] = [
    0xC0, 0x0F, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

pub fn parse_instructions(path: &Path) -> Result<Program, ParseError> {
    let data = std::fs::read(path)?;

    if data.len() < 16 || data[0] != 0x80 {
        return Err(ParseError::InvalidHeader);
    }

    let size = u16::from_le_bytes([data[3], data[4]]) as usize;
    let expected_len = size * 4 + 32;
    if expected_len != data.len() {
        return Err(ParseError::InvalidSize {
            expected: expected_len,
            got: data.len(),
        });
    }

    let n_words = size / 4 + 2;

    let words: Vec<[u8; 16]> = (0..n_words)
        .map(|i| {
            let mut w = [0u8; 16];
            w.copy_from_slice(&data[i * 16..(i + 1) * 16]);
            w
        })
        .collect();

    if words.len() <= 3 || words[3] != START_WORD {
        return Err(ParseError::MissingStartWord);
    }

    if words.last() != Some(&HALT_WORD) {
        return Err(ParseError::MissingHaltWord);
    }

    let instructions: Vec<Instruction> = words
        .iter()
        .enumerate()
        .map(|(i, w)| decode_word(i, w, words.get(i + 1)))
        .collect();

    Ok(Program {
        words,
        instructions,
    })
}

fn word_hi64(w: &[u8; 16]) -> u64 {
    u64::from_le_bytes(w[8..16].try_into().unwrap())
}

/// Decode a `(_, 0x14)` word whose bytes[10-11] == 0x00 0x80, indicating a
/// stride-carrying MAC config.  Slots 0-3 come from hi64 of `w`; slots 4-7
/// from lo64 of `next`.
fn decode_mac_config(w: &[u8; 16], next: &[u8; 16]) -> Instruction {
    let mut slots = [0i16; 8];
    // First 4 slots: bytes 8..16 of this word
    for (i, slot) in slots[..4].iter_mut().enumerate() {
        *slot = i16::from_le_bytes([w[8 + i * 2], w[9 + i * 2]]);
    }
    // Last 4 slots: bytes 0..8 of next word
    for (i, slot) in slots[4..].iter_mut().enumerate() {
        *slot = i16::from_le_bytes([next[i * 2], next[i * 2 + 1]]);
    }
    let input_dim = ((-(slots[5] as i32) + 1) * 4) as u32;
    Instruction::MacConfig { slots, input_dim }
}

fn decode_word(word_index: usize, w: &[u8; 16], next: Option<&[u8; 16]>) -> Instruction {
    // Exact matches
    if *w == START_WORD {
        return Instruction::StartSentinel;
    }
    if *w == HALT_WORD {
        return Instruction::Halt;
    }

    match (w[0], w[1]) {
        (0x80, 0x0f) => {
            let size = u16::from_le_bytes([w[3], w[4]]) as u32;
            let n_words = (size / 4 + 2) as u16;
            Instruction::Header { n_words }
        }
        (0x80, 0xf6) => Instruction::Fence { seq: w[3] },
        (0x00, 0xf9) => Instruction::FencePair,
        (0x40, 0x19) => Instruction::TileConfig {
            multi_tile: w[4] != 0,
        },
        (0x40, 0x10) => Instruction::MacCtrl {
            skeleton: OutputSkeleton::Single,
        },
        (0x00, 0x15) => Instruction::MacCtrl {
            skeleton: OutputSkeleton::Vector,
        },
        // Stride-carrying MAC config: byte[1]=0x14, bytes[10-11]=0x00 0x80
        (_, 0x14) if w[10] == 0x00 && w[11] == 0x80 => {
            let next = next.unwrap_or(&[0u8; 16]);
            decode_mac_config(w, next)
        }
        (_, 0x08) => {
            let val = ((word_hi64(w) >> 6) & 0xFFFF_FFFF) as u32;
            if w[6] == 0xc0 {
                let reg = Register::from_byte(w[7]).unwrap_or(Register::ParamLo);
                Instruction::Movi { reg, imm32: val }
            } else {
                let kind = if word_index == 27 {
                    DmaKind::Infeed
                } else {
                    DmaKind::Generic
                };
                Instruction::DmaSize { kind, bytes: val }
            }
        }
        _ => Instruction::Unknown { raw: *w },
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, instr) in self.instructions.iter().enumerate() {
            match instr {
                Instruction::Header { n_words } => {
                    writeln!(f, "word {:03}  HEADER      n_words={}", i, n_words)?;
                }
                Instruction::Fence { seq } => {
                    writeln!(f, "word {:03}  FENCE       seq={:#04x}", i, seq)?;
                }
                Instruction::FencePair => {
                    writeln!(f, "word {:03}  FENCE_PAIR", i)?;
                }
                Instruction::Movi { reg, imm32 } => {
                    writeln!(
                        f,
                        "word {:03}  MOVI        {:<14}  imm={:#010x}",
                        i,
                        reg.to_string(),
                        imm32
                    )?;
                }
                Instruction::TileConfig { multi_tile } => {
                    writeln!(f, "word {:03}  TILE_CFG    multi_tile={}", i, multi_tile)?;
                }
                Instruction::DmaSize { kind, bytes } => {
                    let kind_str = match kind {
                        DmaKind::Infeed => "infeed ",
                        DmaKind::Generic => "generic",
                    };
                    writeln!(f, "word {:03}  DMA_SIZE    {} {} bytes", i, kind_str, bytes)?;
                }
                Instruction::MacConfig { slots, input_dim } => {
                    writeln!(
                        f,
                        "word {:03}  MAC_CFG     slots={:?}  input_dim={}",
                        i, slots, input_dim
                    )?;
                }
                Instruction::MacCtrl { skeleton } => {
                    writeln!(f, "word {:03}  MAC_CTRL    skeleton={:?}", i, skeleton)?;
                }
                Instruction::StartSentinel => {
                    writeln!(f, "word {:03}  START", i)?;
                }
                Instruction::Halt => {
                    writeln!(f, "word {:03}  HALT", i)?;
                }
                Instruction::Unknown { raw } => {
                    let hex: Vec<String> = raw.iter().map(|b| format!("{:02X}", b)).collect();
                    writeln!(f, "word {:03}  UNKNOWN  {}", i, hex.join(" "))?;
                }
            }
        }
        Ok(())
    }
}

impl Program {
    pub fn extract_params(&self) -> Option<ModelParams> {
        let dma_input_bytes = self.instructions.iter().find_map(|instr| {
            if let Instruction::DmaSize {
                kind: DmaKind::Infeed,
                bytes,
            } = instr
            {
                Some(*bytes)
            } else {
                None
            }
        })?;

        let input_dim = self.instructions.iter().find_map(|instr| {
            if let Instruction::MacConfig { input_dim, .. } = instr {
                Some(*input_dim)
            } else {
                None
            }
        })?;

        let output_skeleton = self.instructions.iter().find_map(|instr| {
            if let Instruction::MacCtrl { skeleton } = instr {
                Some(skeleton.clone())
            } else {
                None
            }
        })?;

        Some(ModelParams {
            input_dim,
            dma_input_bytes,
            output_skeleton,
        })
    }

    /// Return all MOVI instructions with their word index.
    pub fn movi_instructions(&self) -> Vec<(usize, Register, u32)> {
        self.instructions
            .iter()
            .enumerate()
            .filter_map(|(i, instr)| {
                if let Instruction::Movi { reg, imm32 } = instr {
                    Some((i, reg.clone(), *imm32))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn parse_instructions_from_assets() {
        let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let instr_path = assets_dir.join("instructions0_chunk0.bin");

        let prog = parse_instructions(&instr_path).expect("parse_instructions failed");
        assert!(!prog.instructions.is_empty());
        assert!(matches!(prog.instructions[0], Instruction::Header { .. }));
        assert!(matches!(prog.instructions[3], Instruction::StartSentinel));
        assert!(matches!(prog.instructions.last(), Some(Instruction::Halt)));
    }

    #[test]
    #[ignore]
    fn mac_config_input_dim() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("generated_tflite_ops");

        for (model, expected_dim) in [("dotproduct_8", 8u32), ("dotproduct_16", 16u32)] {
            for chunk in ["instructions0_chunk0.bin", "instructions1_chunk0.bin"] {
                let path = base.join(model).join(chunk);
                if !path.exists() {
                    continue;
                }
                let prog =
                    parse_instructions(&path).unwrap_or_else(|e| panic!("{model}/{chunk}: {e}"));
                let mac_cfg = prog.instructions.iter().find_map(|i| {
                    if let Instruction::MacConfig { input_dim, .. } = i {
                        Some(*input_dim)
                    } else {
                        None
                    }
                });
                assert_eq!(
                    mac_cfg,
                    Some(expected_dim),
                    "{model}/{chunk}: expected input_dim={expected_dim}",
                );
            }
        }
    }
}
