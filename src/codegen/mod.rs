//! Edge TPU program generation without `edgetpu_compiler`.
//!
//! [`Asm`] assembles instructions from named fields (see [`crate::isa`]),
//! numbers tile-dispatched instructions, records address relocations and
//! frames the bitstream with `start` / `halt` / `end`. [`Program`] turns the
//! result into an [`ExecInfo`] that [`crate::runner::Device`] runs like a
//! compiled model.

pub mod elementwise;
pub mod fc;

use crate::executable_generated::platforms::darwinn::{Description, ExecutableType, Position};
use crate::isa::{self, IsaError};
use crate::model::{ChunkInfo, ExecInfo, LayerInfo, OffsetInfo, OutputLayoutInfo, TfliteModel};

pub const OP_TENSOR: u8 = 0x01;
pub const OP_RING_PRODUCER: u8 = 0x10;
pub const OP_RING_CONSUMER: u8 = 0x11;
pub const OP_NARROW_TO_WIDE: u8 = 0x13;
pub const OP_WIDE_TO_NARROW: u8 = 0x14;
pub const OP_MESH_N: u8 = 0x17;
pub const OP_SYNC: u8 = 0x1a;
pub const OP_SCALAR: u8 = 0x20;
pub const OP_HALT: u8 = 0x21;
pub const OP_SCSYNC: u8 = 0x24;
pub const OP_POP: u8 = 0x25;
pub const OP_INFEED: u8 = 0x26;
pub const OP_OUTFEED: u8 = 0x27;
pub const OP_START: u8 = 0x3e;
pub const OP_END: u8 = 0x3f;

/// Scalar ALU op that loads `imm_scalar` into register `s_x`.
const S_MOVI: i64 = 0x2f;
/// Bit offset of `scalar.imm_scalar` within the instruction.
const IMM_SCALAR_BIT: usize = 70;

/// Instruction assembler for one bitstream.
pub struct Asm {
    words: Vec<u128>,
    seq: i64,
    relocs: Vec<OffsetInfo>,
}

impl Default for Asm {
    fn default() -> Self {
        Self::new()
    }
}

impl Asm {
    /// Starts a bitstream; `start.length_bits` is filled in by [`Asm::finish`].
    pub fn new() -> Asm {
        let mut a = Asm {
            words: Vec::new(),
            seq: 0,
            relocs: Vec::new(),
        };
        a.emit(OP_START, &[]).unwrap();
        a
    }

    /// Appends one instruction and returns its first word index. Instructions
    /// whose layout has a `seq` field get the next sequence number unless
    /// `fields` sets it.
    pub fn emit(&mut self, op: u8, fields: &[(&str, i64)]) -> Result<usize, IsaError> {
        let layout = isa::layout(op).ok_or(IsaError::UnknownOpcode { opcode: op })?;
        let mut kv: Vec<(&str, i64)> = Vec::with_capacity(fields.len() + 2);
        if layout.field("opcode").is_some() {
            kv.push(("opcode", op as i64));
        } else {
            // scalar bundles: opcode = branch | enable_scalar << 5
            kv.push(("branch", (op & 0x1f) as i64));
        }
        if layout.field("seq").is_some() && !fields.iter().any(|(n, _)| *n == "seq") {
            kv.push(("seq", self.seq));
            self.seq += 1;
        }
        kv.extend_from_slice(fields);
        let at = self.words.len();
        self.words.extend(layout.encode(&kv)?);
        Ok(at)
    }

    /// `mov s<reg>, imm` with `imm` patched at run time to one half of a
    /// buffer address.
    pub fn mov_addr(
        &mut self,
        reg: i64,
        desc: Description,
        position: Position,
        name: &str,
    ) -> Result<(), IsaError> {
        let at = self.emit(OP_SCALAR, &[("s_op", S_MOVI), ("s_x", reg)])?;
        self.relocs.push(OffsetInfo {
            offset_bit: (at * 128 + IMM_SCALAR_BIT) as i32,
            desc,
            position,
            name: name.to_string(),
        });
        Ok(())
    }

    /// `mov s<reg>, imm`.
    pub fn mov(&mut self, reg: i64, imm: i64) -> Result<(), IsaError> {
        self.emit(
            OP_SCALAR,
            &[("s_op", S_MOVI), ("s_x", reg), ("imm_scalar", imm)],
        )?;
        Ok(())
    }

    /// Ends the program (`halt 4; nop x4; end`) and returns the bitstream.
    pub fn finish(mut self) -> Result<ChunkInfo, IsaError> {
        self.emit(OP_HALT, &[("code", 4)])?;
        for _ in 0..4 {
            self.emit(OP_SCALAR, &[])?;
        }
        self.emit(OP_END, &[("length_bits", 128)])?;
        let inner = (self.words.len() - 2) as i64 * 128;
        self.words[0] = isa::layout(OP_START)
            .unwrap()
            .encode(&[("opcode", OP_START as i64), ("length_bits", inner)])?[0];
        let bitstream = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        Ok(ChunkInfo {
            bitstream,
            field_offsets: self.relocs,
        })
    }
}

// ── Program skeleton shared by the generators ───────────────────────────

pub const ALL_TILES: i64 = 0xffff;
/// Scalar registers holding the high / low half of a host buffer address.
pub const R_ADDR_HI: i64 = 10;
pub const R_ADDR_LO: i64 = 11;
/// Host descriptor tags (the `tag` of a USB event).
pub const TAG_INPUT: i64 = 1;
pub const TAG_PARAMS: i64 = 2;
pub const TAG_OUTPUT: i64 = 3;
pub const TAG_DONE: i64 = 4;

/// `sync` on every tile: wait for `counters`, optionally signal the scalar core.
pub fn sync(a: &mut Asm, counters: i64, units: i64, signal: bool) -> Result<(), IsaError> {
    let mut f = vec![
        ("tiles", ALL_TILES),
        ("counters", counters),
        ("units", units),
    ];
    if signal {
        f.push(("signal", 1));
    }
    a.emit(OP_SYNC, &f)?;
    Ok(())
}

/// Scalar core waits for every tile's token.
pub fn wait_tiles(a: &mut Asm) -> Result<(), IsaError> {
    a.emit(
        OP_SCSYNC,
        &[
            ("tiles", ALL_TILES),
            ("tokwait", 1),
            ("tmask", ALL_TILES),
            ("sunits", 0xc),
        ],
    )?;
    Ok(())
}

/// `s<dst> = s<src>`.
pub fn mov_reg(a: &mut Asm, dst: i64, src: i64) -> Result<(), IsaError> {
    a.emit(
        OP_SCALAR,
        &[("s_op", 0xf), ("s_x", dst), ("imm_scalar", src)],
    )?;
    Ok(())
}

/// Resets all counters and loads the parameter / scratch base addresses
/// into s0:s1 and s2:s3 (high:low).
pub fn prologue(a: &mut Asm) -> Result<(), IsaError> {
    use Description as D;
    use Position as P;
    sync(a, 0x7ffff, 0x3ff, true)?;
    a.emit(
        OP_SCSYNC,
        &[
            ("tiles", ALL_TILES),
            ("smask", 0x1ff),
            ("tokwait", 1),
            ("tmask", ALL_TILES),
            ("sunits", 0x1f),
        ],
    )?;
    a.mov_addr(1, D::BASE_ADDRESS_PARAMETER, P::LOWER_32BIT, "")?;
    a.mov_addr(0, D::BASE_ADDRESS_PARAMETER, P::UPPER_32BIT, "")?;
    a.mov_addr(3, D::BASE_ADDRESS_SCRATCH, P::LOWER_32BIT, "")?;
    a.mov_addr(2, D::BASE_ADDRESS_SCRATCH, P::UPPER_32BIT, "")?;
    a.emit(OP_SCSYNC, &[("smask", 5), ("sunits", 0xc)])?;
    sync(a, 0, 0x3ff, true)?;
    wait_tiles(a)
}

/// Loads an input buffer address into s10:s11 and waits for the input stream.
pub fn load_input_addr(a: &mut Asm, name: &str) -> Result<(), IsaError> {
    use Description as D;
    use Position as P;
    a.mov_addr(
        R_ADDR_HI,
        D::BASE_ADDRESS_INPUT_ACTIVATION,
        P::UPPER_32BIT,
        name,
    )?;
    a.mov_addr(
        R_ADDR_LO,
        D::BASE_ADDRESS_INPUT_ACTIVATION,
        P::LOWER_32BIT,
        name,
    )?;
    a.emit(
        OP_SCSYNC,
        &[("smask", 4), ("value", 0x1fff), ("sunits", 0xc)],
    )?;
    a.emit(OP_SCSYNC, &[("smask", 1), ("sunits", 0xc)])?;
    Ok(())
}

/// Loads an output buffer address into s10:s11.
pub fn load_output_addr(a: &mut Asm, name: &str) -> Result<(), IsaError> {
    use Description as D;
    use Position as P;
    a.mov_addr(
        R_ADDR_HI,
        D::BASE_ADDRESS_OUTPUT_ACTIVATION,
        P::UPPER_32BIT,
        name,
    )?;
    a.mov_addr(
        R_ADDR_LO,
        D::BASE_ADDRESS_OUTPUT_ACTIVATION,
        P::LOWER_32BIT,
        name,
    )
}

/// First half of a host descriptor: s5 = s11 (address low), s4 = s10 (high).
pub fn descriptor_addr(a: &mut Asm) -> Result<(), IsaError> {
    mov_reg(a, 5, R_ADDR_LO)?;
    mov_reg(a, 4, R_ADDR_HI)
}

/// Second half: sends `{s5, s4, len, tag}` to the host as a USB event.
/// `v_op` 0xa writes descriptor word `v_offset` from `s[vs_reg]` and 0xc
/// sends it; the compiler bundles the `mov s8` / `mov s9` into those words.
pub fn descriptor_send(a: &mut Asm, len: i64, tag: i64) -> Result<(), IsaError> {
    mov_reg(a, 6, 5)?;
    mov_reg(a, 7, 4)?;
    a.emit(OP_SCALAR, &[("v_op", 0xa), ("vs_reg", 6)])?;
    a.emit(
        OP_SCALAR,
        &[
            ("v_op", 0xa),
            ("v_offset", 1),
            ("vs_reg", 7),
            ("s_op", S_MOVI),
            ("s_x", 8),
            ("imm_scalar", len),
        ],
    )?;
    a.emit(
        OP_SCALAR,
        &[
            ("v_op", 0xa),
            ("v_offset", 2),
            ("vs_reg", 8),
            ("s_op", S_MOVI),
            ("s_x", 9),
            ("imm_scalar", tag),
        ],
    )?;
    a.emit(OP_SCALAR, &[("v_op", 0xa), ("v_offset", 3), ("vs_reg", 9)])?;
    a.emit(OP_SCALAR, &[("v_op", 0xc)])?;
    Ok(())
}

/// Drains the tiles and the outfeed, then sends the completion event.
/// `outfeed_thr` is the compiler's `scsync.thr5` (16 after 16 outfeeds, 2
/// after one; not decoded).
pub fn epilogue(a: &mut Asm, outfeed_thr: i64) -> Result<(), IsaError> {
    sync(a, 0x1ffff, 0x3ff, false)?;
    a.emit(
        OP_SCSYNC,
        &[
            ("tiles", ALL_TILES),
            ("smask", 0x20),
            ("thr5", outfeed_thr),
            ("wait", 0x20),
            ("sunits", 0xc),
        ],
    )?;
    sync(a, 0x20000, 0x3ff, true)?;
    wait_tiles(a)?;
    a.emit(
        OP_SYNC,
        &[
            ("tiles", ALL_TILES),
            ("units", 0x3ff),
            ("signal", 1),
            ("f109", 3),
        ],
    )?;
    wait_tiles(a)?;
    done(a)
}

/// Sends the completion event (tag 4, no address).
pub fn done(a: &mut Asm) -> Result<(), IsaError> {
    a.mov(4, 0)?;
    a.mov(5, 0)?;
    descriptor_send(a, 0, TAG_DONE)
}

/// Quantization of a uint8 tensor: `real = (q - zero_point) * scale`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quant {
    pub scale: f32,
    pub zero_point: i32,
}

/// A uint8 layer of `y x x x z` elements.
pub fn layer(name: &str, (y, x, z): (usize, usize, usize), q: Quant) -> LayerInfo {
    LayerInfo {
        name: name.to_string(),
        size_bytes: (y * x * z) as i32,
        zero_point: q.zero_point,
        dequantization_factor: q.scale,
        y_dim: y as i32,
        x_dim: x as i32,
        z_dim: z as i32,
        output_layout: None,
    }
}

/// A generated stand-alone program with one input and one output.
pub struct Program {
    pub chunk: ChunkInfo,
    pub parameters: Vec<u8>,
    pub input: LayerInfo,
    pub output: LayerInfo,
}

impl Program {
    /// Wraps the program as a model the runner accepts.
    pub fn into_model(self) -> TfliteModel {
        TfliteModel {
            executables: vec![ExecInfo {
                exec_type: ExecutableType::STAND_ALONE,
                parameters: self.parameters,
                instruction_chunks: vec![self.chunk],
                input_layers: vec![self.input],
                output_layers: vec![self.output],
                scratch_size_bytes: 0,
            }],
        }
    }
}
