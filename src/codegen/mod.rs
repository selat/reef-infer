//! Edge TPU program generation without `edgetpu_compiler`.
//!
//! [`Asm`] assembles instructions from named fields (see [`crate::isa`]),
//! numbers tile-dispatched instructions, records address relocations and
//! frames the bitstream with `start` / `halt` / `end`. [`Program`] turns the
//! result into an [`ExecInfo`] that [`crate::runner::Device`] runs like a
//! compiled model.

pub mod conv;
pub mod elementwise;
pub mod fc;

use crate::executable_generated::platforms::darwinn::{Description, ExecutableType, Position};
use crate::isa::{self, IsaError};
use crate::model::{ChunkInfo, ExecInfo, LayerInfo, OffsetInfo, OutputLayoutInfo, TfliteModel};

pub const OP_TENSOR: u8 = 0x01;
pub const OP_RING_PRODUCER: u8 = 0x10;
pub const OP_RING_CONSUMER: u8 = 0x11;
/// ringConsumer variant the compiler uses for a weight multicast.
pub const OP_RING_CONSUMER2: u8 = 0x12;
pub const OP_NARROW_TO_WIDE: u8 = 0x13;
pub const OP_WIDE_TO_NARROW: u8 = 0x14;
/// Mesh moves between neighbouring tiles (halo exchange): data goes to
/// the tile to the north / south / west / east (0x17 also fills).
pub const OP_MESH_S: u8 = 0x15;
pub const OP_MESH_W: u8 = 0x16;
pub const OP_MESH_N: u8 = 0x17;
pub const OP_MESH_E: u8 = 0x18;
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
/// `thr5` is the compiler's `scsync.thr5`: the outfeeds (16 after 16, 2
/// after one), plus the rows of an on-chip loopback (compiled dwconv: 16 + 6
/// -> 22). `thr6` (0 for none) waits on counter 6 for rows that went tile ->
/// tiles: loopback rows (dwconv 6) or a weight multicast's words (conv 1x1
/// Cout 32: 7).
pub fn epilogue(a: &mut Asm, thr5: i64, thr6: i64) -> Result<(), IsaError> {
    sync(a, 0x1ffff, 0x3ff, false)?;
    let (mask, counters) = if thr6 > 0 {
        (0x60, 0x60000)
    } else {
        (0x20, 0x20000)
    };
    let mut f = vec![
        ("tiles", ALL_TILES),
        ("smask", mask),
        ("thr5", thr5),
        ("wait", mask),
        ("sunits", 0xc),
    ];
    if thr6 > 0 {
        f.push(("thr6", thr6));
    }
    a.emit(OP_SCSYNC, &f)?;
    sync(a, counters, 0x3ff, true)?;
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

/// PARAMETER_CACHING program: streams `bytes` of parameters from the host
/// into tile 0 over the parameter path, landing as `consumer` (tile 0's
/// ringConsumer fields) describes. `f438` is the infeed's 16-channel group
/// count - 1.
pub fn param_caching(
    a: &mut Asm,
    bytes: usize,
    f438: i64,
    consumer: &[(&str, i64)],
) -> Result<(), IsaError> {
    use Description as D;
    use Position as P;
    let units = (bytes / 64) as i64; // the parameter stream moves 64 B units
    a.emit(
        OP_SYNC,
        &[
            ("tiles", ALL_TILES),
            ("b43", 1),
            ("counters", 0x7ffff),
            ("units", 0x3ff),
            ("signal", 1),
        ],
    )?;
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
    // descriptor {s3 = s1, s2 = s0, bytes, TAG_PARAMS} via s4..s7
    mov_reg(a, 3, 1)?;
    mov_reg(a, 2, 0)?;
    param_descriptor(a, bytes as i64, TAG_PARAMS)?;
    let mut pop = vec![
        ("d0_stride", 1),
        ("d0_limit", 0x3ff),
        ("d1_stride", 0xfc01),
        ("d2_stride", 0xfc01),
        ("k121", 1),
        ("k124", 1),
        ("par136", 1),
        ("count_m1", units - 1),
        ("k157", 1),
        ("count_neg", 0x400 - units),
        ("k174", 1),
        ("par177", 1),
        ("k178", 1),
        ("rows_neg", 0x8000 - units / 2),
        ("k199", 1),
        ("k215", 1),
    ];
    if units % 2 == 0 {
        pop.push(("b183", 1));
    }
    a.emit(OP_POP, &pop)?;
    a.emit(OP_RING_CONSUMER, consumer)?;
    a.emit(
        OP_INFEED,
        &[
            ("d0_stride", 1),
            ("d0_limit", 0x3ff),
            ("d1_stride", 0x1ffc01),
            ("d2_stride", 0x1ffc01),
            ("k157", 1),
            ("k160", 1),
            ("par172", 1),
            ("count_m1", units - 1),
            ("k203", 1),
            ("count_neg", 0x400 - units),
            ("k311", 1),
            ("par314", 1),
            ("pop_wait", 1),
            ("k336", 1),
            ("k352", 1),
            ("k392", 1),
            ("tiles", 1),
            ("k434", 0xf),
            ("f438", f438),
        ],
    )?;
    a.emit(
        OP_SYNC,
        &[
            ("tiles", ALL_TILES),
            ("b43", 1),
            ("units", 0x3ff),
            ("signal", 1),
            ("f109", 3),
        ],
    )?;
    wait_tiles(a)?;
    a.mov(2, 0)?;
    a.mov(3, 0)?;
    param_descriptor(a, 0, TAG_DONE)
}

/// Descriptor `{s3, s2, len, tag}` built in s4..s7 (the caching program's
/// register choice; [`descriptor_send`] uses s6..s9).
fn param_descriptor(a: &mut Asm, len: i64, tag: i64) -> Result<(), IsaError> {
    mov_reg(a, 4, 3)?;
    mov_reg(a, 5, 2)?;
    a.emit(OP_SCALAR, &[("v_op", 0xa), ("vs_reg", 4)])?;
    a.emit(
        OP_SCALAR,
        &[
            ("v_op", 0xa),
            ("v_offset", 1),
            ("vs_reg", 5),
            ("s_op", S_MOVI),
            ("s_x", 6),
            ("imm_scalar", len),
        ],
    )?;
    a.emit(
        OP_SCALAR,
        &[
            ("v_op", 0xa),
            ("v_offset", 2),
            ("vs_reg", 6),
            ("s_op", S_MOVI),
            ("s_x", 7),
            ("imm_scalar", tag),
        ],
    )?;
    a.emit(OP_SCALAR, &[("v_op", 0xa), ("v_offset", 3), ("vs_reg", 7)])?;
    a.emit(OP_SCALAR, &[("v_op", 0xc)])?;
    Ok(())
}

/// Sends the completion event (tag 4, no address).
pub fn done(a: &mut Asm) -> Result<(), IsaError> {
    a.mov(4, 0)?;
    a.mov(5, 0)?;
    descriptor_send(a, 0, TAG_DONE)
}

/// One tile's 4x4 pixel patch of a `cin`-channel uint8 tensor, stored
/// pixel-major (`[y][x][cin]`) with a halo of `top` / `left` pixels before
/// and `bottom` / `right` after it (a convolution's padding).
#[derive(Debug, Clone, Copy)]
pub struct Patch {
    pub cin: i64,
    pub top: i64,
    pub bottom: i64,
    pub left: i64,
    pub right: i64,
}

impl Patch {
    pub fn plain(cin: usize) -> Patch {
        Patch {
            cin: cin as i64,
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        }
    }
    /// 4-byte words per pixel.
    pub fn cq(&self) -> i64 {
        self.cin / 4
    }
    /// Padded width / height in pixels.
    pub fn pw(&self) -> i64 {
        4 + self.left + self.right
    }
    pub fn ph(&self) -> i64 {
        4 + self.top + self.bottom
    }
    /// Unpadded bytes (what the host sends per tile).
    pub fn tb(&self) -> i64 {
        16 * self.cin
    }
    /// Narrow bytes the infeed lands per tile: at least one 256 B wide word.
    pub fn lb(&self) -> i64 {
        self.tb().max(256)
    }
    /// Padded bytes.
    pub fn tp(&self) -> i64 {
        self.pw() * self.ph() * self.cin
    }
    /// Byte offset of interior pixel (y, x).
    pub fn at(&self, y: i64, x: i64) -> i64 {
        ((y + self.top) * self.pw() + x + self.left) * self.cin
    }
}

/// The copy/reformat TensorOp (`dp_mode` 4) that moves a tile's input from
/// where the infeed lands it (narrow `tp`) into the padded compute layout
/// (narrow 0): the first copy writes rows 0-2, the second row 3.
/// `par_base` is the low byte of the constants' wide address. Formulas fit
/// element-wise (32 ch, no halo, constants at 0x207c) and conv k = 1..7
/// (16 ch).
pub fn copy_op(a: &mut Asm, p: &Patch, par_base: i64, first: bool) -> Result<(), IsaError> {
    let (tb, tp, cq, pw) = (p.tb(), p.tp(), p.cq(), p.pw());
    let w = tb / 64; // 64 B units per tile
    let (in_base, out_base, out_inc3, out_inc4, sync0) = if first {
        (
            tp,
            p.at(0, 0),
            tp / 4 - 2 * pw * cq - (4 * cq - 1),
            -(12 * cq - 1) - 2 * cq * (pw - 4),
            tb / 16,
        )
    } else {
        (
            tp + p.lb(),
            p.at(3, 0),
            tp / 4 - (4 * cq - 1),
            -(4 * cq - 1),
            13 * tb / 64,
        )
    };
    let mut f: Vec<(&str, i64)> = vec![
        ("tile_mask", ALL_TILES),
        ("loop1", w - 1),
        ("loop2", 3),
        ("in_base", in_base),
        ("in_inc0", 1),
        ("in_cnt0", w - 1),
        ("in_inc1", 1),
        ("in_cnt1", 3),
        ("in_mode7", 3),
        ("in_tflags", 0x80),
        ("in_twait", p.lb() / 64),
        ("out_base", out_base),
        ("out_inc0", 1),
        ("out_cnt0", w - 1),
        ("out_inc1", 1),
        ("out_cnt1", 3),
        ("out_inc2", pw * cq - (4 * cq - 1)),
        ("out_inc3", out_inc3),
        ("out_mode7", 3),
        ("out_tflags", 3),
        ("par_base", par_base),
        ("par_sel", 1),
        ("par_cnt0", w - 1),
        ("psum_cnt0", w - 1),
        ("psum_tflags", 0x800),
        ("cfg0", 5),
        ("cfg1", 0x6b),
        ("sync0", sync0),
        ("sync1", 0x4000 + tb / 16),
        ("cfg2", 6),
        ("dp_mode", 4),
        ("out_ch_last", 3),
        ("out_ch", 3),
        ("mult_bits", 1.0f32.to_bits() as i64),
        ("clamp_max_bits", f32::INFINITY.to_bits() as i64),
        ("clamp_min_bits", f32::NEG_INFINITY.to_bits() as i64),
    ];
    for n in [
        "in_inc2", "in_inc3", "in_inc4", "in_inc5", "in_inc6", "in_inc7",
    ] {
        f.push((n, -(tb / 16 - 1)));
    }
    for n in ["out_inc4", "out_inc5", "out_inc6", "out_inc7"] {
        f.push((n, out_inc4));
    }
    if first {
        f.extend([
            ("loop3", 2),
            ("in_cnt2", 2),
            ("out_cnt2", 2),
            ("par_cnt2", 3),
            ("par_cnt3", 2),
            ("psum_cnt2", 3),
            ("psum_cnt3", 2),
        ]);
    } else {
        f.extend([("par_cnt2", 3), ("psum_cnt2", 3)]);
    }
    if tb < 256 {
        // under one wide word per tile
        f.extend([("par_hmode", 1), ("psum_hmode", 1)]);
    }
    a.emit(OP_TENSOR, &f)?;
    Ok(())
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
