//! Element-wise requantization of a 1x16x16x32 uint8 tensor:
//! `out = clamp(round((x - in_zp) * mult), lo, hi) + out_zp`.
//!
//! The program layout follows `edgetpu_compiler` 16.0's RELU on a
//! 1x16x16x32 tensor (`act_in_range8p0_oprelu_shape16x16x32` in the ISA
//! corpus): the input is split over 16 tiles (512 B each), requantized in
//! place and streamed back tile by tile. With the parameters the compiler
//! would choose, the output is byte-identical to its bitstream.

use super::*;

/// Tensor shape (y, x, z) the layout is fixed to.
pub const SHAPE: (usize, usize, usize) = (16, 16, 32);
/// Tensor size: 16 tiles x 512 B.
pub const SIZE: usize = SHAPE.0 * SHAPE.1 * SHAPE.2;

/// Where outputs land: tile `4 * (y / 4) + x / 4` holds a 4x4 patch of
/// pixels, rows of 4 pixels x 32 B. Tiles are sent in order.
fn output_layout() -> OutputLayoutInfo {
    let (ny, nx, nz) = SHAPE;
    let (py, px) = (ny / 4, nx / 4); // patch size per tile
    OutputLayoutInfo {
        y_coordinate_to_linear_tile_id_map: (0..ny).map(|y| (4 * (y / py)) as i32).collect(),
        x_coordinate_to_linear_tile_id_map: (0..nx).map(|x| (x / px) as i32).collect(),
        linearized_tile_byte_offset: (0..16).map(|t| (t * SIZE / 16) as i32).collect(),
        x_coordinate_to_local_byte_offset: (0..nx).map(|x| ((x % px) * nz) as i32).collect(),
        y_coordinate_to_local_y_offset: (0..ny).map(|y| (y % py) as i32).collect(),
        x_coordinate_to_local_y_row_size: vec![(px * nz) as i32; nx],
    }
}

/// The datapath stage of a TensorOp: `uint8 - in_zp`, times `mult` (f32),
/// clamped to `[clamp_min, clamp_max]`, rounded half to even, plus `out_zp`,
/// saturated to uint8. Stage order verified on hardware (2026-10-10).
#[derive(Debug, Clone, Copy)]
pub struct Requant {
    pub in_zp: u8,
    pub mult: f32,
    pub clamp_min: f32,
    pub clamp_max: f32,
    pub out_zp: u8,
}

impl Requant {
    /// The parameters TFLite RELU/identity lowers to: `mult = s_in / s_out`
    /// computed in f32 as `s_in * (1 / s_out)`, clamp to the uint8 range
    /// shifted by `out_zp` and, for RELU, at real zero.
    pub fn from_quant(input: Quant, output: Quant, relu: bool) -> Requant {
        let out_zp = output.zero_point as f32;
        Requant {
            in_zp: input.zero_point as u8,
            mult: input.scale * (1.0 / output.scale),
            clamp_min: if relu { 0.0 } else { -out_zp },
            clamp_max: 255.0 - out_zp,
            out_zp: output.zero_point as u8,
        }
    }

    /// Host reference for one value. The clamp comes before rounding
    /// (clamp_max 40.7 gives 41), and the final add saturates.
    pub fn apply(&self, x: u8) -> u8 {
        let v = ((x as f32 - self.in_zp as f32) * self.mult).clamp(self.clamp_min, self.clamp_max);
        (v.round_ties_even() + self.out_zp as f32).clamp(0.0, 255.0) as u8
    }
}

const IN_WALK: [&str; 11] = [
    "in_inc0", "in_inc1", "in_cnt1", "in_inc2", "in_cnt2", "in_inc3", "in_inc4", "in_inc5",
    "in_inc6", "in_inc7", "in_mode7",
];
const OUT_WALK: [&str; 11] = [
    "out_inc0",
    "out_inc1",
    "out_cnt1",
    "out_inc2",
    "out_cnt2",
    "out_inc3",
    "out_inc4",
    "out_inc5",
    "out_inc6",
    "out_inc7",
    "out_mode7",
];

/// The requantizing TensorOp over each tile's 512 B (16 x 8 words).
fn requant_op(a: &mut Asm, rq: &Requant) -> Result<(), IsaError> {
    let mut f: Vec<(&str, i64)> = vec![
        ("tile_mask", ALL_TILES),
        ("loop0", 15),
        ("loop1", 7),
        ("in_tflags", 1),
        ("out_tflags", 1),
        ("par_base", 0x7c),
        ("par_sel", 1),
        ("par_cnt0", 15),
        ("par_cnt1", 7),
        ("psum_cnt0", 15),
        ("psum_cnt1", 7),
        ("psum_tflags", 0x800),
        ("cfg0", 0x65),
        ("cfg1", 0x4d),
        ("sync0", 0x4000),
        ("sync1", 0x4000),
        ("cfg2", 6),
        ("in_zp", rq.in_zp as i64),
        ("dp_mode", 1),
        ("out_ch_last", 3),
        ("out_ch", 3),
        ("mult_bits", rq.mult.to_bits() as i64),
        ("clamp_max_bits", rq.clamp_max.to_bits() as i64),
        ("clamp_min_bits", rq.clamp_min.to_bits() as i64),
        ("out_zp", rq.out_zp as i64),
    ];
    // in/out walk: 8 words, then 16 rows of 8 words, then rewind.
    for n in [IN_WALK, OUT_WALK] {
        f.extend([(n[0], 1), (n[1], 8), (n[2], 15), (n[3], -119), (n[4], 7)]);
        f.extend(n[5..10].iter().map(|&k| (k, -127)));
        f.push((n[10], 3));
    }
    a.emit(OP_TENSOR, &f)?;
    Ok(())
}

/// Builds the program. `rsv543` sets the unexplained `wideToNarrow` bit 543
/// the compiler sets alongside `skip_en0`.
pub fn build(rq: &Requant, input: Quant, output: Quant, rsv543: bool) -> Result<Program, IsaError> {
    build_inner(rq, input, output, rsv543, None)
}

/// [`build`] plus an on-chip loopback before the output stage: tile `src`
/// sends its result at wide 0x2078 over the ring to scalar memory, and the
/// scalar core multicasts it to all 16 tiles (verified bit-exact on
/// hardware, `examples/broadcast_hw.rs`). The output is still each tile's
/// own result. The loopback fields are the compiler's for a 2-row C8 3x3
/// depthwise program, so the receiving ringConsumer splits each 192 B row
/// into 3 aux units (0x1fe8 + 4 per 16 B) and 9 slot units (0x1fa0 + ...):
/// only the first 384 B of the 512 B arrive, at [`bcast_unit_addr`].
pub fn build_broadcast(
    rq: &Requant,
    input: Quant,
    output: Quant,
    src: usize,
) -> Result<Program, IsaError> {
    build_broadcast_parts(rq, input, output, src, ALL_PARTS)
}

/// Pieces of [`build_broadcast`], for bisecting on hardware.
pub mod part {
    pub const CONSUMER: u8 = 1;
    pub const INFEED: u8 = 2;
    pub const OUTFEED: u8 = 4;
    pub const PRODUCER: u8 = 8;
    pub const BARRIER_PRE: u8 = 16;
    pub const BARRIER_POST: u8 = 32;
    /// Epilogue waits for the loopback rows (counters 5 and 6).
    pub const EPILOGUE: u8 = 64;
}
const ALL_PARTS: u8 = 0x7f;

/// [`build_broadcast`] with only the pieces in `parts` (see [`part`]).
#[doc(hidden)]
pub fn build_broadcast_parts(
    rq: &Requant,
    input: Quant,
    output: Quant,
    src: usize,
    parts: u8,
) -> Result<Program, IsaError> {
    assert!(src < 16);
    build_inner(rq, input, output, true, Some((src as i64, parts)))
}

/// Rows moved by the loopback (192 B each with this consumer).
const BCAST_WORDS: i64 = 2;

/// Wide address at which a tile receives 16 B unit `u` (< 24) of a
/// [`build_broadcast`] loopback: per 192 B row, 3 aux units then 9 slots.
pub fn bcast_unit_addr(u: usize) -> i64 {
    assert!(u < 24);
    let (row, k) = (u / 12, u % 12);
    if k < 3 {
        0x1fe8 + 4 * (3 * row + k) as i64
    } else {
        0x1fa0 + 4 * (9 * row + k - 3) as i64
    }
}
/// Wide address the result sits at and the broadcast lands at.
const OUT_WIDE: i64 = 0x2078;

/// Tile `src` (wide [`OUT_WIDE`]) -> scalar memory -> all tiles. Each
/// tile's ringConsumer is queued first; the scalar core then arms the
/// loopback infeed, the loopback outfeed, and tile `src`'s ringProducer.
fn broadcast(a: &mut Asm, src: i64, parts: u8) -> Result<(), IsaError> {
    let rows_m1 = BCAST_WORDS - 1;
    if parts & part::CONSUMER != 0 {
        // The compiler's 2-row (C8 3x3 dwconv) loopback consumer, verbatim:
        // a 9-slot circular buffer at 0x1fa0 plus aux rows at 0x1fe9 /
        // 0x1ffc. `aux_en0` and `rsv286` are required (clearing either
        // hangs the compiled program).
        a.emit(
            OP_RING_CONSUMER,
            &[
                ("tile_mask", ALL_TILES),
                ("addr", 0x1fa0),
                ("inc0", 1),
                ("cnt0", 8),
                ("inc1", -8),
                ("inc2", -8),
                ("cnt2", 1),
                ("inc3", -8),
                ("sdims", 1),
                ("cbuf", 1),
                ("slots", 9),
                ("aux_en0", 1),
                ("aux_addr0", 0x1fe9),
                ("aux_addr1", 0x1ffc),
                ("aux_en1", 1),
                ("mode", 3),
                ("s_id", 0x21),
                ("s_val", -1),
                ("s_en_a", 1),
                ("s_en_b", 1),
                ("s_y", 1),
                ("rsv286", 2),
                ("rsv319", 1),
            ],
        )?;
    }
    if parts & part::INFEED != 0 {
        a.emit(
            OP_INFEED,
            &[
                ("loop_en", 1),
                ("d0_stride", 1),
                ("d0_limit", 0x3f),
                ("d1_stride", 0x1fffc1),
                ("d1_limit", rows_m1),
                ("d2_stride", 0x1fffc1),
                ("k157", 1),
                ("k160", 1),
                ("loop_k168", 1),
                ("count_m1", 0x1f),
                ("k203", 1),
                ("count_neg", 0x20),
                ("loop_k308", 3),
                ("k311", 1),
                ("par314", 1),
                ("loop_k315", 3),
                ("pop_wait", 1),
                ("k336", 1),
                ("k352", 1),
                ("k392", 1),
                ("tiles", ALL_TILES),
                ("k434", 3),
            ],
        )?;
    }
    if parts & part::OUTFEED != 0 {
        a.emit(
            OP_OUTFEED,
            &[
                ("d0_stride", 1),
                ("d0_limit", 0x3f),
                ("d1_stride", 0xffc1),
                ("d1_limit", rows_m1),
                ("d2_stride", 0xffc1),
                ("d3_stride", 0x809),
                ("k173", 1),
                ("loop_cfg", 0x11),
                ("loop_row", 0x7fe),
                ("loop_en", 1),
                ("f222", 0x3f),
            ],
        )?;
    }
    if parts & part::PRODUCER != 0 {
        a.emit(
            OP_RING_PRODUCER,
            &[
                ("tile_mask", 1 << src),
                ("addr", OUT_WIDE),
                ("inc0", 1),
                ("inc1", 1),
                ("cnt1", rows_m1),
                ("inc2", -rows_m1),
                ("inc3", -rows_m1),
                ("sdims", 3),
                ("pcfg", 0xc),
                ("r0_id", 0x31),
                ("r0_val", -1),
                ("r0_en_a", 1),
                ("r0_en_b", 1),
                ("r1_id", 0x32),
                ("r1_en_a", 1),
                ("r1_en_b", 1),
                ("to_c1", 1),
                ("dest", 0x10000),
            ],
        )?;
    }
    Ok(())
}

fn build_inner(
    rq: &Requant,
    input: Quant,
    output: Quant,
    rsv543: bool,
    bcast: Option<(i64, u8)>,
) -> Result<Program, IsaError> {
    const IN_NAME: &str = "input";
    const OUT_NAME: &str = "output";
    let mut a = Asm::new();

    prologue(&mut a)?;

    // ── Constant setup: fill narrow words 0x600..0x60c with one-hot bytes,
    // copy them to wide row 0x207c and run the first copy op. Meaning not
    // decoded; reproduced as the compiler emits it.
    for (i, fill) in [1i64, 0x100, 0x10000, 0x1000000].into_iter().enumerate() {
        let mut f = vec![
            ("tile_mask", ALL_TILES),
            ("i_addr", 0x600 + 4 * i as i64),
            ("i_inc0", 1),
            ("fill_en", 1),
            ("fill", fill),
        ];
        if i == 3 {
            f.push(("in_mode", 3));
        }
        a.emit(OP_MESH_N, &f)?;
    }
    a.emit(
        OP_NARROW_TO_WIDE,
        &[
            ("tile_mask", ALL_TILES),
            ("narrow_addr", 0x180),
            ("n_inc0", 1),
            ("n_lim0", 3),
            ("n_inc1", -3),
            ("n_inc2", -3),
            ("n_inc3", -3),
            ("n_inc4", -3),
            ("n_inc5", -3),
            ("narrow_lvl_mask", 1),
            ("wide_addr", 0x207c),
            ("w_inc0", 1),
            ("sync_en0", 1),
            ("sync_f1", 1),
            ("sync_id", 5),
            ("sync_wait_lvl", 1),
            ("sync_en1", 1),
            ("sync_en2", 1),
            ("tail_lvl", 1),
        ],
    )?;
    a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("units", 0x20)])?;
    copy_op(&mut a, &Patch::plain(32), 0x7c, true)?;

    // ── Input: ask the host for SIZE bytes, pop them into the staging
    // buffer, and infeed 2 KiB to each row of 4 tiles over the ring.
    load_input_addr(&mut a, IN_NAME)?;
    let units8 = (SIZE / 8) as i64; // pop counts 8-byte units
    a.emit(
        OP_POP,
        &[
            ("stream", 1),
            ("d0_stride", 1),
            ("d0_limit", 0x3fff),
            ("d1_stride", 0xc001),
            ("d2_stride", 0xc001),
            ("k121", 1),
            ("k124", 1),
            ("av140", 1),
            ("count_m1", units8 - 1),
            ("k157", 1),
            ("count_neg", 0x4000 - units8),
            ("k174", 1),
            ("k178", 1),
            ("k199", 1),
            ("k215", 1),
        ],
    )?;
    descriptor_addr(&mut a)?;
    descriptor_send(&mut a, SIZE as i64, TAG_INPUT)?;
    a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("counters", 0x800)])?;

    // Ring -> narrow for each tile column; column c skips the first 32*c words.
    for col in 0..4i64 {
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", 0x1111 << col),
            ("wide_addr", 0x206c),
            ("w_inc0", 1),
            ("w_lim0", 1),
            ("w_inc1", -1),
            ("w_lim1", 3),
            ("w_inc2", -1),
            ("w_inc3", -1),
            ("wide_lvl_mask", 1),
            ("wide_circ", 1),
            ("wide_rows", 2),
            ("narrow_addr", 0x80),
            ("n_inc0", 1),
            ("n_lim0", 0x7f),
            ("n_lim1", 3),
            ("narrow_lvl_mask", 1),
            ("mode", 1),
            ("size64", 8),
            ("sync_en0", 1),
            ("sync_id", 0xe),
            ("sync_wait_lvl", 1),
            ("sync_en1", 1),
            ("sync_val", 0x8000),
            ("sync_en2", 1),
            ("sync_dec_lvl", 1),
            ("sync_dec", -1),
            ("sync_dec_mode", 3),
            ("tail_en870", 0),
            ("tail_f871", 1),
        ];
        for n in ["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"] {
            f.push((n, -127));
        }
        if col > 0 {
            let skip = 32 * col;
            f.extend([
                ("head_words_m1", 127 - skip),
                ("head_en", 1),
                ("tail_words", skip),
                ("skip_en0", 1),
                ("skip_base", 0x80),
                ("skip_m1", skip - 1),
                ("skip_end", 0x80 + skip - 1),
            ]);
            if rsv543 {
                f.push(("rsv543", 1));
            }
        }
        a.emit(OP_WIDE_TO_NARROW, &f)?;
    }
    for row in 0..4i64 {
        a.emit(
            OP_RING_CONSUMER,
            &[
                ("tile_mask", 0xf << (4 * row)),
                ("addr", 0x206c),
                ("inc0", 1),
                ("cnt0", 1),
                ("inc1", -1),
                ("cnt1", 3),
                ("inc2", -1),
                ("inc3", -1),
                ("sdims", 1),
                ("cbuf", 1),
                ("slots", 2),
                ("mode", 3),
                ("s_id", 0x2b),
                ("s_val", -128),
                ("s_cnt", 2),
                ("s_en_b", 1),
                ("s_y", 1),
            ],
        )?;
        let per_row = units8 / 4;
        a.emit(
            OP_INFEED,
            &[
                ("stream", 1),
                ("buf_off", per_row * row),
                ("d0_stride", 1),
                ("d0_limit", 0x3fff),
                ("d1_stride", 0x1fc001),
                ("d2_stride", 0x1fc001),
                ("k157", 1),
                ("k160", 1),
                ("av176", 1),
                ("count_m1", per_row - 1),
                ("k203", 1),
                ("count_neg", 0x4000 - per_row),
                ("k311", 1),
                ("pop_wait", per_row * row + 1),
                ("k336", 1),
                ("k352", 1),
                ("k392", 1),
                ("tiles", 0xf << (4 * row)),
                ("k434", 0xf),
                ("f438", 3),
            ],
        )?;
    }
    sync(&mut a, 0, 0x3ff, false)?;

    // ── Compute.
    copy_op(&mut a, &Patch::plain(32), 0x7c, false)?;
    sync(&mut a, 0x1ffff, 0x3ff, false)?;
    sync(&mut a, 0x1ffff, 0x3ff, false)?;
    requant_op(&mut a, rq)?;
    sync(&mut a, 0x1ffff, 0x3ff, false)?;
    sync(&mut a, 0, 0x3ff, true)?;
    wait_tiles(&mut a)?;

    // ── Output: narrow -> wide row 0x2078 on every tile, then each tile in
    // turn produces its 512 B onto the ring and the scalar core outfeeds it.
    load_output_addr(&mut a, OUT_NAME)?;
    // the compiler interleaves the narrow -> wide copy with the descriptor
    descriptor_addr(&mut a)?;
    a.emit(
        OP_NARROW_TO_WIDE,
        &[
            ("tile_mask", ALL_TILES),
            ("n_inc0", 1),
            ("n_lim0", 7),
            ("n_inc1", 1),
            ("n_lim1", 3),
            ("n_inc2", 1),
            ("n_lim2", 3),
            ("n_inc3", -127),
            ("n_inc4", -127),
            ("n_inc5", -127),
            ("narrow_lvl_mask", 7),
            ("wide_addr", OUT_WIDE),
            ("w_lim0", 1),
            ("wide_rows", 1),
            ("sync_en0", 1),
            ("sync_id", 0x10),
            ("sync_en1", 1),
            ("sync_val", 0xffff),
            ("sync_en2", 1),
            ("sync_f45", 1),
            ("tail_lvl", 3),
        ],
    )?;
    let parts = bcast.map_or(0, |b| b.1);
    if let Some((src, parts)) = bcast {
        if parts & part::BARRIER_PRE != 0 {
            sync(&mut a, 0, 0x3ff, true)?;
            wait_tiles(&mut a)?;
        }
        broadcast(&mut a, src, parts)?;
        if parts & part::BARRIER_POST != 0 {
            sync(&mut a, 0, 0x3ff, true)?;
            wait_tiles(&mut a)?;
        }
    }
    descriptor_send(&mut a, SIZE as i64, TAG_OUTPUT)?;
    // r0 (id 0x51) indexes the rows sent to the scalar core; loopback rows
    // come first (compiled dwconv: 6 rows, then 6..=21).
    let row0 = if parts & part::PRODUCER != 0 {
        BCAST_WORDS
    } else {
        0
    };
    for t in 0..16i64 {
        a.emit(
            OP_OUTFEED,
            &[("d0_limit", 0x3f), ("k173", 3), ("f222", 0x7f)],
        )?;
        a.emit(
            OP_RING_PRODUCER,
            &[
                ("tile_mask", 1 << t),
                ("addr", OUT_WIDE),
                ("inc0", 1),
                ("cnt1", 1),
                ("sdims", 1),
                ("cbuf", 1),
                ("slots", 1),
                ("mode", 3),
                ("pcfg", 0x14),
                ("r0_id", 0x51),
                ("r0_val", row0 + t),
                ("r0_en_a", 1),
                ("r0_en_b", 1),
                ("r1_id", 0x2d),
                ("r1_val", 1),
                ("r1_en_a", 1),
                ("r1_en_b", 1),
                ("dest", 0x10000), // bit 16: scalar core / outfeed
            ],
        )?;
    }

    // loopback rows count towards counter 5 and are awaited on counter 6
    let loop_rows = if parts & part::EPILOGUE != 0 {
        BCAST_WORDS
    } else {
        0
    };
    epilogue(&mut a, 16 + loop_rows, loop_rows)?;

    Ok(Program {
        chunk: a.finish()?,
        parameters: Vec::new(),
        input: layer(IN_NAME, SHAPE, input),
        output: LayerInfo {
            output_layout: Some(output_layout()),
            ..layer(OUT_NAME, SHAPE, output)
        },
    })
}
