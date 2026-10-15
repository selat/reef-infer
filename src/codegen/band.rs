//! The 16-tile band loader shared by the pool, average pool and add programs (the
//! compiler's element-wise layout): each tile holds a `tw x th` patch of a
//! `4 tw x 4 th x C` image, pixel after pixel from a narrow address.
//!
//! The host's image is popped once and infed a row of 4 tiles at a time
//! into wide staging slots (one image row: `tw C / 64` slots of 256 B);
//! each tile column keeps its `tw C` bytes of every row with a wideToNarrow
//! skip, landing the rows at a narrow staging address. Two copy TensorOps
//! then move the rows into the patch: the first rows `0 ..= th - 2`
//! (re-reading the staging row for each, stride 0), the second the last.
//! The copies' datapath constants are 4 one-hot words set up by mesh fills
//! and moved to wide memory ([`constants`]).

use super::*;

/// A per-tile patch: `tw x th` pixels of `c` channels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Grid {
    pub tw: i64,
    pub th: i64,
    pub c: i64,
}

impl Grid {
    pub fn new(height: usize, width: usize, c: usize) -> Grid {
        Grid {
            tw: width as i64 / 4,
            th: height as i64 / 4,
            c: c as i64,
        }
    }
    /// Words (4 B) per pixel.
    pub fn cw(&self) -> i64 {
        self.c / 4
    }
    /// Bytes per tile.
    pub fn b(&self) -> i64 {
        self.tw * self.th * self.c
    }
    /// Wide staging slots of 256 B: one image row.
    pub fn slots(&self) -> i64 {
        self.tw * self.c / 64
    }
}

/// Levels `(stride, count)` -> per level `(increment, count - 1)` for `n`
/// levels (missing levels: stride 0, count 1).
pub(crate) fn walk(levels: &[(i64, i64)], n: usize) -> Vec<(i64, i64)> {
    let mut span = 0;
    (0..n)
        .map(|k| {
            let (stride, count) = levels.get(k).copied().unwrap_or((0, 1));
            let inc = stride - span;
            span += (count - 1) * stride;
            (inc, count - 1)
        })
        .collect()
}

/// TensorOp walk fields `{p}_inc0..7` and `{p}_cnt0..{cnts - 1}`.
pub(crate) fn op_walk(
    f: &mut Vec<(&'static str, i64)>,
    p: &str,
    levels: &[(i64, i64)],
    cnts: usize,
) {
    const INC: [[&str; 8]; 2] = [
        [
            "in_inc0", "in_inc1", "in_inc2", "in_inc3", "in_inc4", "in_inc5", "in_inc6", "in_inc7",
        ],
        [
            "out_inc0", "out_inc1", "out_inc2", "out_inc3", "out_inc4", "out_inc5", "out_inc6",
            "out_inc7",
        ],
    ];
    const CNT: [[&str; 4]; 2] = [
        ["in_cnt0", "in_cnt1", "in_cnt2", "in_cnt3"],
        ["out_cnt0", "out_cnt1", "out_cnt2", "out_cnt3"],
    ];
    let i = (p == "out") as usize;
    for (k, (inc, cnt)) in walk(levels, 8).into_iter().enumerate() {
        f.push((INC[i][k], inc));
        if k < cnts {
            f.push((CNT[i][k], cnt));
        }
    }
}

/// The copies' constants: 4 one-hot words filled at narrow `const_n`
/// (bytes), moved to wide `const_w`.
pub(crate) fn constants(a: &mut Asm, const_n: i64, const_w: i64) -> Result<(), IsaError> {
    for (i, fill) in [1i64, 0x100, 0x10000, 0x1000000].into_iter().enumerate() {
        let mut f = vec![
            ("tile_mask", ALL_TILES),
            ("i_addr", const_n + 4 * i as i64),
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
            ("narrow_addr", const_n / 4),
            ("n_inc0", 1),
            ("n_lim0", 3),
            ("n_inc1", -3),
            ("n_inc2", -3),
            ("n_inc3", -3),
            ("n_inc4", -3),
            ("n_inc5", -3),
            ("narrow_lvl_mask", 1),
            ("wide_addr", const_w),
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
    Ok(())
}

/// Loads input layer `name` (16 tiles x `g.b()` bytes) through the wide
/// staging slots at `stage_w` into narrow `stage` (bytes) on every tile,
/// ending with the sync before the second copy.
pub(crate) fn load(
    a: &mut Asm,
    g: &Grid,
    name: &str,
    stage: i64,
    stage_w: i64,
) -> Result<(), IsaError> {
    let (tw, th, cw, b, s) = (g.tw, g.th, g.cw(), g.b(), g.slots());
    load_input_addr(a, name)?;
    let units8 = 2 * b; // all tiles, 8 B units
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
    descriptor_addr(a)?;
    descriptor_send(a, 16 * b, TAG_INPUT)?;
    if b > 256 || th > 2 {
        // not with one band of 2 rows of up to 256 B per tile (the compiled
        // 8x8 average pools)
        a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("counters", 0x800)])?;
    }
    // Ring -> narrow per tile column: column `col` skips its left
    // neighbours' `tw * cw` words of each image row.
    let row_words = s * 64; // words per image row
    // an image row in one slot: the rows of a band one after the other in
    // `th` slots (the compiler's form; elsewhere a circular buffer of `s`)
    let linear = s == 1;
    let wide: Vec<(&str, i64)> = if linear {
        vec![("wide_addr", stage_w), ("w_lim0", th - 1), ("wide_rows", s)]
    } else {
        vec![
            ("wide_addr", stage_w),
            ("w_inc0", 1),
            ("w_lim0", s - 1),
            ("w_inc1", -(s - 1)),
            ("w_lim1", th - 1),
            ("w_inc2", -(s - 1)),
            ("w_inc3", -(s - 1)),
            ("wide_lvl_mask", 1),
            ("wide_circ", 1),
            ("wide_rows", s),
            ("sync_wait_lvl", 1),
        ]
    };
    for col in 0..4i64 {
        let mut f: Vec<(&str, i64)> = vec![("tile_mask", 0x1111 << col)];
        f.extend(wide.iter().copied());
        f.extend([
            ("narrow_addr", stage / 4),
            ("n_inc0", 1),
            ("n_lim0", row_words - 1),
            ("n_lim1", th - 1),
            ("narrow_lvl_mask", 1),
            ("mode", 1),
            ("size64", 4 * s),
            ("sync_en0", 1),
            ("sync_id", 0xe),
            ("sync_en1", 1),
            ("sync_val", 0x8000),
            ("sync_en2", 1),
            ("sync_dec_lvl", 1),
            ("sync_dec", -1),
            ("sync_dec_mode", 3),
            ("tail_en870", 0),
            ("tail_f871", 1),
        ]);
        for n in ["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"] {
            f.push((n, -(row_words - 1)));
        }
        if col > 0 {
            let skip = tw * cw * col;
            f.extend([
                ("head_words_m1", row_words - 1 - skip),
                ("head_en", 1),
                ("tail_words", skip),
                ("skip_en0", 1),
                ("skip_base", stage / 4),
                ("skip_m1", skip - 1),
                ("skip_end", stage / 4 + skip - 1),
                ("rsv543", 1),
            ]);
        }
        a.emit(OP_WIDE_TO_NARROW, &f)?;
    }
    let per_row = b / 2; // 4 tiles x b bytes, 8 B units
    for row in 0..4i64 {
        let mut f: Vec<(&str, i64)> = if linear {
            vec![("cnt0", th - 1), ("slots", 1), ("mode", 1), ("s_id", 0xb)]
        } else {
            vec![
                ("inc0", 1),
                ("cnt0", s - 1),
                ("inc1", -(s - 1)),
                ("cnt1", th - 1),
                ("inc2", -(s - 1)),
                ("inc3", -(s - 1)),
                ("sdims", 1),
                ("cbuf", 1),
                ("slots", s),
                ("mode", 3),
                ("s_id", 0x2b),
            ]
        };
        f.extend([
            ("tile_mask", 0xf << (4 * row)),
            ("addr", stage_w),
            ("s_val", -64 * s),
            ("s_cnt", s),
            ("s_en_b", 1),
            ("s_y", 1),
        ]);
        a.emit(OP_RING_CONSUMER, &f)?;
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
    sync(a, 0, 0x3ff, false)?;
    Ok(())
}

pub(crate) fn copy_op(
    a: &mut Asm,
    g: &Grid,
    const_w: i64,
    first: bool,
    stage: i64,
    dest: i64,
) -> Result<(), IsaError> {
    let (tw, th, cw) = (g.tw, g.th, g.cw());
    let row = tw * cw; // words per patch row
    let rows = if first { th - 1 } else { 1 };
    let mut f: Vec<(&'static str, i64)> = vec![
        ("tile_mask", ALL_TILES),
        ("loop1", cw - 1),
        ("loop2", tw - 1),
        ("in_base", if first { stage } else { stage + 4 * row * 4 }),
        ("in_mode7", 3),
        ("in_tflags", 0x80),
        ("in_twait", row / 4),
        (
            "out_base",
            if first {
                dest
            } else {
                dest + 4 * row * (th - 1)
            },
        ),
        ("out_mode7", 3),
        ("out_tflags", 3),
        // the constants' wide address: its low byte in the top page
        // (`par_sel`), else in full (the compiler's)
        (
            "par_base",
            if const_w >= 0x2000 {
                const_w & 0xff
            } else {
                const_w
            },
        ),
        ("par_sel", (const_w >= 0x2000) as i64),
        ("par_cnt0", cw - 1),
        ("par_cnt2", tw - 1),
        ("psum_cnt0", cw - 1),
        ("psum_cnt2", tw - 1),
        ("psum_tflags", 0x800),
        ("cfg0", 5),
        ("cfg1", 0x6b),
        ("sync0", if first { row } else { row * (th - 1) + row / 4 }),
        ("sync1", 0x4000 + row),
        ("cfg2", 6),
        ("dp_mode", 4),
        ("out_ch_last", 3),
        ("out_ch", 3),
        ("mult_bits", 1.0f32.to_bits() as i64),
        ("clamp_max_bits", f32::INFINITY.to_bits() as i64),
        ("clamp_min_bits", f32::NEG_INFINITY.to_bits() as i64),
    ];
    // in: the staging row is re-read for every output row (stride 0)
    let in_levels: &[(i64, i64)] = if first {
        &[(1, cw), (cw, tw), (0, rows)]
    } else {
        &[(1, cw), (cw, tw)]
    };
    op_walk(&mut f, "in", in_levels, 3);
    op_walk(
        &mut f,
        "out",
        &[(1, cw), (cw, tw), (row, rows), (row * th, 1)],
        3,
    );
    if first {
        f.extend([
            ("loop3", th - 2),
            ("par_cnt3", th - 2),
            ("psum_cnt3", th - 2),
        ]);
    }
    // a count-1 level: no count field when it is the second copy
    f.retain(|&(k, v)| !(v == 0 && k.ends_with("cnt2") && !first));
    a.emit(OP_TENSOR, &f)?;
    Ok(())
}
