//! Global AVERAGE_POOL_2D over an 8x8x64 uint8 tensor (ResNet-8's head),
//! with no parameters: a mesh reduction into tile 0.
//!
//! Follows `edgetpu_compiler` 16.0's program (`pool_C64_H8_W8_opavg_padvalid
//! _size8_stride8` in the ISA corpus):
//!
//! 1. The band loader ([`super::band`]) gives every tile its 2x2 patch
//!    (256 B) at narrow 0x1000, staged at narrow 0.
//! 2. Column gather: each row-0 tile copies its own patch to 0x1100 (through
//!    wide 0x2034) and receives the three tiles below it over the mesh
//!    (northward) at 0x1200, 0x1300, 0x1400; tiles in between forward
//!    through a one-slot buffer at 0x1500. The four blocks are the column's
//!    8 image rows of 2 pixels.
//! 3. Row gather: tile 0 spreads its own column block into an 8x8 image at
//!    narrow 0 (through wide 0x2034), and receives tiles 1-3's blocks
//!    (westward, forwarded through 0x1510) at x = 2, 4, 6 of every row.
//! 4. Tile 0 regroups the image into wide words of 4 pixels (as the max
//!    pool does, at wide 0x2040), and one TensorOp with an 8x8 window
//!    (`loop0` / `loop1`, the pool op's `cfg0`) multiplies each pixel on its
//!    parameter path minus `w_zp` by the input path's constant word 1
//!    (mesh-filled at 0x1520) and sums the 64 pixels per channel:
//!    `out = sat_u8(round_half_even(clamp(sum (x - w_zp) mult, lo, hi)) + out_zp)`.
//! 5. Tile 0 outfeeds its 64 bytes.
//!
//! The compiler uses `w_zp = out_zp` = the zero point and `mult = 1 / 64`,
//! which rounds ties to even. [`GlobalAvgPool::tflite`] instead centres on
//! 128 (an int8 model's raw values) and takes the smallest f32 multiplier
//! above 1 / 64 that reproduces TFLite's rounding (half away from zero) for
//! every possible sum, so the result is TFLite's exactly.

use super::band::{self, Grid};
use super::*;

/// Mesh move west (data goes to the tile to the west).
const MESH_W: u8 = OP_MESH_W;

#[derive(Debug, Clone)]
pub struct GlobalAvgPool {
    pub height: usize,
    pub width: usize,
    pub c: usize,
    /// Value subtracted from each input before summing (`w_zp`) and added
    /// back to the output (`out_zp`).
    pub center: i32,
    /// Multiplier on the sum.
    pub mult: f32,
    /// Input and output quantization (equal, as TFLite requires).
    pub quant: Quant,
}

impl GlobalAvgPool {
    /// The pool TFLite computes on an int8 model (`int8` true: sums the raw
    /// int8 values, centre 128 in uint8) or a uint8 one (centre 0), with an
    /// exact multiplier: `round_half_even(sum m)` equals TFLite's `sum / n`
    /// rounded half away from zero for every possible sum.
    pub fn tflite(
        height: usize,
        width: usize,
        c: usize,
        quant: Quant,
        int8: bool,
    ) -> Result<GlobalAvgPool, String> {
        let n = (height * width) as i64;
        let center = if int8 { 128 } else { 0 };
        let (lo, hi) = (-center as i64 * n, (255 - center as i64) * n);
        let mult = exact_multiplier(n, 1, lo..=hi, (-center) as f32, (255 - center) as f32)
            .ok_or(format!("average pool over {n} pixels: no exact multiplier"))?;
        let p = GlobalAvgPool {
            height,
            width,
            c,
            center,
            mult,
            quant,
        };
        p.check()?;
        Ok(p)
    }

    pub fn check(&self) -> Result<(), String> {
        if (self.height, self.width, self.c) != (8, 8, 64) {
            return Err(format!(
                "global average pool {}x{}x{}: only 8x8x64",
                self.height, self.width, self.c
            ));
        }
        if !(0..=255).contains(&self.center) {
            return Err(format!("centre {} outside 0..=255", self.center));
        }
        Ok(())
    }

    /// Clamp bounds around the centre: the uint8 range.
    fn clamp(&self) -> (f32, f32) {
        ((-self.center) as f32, (255 - self.center) as f32)
    }

    /// Host reference (bit-exact with the TPU): y-x-z in, `c` bytes out.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        let (lo, hi) = self.clamp();
        (0..self.c)
            .map(|z| {
                let sum: i32 = x
                    .iter()
                    .skip(z)
                    .step_by(self.c)
                    .map(|&v| v as i32 - self.center)
                    .sum();
                let v = (sum as f32 * self.mult).clamp(lo, hi).round_ties_even();
                (v as i32 + self.center) as u8
            })
            .collect()
    }

    pub fn build(&self) -> Result<Program, Box<dyn std::error::Error>> {
        self.check()?;
        Ok(self.program(None)?)
    }

    /// The tile patch a chained pool reads: the plain 2x2 pixels.
    pub fn input_patch(&self) -> Patch {
        Patch {
            rows: self.height as i64 / 4,
            cols: self.width as i64 / 4,
            ..Patch::plain(self.c)
        }
    }

    /// End (bytes) of the narrow memory the program uses itself (the
    /// image gathered on tile 0, the column blocks, forward slots and the
    /// constant).
    pub fn narrow_end(&self) -> i64 {
        0x1600
    }

    /// The program with its input already on every tile at narrow `.0`
    /// (geometry `.1`, [`GlobalAvgPool::input_patch`]) instead of from the
    /// host; the output goes to the host ("output").
    pub fn chain_program(
        &self,
        input_at: (i64, Patch),
    ) -> Result<ChunkInfo, Box<dyn std::error::Error>> {
        self.check()?;
        let (at, p) = input_at;
        if p != self.input_patch() || at < self.narrow_end() || at % 4 != 0 {
            return Err(format!(
                "chained average pool: input {p:?} at {at:#x}, need {:?} above {:#x}",
                self.input_patch(),
                self.narrow_end()
            )
            .into());
        }
        Ok(self.program(Some(at))?.chunk)
    }

    /// `input_at`: the patch's narrow address when chained (`None`: from
    /// the host).
    fn program(&self, input_at: Option<i64>) -> Result<Program, IsaError> {
        const IN_NAME: &str = "input";
        const OUT_NAME: &str = "output";
        let g = Grid::new(self.height, self.width, self.c);
        let b = g.b(); // 256 B per tile
        let bw = b / 4; // its words
        // Narrow (bytes): staging at 0, the patch at `patch`, then the
        // column blocks, forward slots and the constant 1. Wide: the
        // gather row, constants, staging and regrouped image near the top.
        let (patch, col, fwd_n, fwd_w, one) = (0x1000, 0x1100, 0x1500, 0x1510, 0x1520);
        let (const_n, const_w, stage_w, gather_w, image_w) =
            (0x400, 0x203c, 0x2070, 0x2034, 0x2040);
        let row_words = self.width as i64 * g.cw(); // one image row
        let col_words = g.tw * g.cw(); // one tile column's pixels in a row

        // where the tiles' patches are: loaded at `patch`, or chained
        let src = input_at.unwrap_or(patch);

        let mut a = Asm::new();
        prologue(&mut a)?;
        if input_at.is_none() {
            band::constants(&mut a, const_n, const_w)?;
            band::copy_op(&mut a, &g, const_w & 0xff, true, 0, patch)?;
            band::load(&mut a, &g, IN_NAME, 0, stage_w)?;
            band::copy_op(&mut a, &g, const_w & 0xff, false, 0, patch)?;
            sync(&mut a, 0x1ffff, 0x3ff, false)?;
        }
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── Column gather into row 0: own patch -> `col`, row r's -> `col + r b`.
        a.emit(
            OP_NARROW_TO_WIDE,
            &[
                &[("tile_mask", 0xf), ("narrow_addr", src / 4)][..],
                &walk1(bw - 1),
                &[
                    ("wide_addr", gather_w),
                    ("w_inc0", 1),
                    ("wide_lvl_mask", 1),
                    ("sync_en0", 1),
                    ("sync_en1", 0),
                    ("sync_en2", 0),
                    ("tail_lvl", 1),
                ],
            ]
            .concat(),
        )?;
        a.emit(
            OP_WIDE_TO_NARROW,
            &[
                &[
                    ("tile_mask", 0xf),
                    ("wide_addr", gather_w),
                    ("w_inc0", 1),
                    ("wide_lvl_mask", 1),
                    ("narrow_addr", col / 4),
                ][..],
                &walk1(bw - 1),
                &[
                    ("sync_en0", 1),
                    ("sync_id", 0xd),
                    ("sync_en1", 1),
                    ("sync_val", 0x8000),
                    ("sync_en2", 1),
                    ("tail_en870", 1),
                ],
            ]
            .concat(),
        )?;
        let north = |a: &mut Asm, f: Vec<(&str, i64)>| a.emit(OP_MESH_N, &f).map(|_| ());
        let send = |tiles: i64| mesh_send(tiles, src, bw);
        let recv = |tiles: i64, at: i64| mesh_recv(tiles, at, bw);
        let fwd = |tiles: i64, hop: i64| mesh_fwd(tiles, fwd_n, bw, hop, (0xa, 0xe), b);
        north(&mut a, send(0xf0))?;
        north(&mut a, recv(0xf, col + b))?;
        north(&mut a, send(0xf00))?;
        north(&mut a, fwd(0xf0, 1))?;
        for c in 0..4 {
            north(&mut a, recv(1 << c, col + 2 * b))?;
            north(&mut a, send(0x1000 << c))?;
            north(&mut a, fwd(0x100 << c, 1))?;
            north(&mut a, fwd(0x10 << c, 2))?;
            north(&mut a, recv(1 << c, col + 3 * b))?;
        }
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── Row gather into tile 0: its own block -> x = 0..2 of every row
        // at narrow 0, tile c's -> x = 2c.
        let block = 4 * b; // a column block: 8 rows x 2 pixels
        let rows = |n_base: i64| -> Vec<(&'static str, i64)> {
            // a column block's rows (`col_words` each) at image-row stride
            let mut f = vec![("narrow_addr", n_base), ("narrow_lvl_mask", 7)];
            let n = ["n_inc0", "n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"];
            let lims = ["n_lim0", "n_lim1", "n_lim2"];
            let levels = [(1, col_words), (row_words, g.th), (g.th * row_words, 4)];
            for (k, (inc, lim)) in band::walk(&levels, 6).into_iter().enumerate() {
                f.push((n[k], inc));
                if k < 3 {
                    f.push((lims[k], lim));
                }
            }
            f
        };
        a.emit(
            OP_NARROW_TO_WIDE,
            &[
                ("tile_mask", 1),
                ("narrow_addr", col / 4),
                ("n_inc0", 1),
                ("n_lim0", bw - 1),
                ("n_inc1", 1),
                ("n_lim1", 3),
                ("n_inc2", -(block / 4 - 1)),
                ("n_inc3", -(block / 4 - 1)),
                ("n_inc4", -(block / 4 - 1)),
                ("n_inc5", -(block / 4 - 1)),
                ("narrow_lvl_mask", 3),
                ("wide_addr", gather_w),
                ("w_lim0", 3),
                ("wide_rows", 1),
                ("sync_en0", 1),
                ("sync_id", 0xb),
                ("sync_en1", 1),
                ("sync_val", 0xffff),
                ("sync_en2", 1),
                ("sync_f45", 1),
                ("tail_lvl", 1),
            ],
        )?;
        a.emit(
            OP_WIDE_TO_NARROW,
            &[
                &[
                    ("tile_mask", 1),
                    ("wide_addr", gather_w),
                    ("w_lim0", 3),
                    ("wide_rows", 1),
                ][..],
                &rows(0),
                &[
                    ("sync_en0", 1),
                    ("sync_f2", 1),
                    ("sync_id", 0xd),
                    ("sync_en1", 1),
                    ("sync_val", 0x8000),
                    ("sync_en2", 1),
                    ("tail_en870", 0),
                    ("tail_f871", 1),
                ],
            ]
            .concat(),
        )?;
        let west = |a: &mut Asm, f: Vec<(&str, i64)>| a.emit(MESH_W, &f).map(|_| ());
        let bwords = block / 4;
        let send = |tiles: i64| mesh_send(tiles, col, bwords);
        let recv = |c: i64| -> Vec<(&str, i64)> {
            // tile 0 lands tile c's block at x = c tw, row after row
            let at = 4 * c * col_words;
            let mut f = vec![("tile_mask", 1), ("i_addr", at)];
            let inc = ["i_inc0", "i_inc1", "i_inc2", "i_inc3"];
            let cnt = ["i_cnt0", "i_cnt1"];
            let levels = [(1, col_words), (row_words, self.height as i64)];
            for (k, (i, n)) in band::walk(&levels, 4).into_iter().enumerate() {
                f.push((inc[k], i));
                if k < 2 {
                    f.push((cnt[k], n));
                }
            }
            f.extend([("i_sdims", 3), ("in_mode", 3)]);
            f
        };
        let fwd = |tiles: i64, hop: i64| mesh_fwd(tiles, fwd_w, bwords, hop, (8, 0x14), block);
        west(&mut a, send(2))?;
        west(&mut a, recv(1))?;
        west(&mut a, send(4))?;
        west(&mut a, fwd(2, 1))?;
        west(&mut a, recv(2))?;
        west(&mut a, send(8))?;
        west(&mut a, fwd(4, 1))?;
        west(&mut a, fwd(2, 2))?;
        west(&mut a, recv(3))?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── The constant 1 on the input path; the image regrouped into wide
        // words of 4 pixels (the max pool's regroup, one 64-channel block).
        a.emit(
            OP_MESH_N,
            &[
                ("tile_mask", 1),
                ("i_addr", one),
                ("i_inc0", 1),
                ("fill_en", 1),
                ("fill", 1),
            ],
        )?;
        sync(&mut a, 0, 0x3ff, false)?;
        let (cw, px) = (g.cw(), self.width as i64);
        let wg = px * self.height as i64 / 4; // wide words
        let levels = [
            (cw, 4),
            (1, cw),
            (4 * cw, px / 4),
            (px * cw, self.height as i64),
        ];
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", 1),
            ("narrow_lvl_mask", 0xf),
            ("wide_addr", image_w),
            ("w_inc0", 1),
            ("w_lim0", wg - 1),
            ("w_inc1", -(wg - 1)),
            ("w_inc2", -(wg - 1)),
            ("w_inc3", -(wg - 1)),
            ("wide_lvl_mask", 1),
            ("sync_en0", 1),
            ("sync_f1", 1),
            ("sync_en1", 0),
            ("sync_val", 0x8000),
            ("tail_f799", 1),
            ("tail_lvl", 2),
            // undecoded (narrowToWide 554-756, "sync-like records"); as compiled
            ("rsv554", 0x1),
            ("rsv589", 0x1000084),
            ("rsv621", 0x21900),
            ("rsv653", 0x8540004),
            ("rsv685", 0xd0001000),
            ("rsv717", 0x400020),
            ("rsv749", 0x40),
        ];
        let n = ["n_inc0", "n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"];
        let lims = ["n_lim0", "n_lim1", "n_lim2", "n_lim3"];
        for (k, (inc, lim)) in band::walk(&levels, 6).into_iter().enumerate() {
            f.push((n[k], inc));
            if k < 4 {
                f.push((lims[k], lim));
            }
        }
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        sync(&mut a, 0, 0x3ff, false)?;

        // ── The reduction: an 8x8 window over the wide words, 64 channels.
        let (lo, hi) = self.clamp();
        let (wx, wy) = (px - 1, self.height as i64 - 1);
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", 1),
            ("loop0", wx),
            ("loop1", wy),
            ("in_base", one),
            ("in_cnt0", wx),
            ("in_cnt1", wy),
            ("in_inc2", 1),
            ("in_inc3", 1),
            ("in_tflags", 3),
            ("out_base", patch),
            ("out_inc0", 1),
            ("out_cnt0", cw - 1),
            ("out_inc1", 1),
            ("out_inc2", 1),
            ("out_inc3", 1),
            ("out_inc4", 1),
            ("out_inc5", -(cw - 1)),
            ("out_inc6", -(cw - 1)),
            ("out_inc7", -(cw - 1)),
            ("out_mode7", 3),
            ("out_tflags", 7),
            ("par_base", image_w & 0xff),
            ("par_sel", 1),
            ("par_hmode", 1),
            ("par_cnt0", wx),
            ("par_mode0", 1),
            ("par_cnt1", wy),
            ("par_mode1", 1),
            ("par_inc2", -(wg - 2)),
            ("par_mode2", 1),
            ("par_mode3", 1),
            ("par_mode4", 1),
            ("par_mode5", 1),
            ("par_inc6", -wg),
            ("par_mode6", 1),
            ("par_inc7", -wg),
            ("par_mode7", 3),
            ("par_tflags", 0xf),
            ("psum_cnt0", wg * 4 - 1),
            ("psum_tflags", 0x840),
            ("cfg0", 0x127),
            ("cfg2", 0xe),
            ("cfg3", 6),
            ("w_zp", self.center as i64),
            ("dp_mode", 4),
            ("reduce_mask", 3),
            ("out_ch_last", self.c as i64 - 1),
            ("out_ch", self.c as i64 - 1),
            ("mult_bits", self.mult.to_bits() as i64),
            ("clamp_max_bits", hi.to_bits() as i64),
            ("clamp_min_bits", lo.to_bits() as i64),
            ("out_zp", self.center as i64),
        ];
        f.retain(|&(_, v)| v != 0);
        a.emit(OP_TENSOR, &f)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Output: tile 0's `c` bytes -> wide 0x2078 -> host.
        load_output_addr(&mut a, OUT_NAME)?;
        descriptor_addr(&mut a)?;
        a.emit(
            OP_NARROW_TO_WIDE,
            &[
                &[("tile_mask", 1), ("narrow_addr", patch / 4)][..],
                &walk1(cw - 1),
                &[
                    ("wide_addr", 0x2078),
                    ("w_inc0", 1),
                    ("wide_circ", 1),
                    ("wide_rows", 1),
                    ("sync_en0", 1),
                    ("sync_f1", 1),
                    ("sync_id", 0x10),
                    ("sync_wait_lvl", 1),
                    ("sync_en1", 1),
                    ("sync_val", 0xffff),
                    ("sync_en2", 1),
                    ("sync_f45", 1),
                    ("tail_lvl", 1),
                ],
            ]
            .concat(),
        )?;
        let ob = self.c as i64;
        descriptor_send(&mut a, ob, TAG_OUTPUT)?;
        a.emit(
            OP_OUTFEED,
            &[("d0_limit", ob / 8 - 1), ("k173", 3), ("f222", 0x7f)],
        )?;
        a.emit(
            OP_RING_PRODUCER,
            &[
                ("tile_mask", 1),
                ("addr", 0x2078),
                ("inc0", 1),
                ("sdims", 1),
                ("cbuf", 1),
                ("slots", 1),
                ("mode", 3),
                ("pcfg", 0x14),
                ("r0_id", 0x51),
                ("r0_en_a", 1),
                ("r0_en_b", 1),
                ("r1_id", 0x2d),
                ("r1_val", 1),
                ("r1_en_a", 1),
                ("r1_en_b", 1),
                ("dest", 0x10000),
            ],
        )?;
        epilogue(&mut a, 1, 0)?;

        Ok(Program {
            chunk: a.finish()?,
            parameters: Vec::new(),
            extra_inputs: Vec::new(),
            input: layer(IN_NAME, (self.height, self.width, self.c), self.quant),
            output: LayerInfo {
                output_layout: Some(OutputLayoutInfo {
                    y_coordinate_to_linear_tile_id_map: vec![0],
                    x_coordinate_to_linear_tile_id_map: vec![0],
                    // tile 0 holds everything; the others are past its bytes
                    linearized_tile_byte_offset: (0..16)
                        .map(|t| if t == 0 { 0 } else { self.c as i32 })
                        .collect(),
                    x_coordinate_to_local_byte_offset: vec![0],
                    y_coordinate_to_local_y_offset: vec![0],
                    x_coordinate_to_local_y_row_size: vec![self.c as i32],
                }),
                ..layer(OUT_NAME, (1, 1, self.c), self.quant)
            },
        })
    }
}

/// The smallest f32 at or above `1 / (w n)` with which
/// `round_half_even(clamp(w s m, lo, hi))` equals `s / n` rounded half away
/// from zero (TFLite's average pool) for every sum `s` in `sums`, where `w`
/// is a constant weight the TPU multiplies each value by.
pub fn exact_multiplier(
    n: i64,
    w: i64,
    sums: std::ops::RangeInclusive<i64>,
    lo: f32,
    hi: f32,
) -> Option<f32> {
    let tflite = |s: i64| {
        if s > 0 {
            (s + n / 2) / n
        } else {
            (s - n / 2) / n
        }
    };
    let fits = |m: f32| {
        sums.clone().all(|s| {
            let v = ((w * s) as f32 * m).clamp(lo, hi).round_ties_even();
            v as i64 == tflite(s).clamp(lo as i64, hi as i64)
        })
    };
    let mut m = (1.0 / (w * n) as f64) as f32;
    for _ in 0..1000 {
        if fits(m) {
            return Some(m);
        }
        m = f32::from_bits(m.to_bits() + 1);
    }
    None
}

/// Narrow-side fields of a one-level walk of `lim + 1` words.
fn walk1(lim: i64) -> Vec<(&'static str, i64)> {
    let mut f = vec![("n_inc0", 1), ("n_lim0", lim), ("narrow_lvl_mask", 1)];
    f.extend(["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"].map(|n| (n, -lim)));
    f
}

/// A mesh send of `words` words from narrow byte `at` (outbound half).
fn mesh_send(tiles: i64, at: i64, words: i64) -> Vec<(&'static str, i64)> {
    vec![
        ("tile_mask", tiles),
        ("o_addr", at),
        ("o_inc0", 1),
        ("o_cnt0", words - 1),
        ("o_inc1", -(words - 1)),
        ("o_inc2", -(words - 1)),
        ("o_inc3", -(words - 1)),
        ("o_sdims", 1),
        ("out_mode", 1),
    ]
}

/// A mesh receive of `words` words at narrow byte `at` (inbound half).
fn mesh_recv(tiles: i64, at: i64, words: i64) -> Vec<(&'static str, i64)> {
    vec![
        ("tile_mask", tiles),
        ("i_addr", at),
        ("i_inc0", 1),
        ("i_cnt0", words - 1),
        ("i_inc1", -(words - 1)),
        ("i_inc2", -(words - 1)),
        ("i_inc3", -(words - 1)),
        ("i_sdims", 1),
        ("in_mode", 1),
    ]
}

/// A tile passing `words` words on through a one-slot buffer at narrow
/// byte `at`, the `hop`-th tile from the sender. Its two sync records (ids
/// per direction) count bytes: `bytes (hop - 1) + 4` and `bytes hop - 4`.
fn mesh_fwd(
    tiles: i64,
    at: i64,
    words: i64,
    hop: i64,
    (id0, id1): (i64, i64),
    bytes: i64,
) -> Vec<(&'static str, i64)> {
    vec![
        ("tile_mask", tiles),
        ("o_addr", at),
        ("o_cnt0", words - 1),
        ("o_slots", 1),
        ("i_addr", at),
        ("i_cnt0", words - 1),
        ("i_slots", 1),
        ("out_mode", 1),
        ("in_mode", 1),
        ("s0_id", id0),
        ("s0_val", bytes * (hop - 1) + 4),
        ("s0_en_a", 1),
        ("s0_en_b", 1),
        ("s1_id", id1),
        ("s1_val", bytes * hop - 4),
        ("s1_en_a", 1),
        ("s1_en_b", 1),
        ("s2_id", 1),
    ]
}
