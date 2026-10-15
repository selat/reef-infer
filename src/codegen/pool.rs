//! MAX_POOL_2D, 2x2 window, stride 2, valid padding, on a `H x W x C` uint8
//! tensor split over 16 tiles (each a `H/4 x W/4` patch).
//!
//! Follows `edgetpu_compiler` 16.0's program (`pool_C*_H*_W*_opmax_padvalid
//! _size2_stride2` in the ISA corpus). The input stage is the element-wise
//! program's (the RELU layout): a band loader and two copy TensorOps (rows
//! `0 ..= th - 2`, then the last row) into narrow 0, one pixel after the
//! other. Then a narrowToWide regroups the patch into wide words of 4
//! pixels x up to 64 channels, and the pool op (opcode 0x02) reads the 2x2
//! windows from there (its parameter path) and writes the outputs to narrow,
//! from where each tile outfeeds them in turn.
//!
//! All loop increments follow the walk rule (stride minus the span of the
//! inner levels). The op's datapath stage is an identity requantization:
//! `clamp(max - zp, -zp, 255 - zp) + zp`.

use super::band::{self, Grid, op_walk, walk};
use super::*;

/// Opcode of the pool op (a TensorOp variant with window loops).
pub const OP_POOL: u8 = 0x02;

#[derive(Debug, Clone)]
pub struct MaxPool {
    pub height: usize,
    pub width: usize,
    pub c: usize,
    /// Input and output quantization (TFLite keeps it through MAX_POOL_2D).
    pub quant: Quant,
}

impl MaxPool {
    pub fn check(&self) -> Result<(), String> {
        let ok = matches!(self.height, 16 | 32)
            && matches!(self.width, 16 | 32)
            && self.c % 32 == 0
            && (32..=128).contains(&self.c)
            && self.out_tile_bytes() <= 512;
        if !ok {
            return Err(format!(
                "max pool {}x{}x{}: need H, W in 16 / 32, C 32 / 64 / 128, <= 512 B out per tile",
                self.height, self.width, self.c
            ));
        }
        Ok(())
    }

    /// Patch per tile: (columns, rows).
    fn patch(&self) -> (i64, i64) {
        (self.width as i64 / 4, self.height as i64 / 4)
    }

    fn out_tile_bytes(&self) -> usize {
        self.width * self.height * self.c / 64
    }

    pub fn out_shape(&self) -> (usize, usize, usize) {
        (self.height / 2, self.width / 2, self.c)
    }

    /// Host reference over a y-x-z input.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        let (h, w, c) = (self.height, self.width, self.c);
        let mut out = Vec::with_capacity(h * w * c / 4);
        for y in (0..h).step_by(2) {
            for xx in (0..w).step_by(2) {
                for z in 0..c {
                    let at = |dy: usize, dx: usize| x[((y + dy) * w + xx + dx) * c + z];
                    out.push(at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1)));
                }
            }
        }
        out
    }

    pub fn build(&self) -> Result<Program, Box<dyn std::error::Error>> {
        self.check()?;
        Ok(self.program()?)
    }

    fn program(&self) -> Result<Program, IsaError> {
        const IN_NAME: &str = "input";
        const OUT_NAME: &str = "output";
        let (tw, th) = self.patch();
        let c = self.c as i64;
        let cw = c / 4; // words per pixel
        let b = tw * th * c; // input bytes per tile
        let bo = b / 4; // output bytes per tile
        let (ow, oh) = (tw / 2, th / 2);
        let s = tw * c / 64; // staging slots of 256 B: one image row
        let cwb = cw.min(16); // channel words per wide word (<= 64 channels)
        let h = cw / cwb; // 64-channel blocks
        let g = tw / 4; // 4-pixel groups per patch row
        let wg = tw * th / 4; // wide words per channel block
        // Wide memory, top down: the regrouped patch, one image row of
        // staging, the 4 constant words. Narrow: patch at 0, staging at
        // `b` (two 4-row bands), constants after.
        let pool_w = 0x2080 - 4 * wg * h;
        let stage_w = pool_w - 8 * s;
        let const_w = stage_w - 4;
        let const_n = b + 8 * tw * c;
        let zp = self.quant.zero_point as i64;

        let grid = Grid::new(self.height, self.width, self.c);
        let mut a = Asm::new();
        prologue(&mut a)?;
        band::constants(&mut a, const_n, const_w)?;
        band::copy_op(&mut a, &grid, const_w, true, b, 0)?;
        band::load(&mut a, &grid, IN_NAME, b, stage_w)?;
        band::copy_op(&mut a, &grid, const_w, false, b, 0)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, false)?;

        // ── Regroup the patch into wide words of 4 pixels x cwb words:
        // pixels (stride cw), words, 4-pixel groups, rows, channel blocks.
        let mut levels = vec![(cw, 4), (1, cwb)];
        if g > 1 {
            levels.push((4 * cw, g));
        }
        levels.push((tw * cw, th));
        if h > 1 {
            levels.push((16, h));
        }
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("narrow_lvl_mask", (1 << levels.len()) - 1),
            ("wide_addr", pool_w),
            ("wide_lvl_mask", if h > 1 { 3 } else { 1 }),
            ("sync_en0", 1),
            ("sync_en1", 0),
            ("sync_en2", 0),
            ("sync_f1", 1),
            ("tail_f799", 1),
            ("tail_lvl", 2),
        ];
        let n_walk = walk(&levels, 6);
        for (k, &(inc, lim)) in n_walk.iter().enumerate() {
            f.push((
                ["n_inc0", "n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"][k],
                inc,
            ));
            if k < 4 {
                f.push((["n_lim0", "n_lim1", "n_lim2", "n_lim3"][k], lim));
            }
        }
        let mut wl = vec![(1, wg)];
        if h > 1 {
            wl.push((wg, h));
        }
        for (k, &(inc, lim)) in walk(&wl, 4).iter().enumerate() {
            f.push((["w_inc0", "w_inc1", "w_inc2", "w_inc3"][k], inc));
            if k < 2 {
                f.push((["w_lim0", "w_lim1"][k], lim));
            }
        }
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        sync(&mut a, 0, 0x3ff, false)?;

        // ── The pool op: 2x2 window loops over the wide words, outputs to
        // narrow `b`, one pixel after the other.
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("loop0", 1),
            ("loop1", 1),
            ("loop2", ow - 1),
            ("loop3", oh - 1),
            ("loop4", h - 1),
            ("out_base", b),
            ("out_mode7", 3),
            ("out_tflags", 7),
            ("par_base", pool_w & 0xff),
            ("par_sel", 1),
            ("par_hmode", 1),
            ("par_cnt0", 1),
            ("par_mode0", 3),
            ("par_inc1", g - 1),
            ("par_cnt1", 1),
            ("par_mode1", 1),
            ("par_inc2", -g),
            ("par_cnt2", ow - 1),
            ("par_mode2", 1),
            ("par_cnt3", oh - 1),
            ("par_mode3", 1),
            ("par_cnt4", h - 1),
            ("par_mode4", 1),
            ("par_mode5", 1),
            ("par_inc6", -wg * h),
            ("par_mode6", 1),
            ("par_inc7", -wg * h),
            ("par_mode7", 3),
            ("par_tflags", 0xf),
            ("psum_cnt0", 3),
            ("psum_cnt1", wg * h - 1),
            ("psum_tflags", 0x840),
            ("cfg0", 0x127),
            ("cfg2", 6),
            ("cfg3", 7),
            ("in_zp", zp),
            ("dp_mode", 4),
            ("reduce_mask", 3),
            ("out_ch_last", 4 * cwb - 1),
            ("out_ch", 4 * cwb - 1),
            ("mult_bits", 1.0f32.to_bits() as i64),
            ("clamp_max_bits", ((255 - zp) as f32).to_bits() as i64),
            ("clamp_min_bits", (-zp as f32).to_bits() as i64),
            ("offset", -zp),
            ("out_zp", zp),
        ];
        op_walk(
            &mut f,
            "out",
            &[
                (1, cwb),
                (cw, ow),
                (ow * cw, oh),
                (cwb, h),
                (ow * oh * cw, 1),
            ],
            4,
        );
        a.emit(OP_POOL, &f)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Output: narrow b -> wide 0x2078 on every tile, then each tile
        // in turn produces its `bo` bytes for the outfeed.
        load_output_addr(&mut a, OUT_NAME)?;
        descriptor_addr(&mut a)?;
        let big = bo >= 512; // a full wide row: no circular buffer
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("narrow_addr", b / 4),
            ("narrow_lvl_mask", 7),
            ("wide_addr", 0x2078),
            ("wide_rows", 1),
            ("sync_en0", 1),
            ("sync_id", 0x10),
            ("sync_en1", 1),
            ("sync_val", 0xffff),
            ("sync_en2", 1),
            ("sync_f45", 1),
            ("tail_lvl", 3),
        ];
        if big {
            f.push(("w_lim0", bo / 256 - 1));
        } else {
            f.extend([
                ("w_inc0", 1),
                ("wide_circ", 1),
                ("sync_f1", 1),
                ("sync_wait_lvl", 1),
            ]);
        }
        for (k, &(inc, lim)) in walk(&[(1, cw), (cw, ow), (ow * cw, oh)], 6)
            .iter()
            .enumerate()
        {
            f.push((
                ["n_inc0", "n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"][k],
                inc,
            ));
            if k < 3 {
                f.push((["n_lim0", "n_lim1", "n_lim2"][k], lim));
            }
        }
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        descriptor_send(&mut a, 16 * bo, TAG_OUTPUT)?;
        for t in 0..16i64 {
            a.emit(
                OP_OUTFEED,
                &[("d0_limit", bo / 8 - 1), ("k173", 3), ("f222", 0x7f)],
            )?;
            a.emit(
                OP_RING_PRODUCER,
                &[
                    ("tile_mask", 1 << t),
                    ("addr", 0x2078),
                    ("inc0", 1),
                    ("cnt1", big as i64),
                    ("sdims", 1),
                    ("cbuf", 1),
                    ("slots", 1),
                    ("mode", 3),
                    ("pcfg", 0x14),
                    ("r0_id", 0x51),
                    ("r0_val", t),
                    ("r0_en_a", 1),
                    ("r0_en_b", 1),
                    ("r1_id", 0x2d),
                    ("r1_val", 1),
                    ("r1_en_a", 1),
                    ("r1_en_b", 1),
                    ("dest", 0x10000),
                ],
            )?;
        }
        epilogue(&mut a, 16, 0)?;

        let (oy, ox, oz) = self.out_shape();
        let (py, px) = (oh as usize, ow as usize);
        Ok(Program {
            chunk: a.finish()?,
            parameters: Vec::new(),
            extra_inputs: Vec::new(),
            input: layer(IN_NAME, (self.height, self.width, self.c), self.quant),
            output: LayerInfo {
                output_layout: Some(OutputLayoutInfo {
                    y_coordinate_to_linear_tile_id_map: (0..oy)
                        .map(|y| (4 * (y / py)) as i32)
                        .collect(),
                    x_coordinate_to_linear_tile_id_map: (0..ox).map(|x| (x / px) as i32).collect(),
                    linearized_tile_byte_offset: (0..16).map(|t| t * bo as i32).collect(),
                    x_coordinate_to_local_byte_offset: (0..ox)
                        .map(|x| ((x % px) * oz) as i32)
                        .collect(),
                    y_coordinate_to_local_y_offset: (0..oy).map(|y| (y % py) as i32).collect(),
                    x_coordinate_to_local_y_row_size: vec![(px * oz) as i32; ox],
                }),
                ..layer(OUT_NAME, (oy, ox, oz), self.quant)
            },
        })
    }
}
