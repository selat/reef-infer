//! Element-wise ADD of two `H x W x C` uint8 tensors with requantization
//! (and an optional fused RELU), split over 16 tiles.
//!
//! Follows `edgetpu_compiler` 16.0's program (`eltwise_*_opadd_*` in the
//! ISA corpus): input A goes through the band loader ([`super::band`]) into
//! the patch at narrow 0, input B into the patch at narrow `B` (the bytes
//! per tile), each staged right above its patch. The add op then reads word
//! pairs of both patches and computes, per value,
//!
//! `out = sat_u8(round_half_even(clamp((wa a + wb b + offset) mult, lo, hi)) + out_zp)`
//!
//! with integer weights `wa : wb` approximating `s_a : s_b` (the best
//! fraction with denominator <= 32767, the larger scale getting the larger
//! weight), `offset = -(wa za + wb zb)` and `mult = s_big / (w_big s_out)`.
//! The weights are mesh-filled into narrow memory and moved to wide 0x2070
//! (the op's parameter path); the output overwrites B's patch.
//!
//! Chained on chip ([`Add::chain_program`], [`super::chain`]) both inputs
//! are already in narrow memory, in padded patches of one geometry: the op
//! walks their interiors (B at a word offset from A) and writes the sum
//! into the next layer's patch, or into a plain buffer for the host.

use super::band::{self, Grid, walk};
use super::*;

/// Where a chained add's inputs are and its output goes.
#[derive(Debug, Clone, Copy)]
pub struct Io {
    /// Narrow byte addresses of A and B, each in a patch of geometry
    /// `patch` with its interior filled (the halo is not read).
    pub a: i64,
    pub b: i64,
    pub patch: Patch,
    /// Write the output into a patch at narrow `.0` of geometry `.1`, or
    /// (`None`) send it to the host.
    pub output_to: Option<(i64, Patch)>,
    /// Free narrow bytes ([`Add::scratch_bytes`]): the weights, and the
    /// output on its way to the host.
    pub scratch: i64,
}

/// Where the op's operands are and its output goes, in narrow bytes and
/// words: A's first interior byte, B's offset from it (words) and the
/// input row stride (words), the output's first byte and row stride, the
/// weights.
struct Place {
    in_base: i64,
    b_off: i64,
    in_rs: i64,
    out_base: i64,
    out_rs: i64,
    weights: i64,
}

#[derive(Debug, Clone)]
pub struct Add {
    pub height: usize,
    pub width: usize,
    pub c: usize,
    pub a: Quant,
    pub b: Quant,
    pub output: Quant,
    pub relu: bool,
}

/// The best rational approximation `p / q` of `n / d` (`n <= d`) with
/// `q <= max_q`, as Python's `Fraction.limit_denominator`: continued-fraction
/// convergents, then the closer of the last convergent and semiconvergent.
fn limit_denominator(n: u128, d: u128, max_q: u128) -> (u128, u128) {
    let g = gcd(n, d);
    let (n0, d0) = (n / g, d / g);
    if d0 <= max_q {
        return (n0, d0);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0u128, 1u128, 1u128, 0u128);
    let (mut n, mut d) = (n0, d0);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_q {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (max_q - q0) / q1;
    let (bp, bq) = (p0 + k * p1, q0 + k * q1);
    // |p1/q1 - x| <= |bp/bq - x| with x = n0/d0, cross-multiplied
    let err = |p: u128, q: u128| (p * d0).abs_diff(n0 * q);
    if err(p1, q1) * bq <= err(bp, bq) * q1 {
        (p1, q1)
    } else {
        (bp, bq)
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// An f32 as an exact `mantissa * 2^exp`.
fn exact(x: f32) -> (u128, i32) {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32;
    let m = (bits & 0x7f_ffff) as u128;
    if e == 0 {
        (m, -149)
    } else {
        (m | 0x80_0000, e - 150)
    }
}

impl Add {
    pub fn check(&self) -> Result<(), String> {
        let g = self.grid();
        let ok = matches!(self.height, 16 | 32)
            && matches!(self.width, 16 | 32)
            && self.c % 16 == 0
            && [2, 4, 8].contains(&g.slots())
            && self.a.scale > 0.0
            && self.b.scale > 0.0;
        if !ok {
            return Err(format!(
                "add {}x{}x{}: need H, W in 16 / 32 and an image row of 512 / 1024 / 2048 B",
                self.height, self.width, self.c
            ));
        }
        Ok(())
    }

    fn grid(&self) -> Grid {
        Grid::new(self.height, self.width, self.c)
    }

    /// Integer weights `(wa, wb)`: `s_small / s_big` as the best fraction
    /// with denominator <= 32767, the larger scale getting the denominator.
    pub fn weights(&self) -> (i64, i64) {
        let (sa, sb) = (self.a.scale, self.b.scale);
        let (big, small) = if sa >= sb { (sa, sb) } else { (sb, sa) };
        let ((ms, es), (mb, eb)) = (exact(small), exact(big));
        // small / big = ms 2^es / (mb 2^eb), exactly
        let (n, d) = if es >= eb {
            (ms << (es - eb), mb)
        } else {
            (ms, mb << (eb - es))
        };
        let (p, q) = limit_denominator(n, d, 32767);
        let (wbig, wsmall) = (q as i64, p as i64);
        if sa >= sb {
            (wbig, wsmall)
        } else {
            (wsmall, wbig)
        }
    }

    /// `s_big / (w_big s_out)`, in f32 as the compiler computes it.
    pub fn mult(&self) -> f32 {
        let (wa, wb) = self.weights();
        let (big, wbig) = if self.a.scale >= self.b.scale {
            (self.a.scale, wa)
        } else {
            (self.b.scale, wb)
        };
        big * (1.0 / (wbig as f32 * self.output.scale))
    }

    pub fn offset(&self) -> i64 {
        let (wa, wb) = self.weights();
        -(wa * self.a.zero_point as i64 + wb * self.b.zero_point as i64)
    }

    /// Clamp bounds as for FC / conv (one ulp inside the integer bounds).
    pub fn clamp(&self) -> (f32, f32) {
        let (s, zp) = (self.output.scale, self.output.zero_point as f32);
        let bound = |q: f32| ((q - zp) * s) * (1.0 / s);
        (if self.relu { 0.0 } else { bound(0.0) }, bound(255.0))
    }

    /// Host reference over y-x-z inputs.
    pub fn reference(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        let (wa, wb) = self.weights();
        let (off, mult) = (self.offset(), self.mult());
        let (lo, hi) = self.clamp();
        let zo = self.output.zero_point as f32;
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let acc = wa * x as i64 + wb * y as i64 + off;
                let v = (acc as f32 * mult).clamp(lo, hi).round_ties_even();
                (v + zo).clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    pub fn build(&self) -> Result<Program, Box<dyn std::error::Error>> {
        self.check()?;
        Ok(self.program()?)
    }

    /// Narrow bytes a chained add needs to itself ([`Io::scratch`]).
    pub fn scratch_bytes(&self, to_host: bool) -> i64 {
        0x100 + if to_host { self.grid().b() } else { 0 }
    }

    /// The patch interior this add reads and writes, unpadded.
    pub fn patch(&self) -> Patch {
        let g = self.grid();
        Patch {
            rows: g.th,
            cin: g.tw * g.c / 4,
            ..Patch::plain(0)
        }
    }

    /// The execution program with the inputs on chip as `io` says; its
    /// output layer (host output only) is "output".
    pub fn chain_program(&self, io: Io) -> Result<ChunkInfo, Box<dyn std::error::Error>> {
        self.check()?;
        // the op reads A then B at a positive offset: swap the operands
        // (the arithmetic is symmetric) to keep A below B
        if io.b < io.a {
            let swapped = Add {
                a: self.b,
                b: self.a,
                ..self.clone()
            };
            return swapped.chain_program(Io {
                a: io.b,
                b: io.a,
                ..io
            });
        }
        let own = self.patch();
        let fits = |p: &Patch| p.rows == own.rows && p.cin == own.cin;
        if !fits(&io.patch) || io.output_to.is_some_and(|(_, p)| !fits(&p)) {
            return Err(format!(
                "chained add: patches {:?} / {:?} do not hold {}x{}x{}",
                io.patch, io.output_to, self.height, self.width, self.c
            )
            .into());
        }
        if io.b - io.a < io.patch.tp() || io.a % 4 != 0 || io.b % 4 != 0 {
            return Err(format!("chained add: inputs at {:#x} / {:#x} overlap", io.a, io.b).into());
        }
        let rs = |p: &Patch| p.pw() * p.cq();
        let (out_base, out_rs) = match io.output_to {
            Some((at, p)) => (at + p.at(0, 0), rs(&p)),
            None => (io.scratch + 0x100, rs(&own)),
        };
        let place = Place {
            in_base: io.a + io.patch.at(0, 0),
            b_off: (io.b - io.a) / 4,
            in_rs: rs(&io.patch),
            out_base,
            out_rs,
            weights: io.scratch,
        };
        Ok(self.emit(Some(place), io.output_to.is_none())?.0)
    }

    /// The program; its input layers are "input_a" and "input_b" (the
    /// host sends them back to back).
    pub fn program(&self) -> Result<Program, IsaError> {
        let (chunk, output) = self.emit(None, true)?;
        let shape = (self.height, self.width, self.c);
        Ok(Program {
            chunk,
            parameters: Vec::new(),
            input: layer("input_a", shape, self.a),
            extra_inputs: vec![layer("input_b", shape, self.b)],
            output,
        })
    }

    /// The bitstream: host inputs (`place` `None`) or chained, the output to
    /// the host or not, and the host output layer.
    fn emit(
        &self,
        place: Option<Place>,
        to_host: bool,
    ) -> Result<(ChunkInfo, LayerInfo), IsaError> {
        const A_NAME: &str = "input_a";
        const B_NAME: &str = "input_b";
        const OUT_NAME: &str = "output";
        let g = self.grid();
        let (tw, th, cw, b, s) = (g.tw, g.th, g.cw(), g.b(), g.slots());
        let w = b / 4; // words per patch
        let chained = place.is_some();
        // host inputs: A at 0, B right above it, the output over B
        let p = place.unwrap_or(Place {
            in_base: 0,
            b_off: w,
            in_rs: tw * cw,
            out_base: b,
            out_rs: tw * cw,
            weights: 2 * b,
        });
        // Wide memory: with a 2-slot image row the constants sit on top and
        // the staging below them; otherwise the staging on top.
        let (stage_w, const_w) = if s == 2 {
            (0x206c, 0x207c)
        } else {
            (0x2080 - 8 * s, 0x2080 - 8 * s - 4)
        };
        let const_n = b + 8 * tw * self.c as i64;
        let par = const_w & 0xff;

        let mut a = Asm::new();
        prologue(&mut a)?;
        if !chained {
            band::constants(&mut a, const_n, const_w)?;
            // input A: staged at b, patch at 0
            band::copy_op(&mut a, &g, par, true, b, 0)?;
            band::load(&mut a, &g, A_NAME, b, stage_w)?;
            band::copy_op(&mut a, &g, par, false, b, 0)?;
            sync(&mut a, 0x1ffff, 0x3ff, false)?;
            a.emit(OP_SCSYNC, &[("smask", 5), ("sunits", 0xc)])?;
            sync(&mut a, 0, 0x3ff, true)?;
            wait_tiles(&mut a)?;
            a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("units", 0x20)])?;
            // input B: staged at 2b, patch at b
            band::copy_op(&mut a, &g, par, true, 2 * b, b)?;
            band::load(&mut a, &g, B_NAME, 2 * b, stage_w)?;
            band::copy_op(&mut a, &g, par, false, 2 * b, b)?;
            sync(&mut a, 0x1ffff, 0x3ff, false)?;
        }
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── Weights: wa, wb as 16-bit halves at narrow 2b, filled from the
        // four mesh directions, then moved to wide 0x2070.
        let (wa, wb) = self.weights();
        let wn = p.weights;
        let zero4: &[(&str, i64)] = &[
            ("i_cnt0", 3),
            ("i_inc1", -3),
            ("i_inc2", -3),
            ("i_inc3", -3),
            ("i_sdims", 1),
        ];
        for (op, off, fill) in [
            (OP_MESH_N, 0x00, Some(wa)),
            (OP_MESH_E, 0x04, Some(wa << 16)),
            (OP_MESH_W, 0x08, None),
            (OP_MESH_S, 0x18, Some(wa)),
            (OP_MESH_N, 0x1c, Some(wa << 16)),
            (OP_MESH_E, 0x20, Some(wb)),
            (OP_MESH_W, 0x24, Some(wb << 16)),
            (OP_MESH_S, 0x28, None),
            (OP_MESH_N, 0x38, Some(wb)),
            (OP_MESH_E, 0x3c, Some(wb << 16)),
        ] {
            let mut f: Vec<(&str, i64)> = vec![
                ("tile_mask", ALL_TILES),
                ("i_addr", wn + off),
                ("i_inc0", 1),
                ("in_mode", 3),
                ("fill_en", 1),
            ];
            match fill {
                Some(v) => f.push(("fill", v)),
                None => f.extend_from_slice(zero4), // 4 zero words
            }
            a.emit(op, &f)?;
        }
        a.emit(
            OP_NARROW_TO_WIDE,
            &[
                ("tile_mask", ALL_TILES),
                ("narrow_addr", wn / 4),
                ("n_inc0", 1),
                ("n_lim0", 3),
                ("n_inc1", 1),
                ("n_lim1", 3),
                ("n_inc2", -0xf),
                ("n_inc3", -0xf),
                ("n_inc4", -0xf),
                ("n_inc5", -0xf),
                ("narrow_lvl_mask", 3),
                ("wide_addr", 0x2070),
                ("w_inc0", 1),
                ("w_lim0", 3),
                ("w_inc1", -3),
                ("w_inc2", -3),
                ("w_inc3", -3),
                ("wide_lvl_mask", 1),
                ("sync_en0", 1),
                ("sync_f1", 1),
                ("sync_id", 3),
                ("sync_wait_lvl", 2),
                ("sync_en1", 0),
                ("sync_val", 0x8001),
                ("sync_en2", 1),
                ("tail_lvl", 1),
                // unexplained, constant in the corpus
                ("rsv589", 0x1000345),
                ("rsv621", 0xd1900),
                ("rsv653", 0x24440004),
                ("rsv685", 0x10001000),
            ],
        )?;

        // ── The add op: word pairs of both patches (stride w), pixels, rows,
        // channel words; output over B's patch.
        let (lo, hi) = self.clamp();
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("loop0", 1),
            ("loop1", 1),
            ("loop2", tw - 1),
            ("loop3", th - 1),
            ("loop4", cw - 1),
            ("in_hmode", 2),
            ("in_mode7", 3),
            ("in_tflags", 0xf),
            ("out_base", p.out_base),
            ("out_mode7", 3),
            ("out_tflags", 7),
            ("par_base", 0x70),
            ("par_sel", 1),
            ("par_inc0", 1),
            ("par_cnt0", 3),
            ("par_inc1", -3),
            ("par_cnt1", w - 1),
            ("par_mode7", 1),
            ("psum_cnt0", 3),
            ("psum_cnt1", w - 1),
            ("psum_tflags", 0x840),
            ("cfg0", 9),
            ("cfg1", 0x1ad),
            ("sync0", 0x4000),
            ("sync1", 0x4000),
            ("cfg2", 0x16),
            ("dp_mode", 4),
            ("reduce_mask", 3),
            ("out_ch_last", 3),
            ("out_ch", 3),
            ("mult_bits", self.mult().to_bits() as i64),
            ("clamp_max_bits", hi.to_bits() as i64),
            ("clamp_min_bits", lo.to_bits() as i64),
            ("offset", self.offset()),
            ("out_zp", self.output.zero_point as i64),
        ];
        for k in 2..8 {
            f.push((
                [
                    "", "", "par_inc2", "par_inc3", "par_inc4", "par_inc5", "par_inc6", "par_inc7",
                ][k],
                -3,
            ));
        }
        const IN_INC: [&str; 8] = [
            "in_inc0", "in_inc1", "in_inc2", "in_inc3", "in_inc4", "in_inc5", "in_inc6", "in_inc7",
        ];
        const IN_CNT: [&str; 6] = [
            "in_cnt0", "in_cnt1", "in_cnt2", "in_cnt3", "in_cnt4", "in_cnt5",
        ];
        const IN_MODE: [&str; 7] = [
            "in_mode0", "in_mode1", "in_mode2", "in_mode3", "in_mode4", "in_mode5", "in_mode6",
        ];
        if p.in_base != 0 {
            f.push(("in_base", p.in_base));
        }
        let in_levels = [
            (1, 2),
            (p.b_off, 2),
            (cw, tw),
            (p.in_rs, th),
            (p.b_off, 1),
            (1, cw),
        ];
        for (k, (inc, cnt)) in walk(&in_levels, 8).into_iter().enumerate() {
            if k > 0 {
                f.push((IN_INC[k], inc)); // the first level's stride is implicit
            }
            if k < 6 {
                f.push((IN_CNT[k], cnt));
            }
            if k < 7 {
                f.push((IN_MODE[k], 2));
            }
        }
        const OUT_INC: [&str; 8] = [
            "out_inc0", "out_inc1", "out_inc2", "out_inc3", "out_inc4", "out_inc5", "out_inc6",
            "out_inc7",
        ];
        const OUT_CNT: [&str; 5] = ["out_cnt0", "out_cnt1", "out_cnt2", "out_cnt3", "out_cnt4"];
        let out_levels = [(1, 1), (cw, tw), (p.out_rs, th), (p.b_off, 1), (1, cw)];
        for (k, (inc, cnt)) in walk(&out_levels, 8).into_iter().enumerate() {
            f.push((OUT_INC[k], inc));
            if k < 5 {
                f.push((OUT_CNT[k], cnt));
            }
        }
        a.emit(OP_TENSOR, &f)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;
        if !to_host {
            // the output stays on chip
            epilogue(&mut a, 0, 0)?;
            return Ok((a.finish()?, layer(OUT_NAME, (0, 0, 0), self.output)));
        }

        // ── Output: narrow b -> wide 0x2078 on every tile, then each tile in
        // turn produces its b bytes for the outfeed.
        load_output_addr(&mut a, OUT_NAME)?;
        descriptor_addr(&mut a)?;
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("narrow_addr", p.out_base / 4),
            ("narrow_lvl_mask", 7),
            ("wide_addr", 0x2078),
            ("w_lim0", b / 256 - 1),
            ("wide_rows", 1),
            ("sync_en0", 1),
            ("sync_id", 0x10),
            ("sync_en1", 1),
            ("sync_val", 0xffff),
            ("sync_en2", 1),
            ("sync_f45", 1),
            ("tail_lvl", 3),
        ];
        for (k, &(inc, lim)) in walk(&[(1, cw), (cw, tw), (tw * cw, th)], 6)
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
        descriptor_send(&mut a, 16 * b, TAG_OUTPUT)?;
        for t in 0..16i64 {
            a.emit(
                OP_OUTFEED,
                &[("d0_limit", b / 8 - 1), ("k173", 3), ("f222", 0x7f)],
            )?;
            a.emit(
                OP_RING_PRODUCER,
                &[
                    ("tile_mask", 1 << t),
                    ("addr", 0x2078),
                    ("inc0", 1),
                    ("cnt1", b / 256 - 1),
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

        let shape = (self.height, self.width, self.c);
        let (py, px, nz) = (th as usize, tw as usize, self.c);
        let mut output = layer(OUT_NAME, shape, self.output);
        output.output_layout = Some(OutputLayoutInfo {
            y_coordinate_to_linear_tile_id_map: (0..shape.0)
                .map(|y| (4 * (y / py)) as i32)
                .collect(),
            x_coordinate_to_linear_tile_id_map: (0..shape.1).map(|x| (x / px) as i32).collect(),
            linearized_tile_byte_offset: (0..16).map(|t| (t as i64 * b) as i32).collect(),
            x_coordinate_to_local_byte_offset: (0..shape.1)
                .map(|x| ((x % px) * nz) as i32)
                .collect(),
            y_coordinate_to_local_y_offset: (0..shape.0).map(|y| (y % py) as i32).collect(),
            x_coordinate_to_local_y_row_size: vec![(px * nz) as i32; shape.1],
        });
        Ok((a.finish()?, output))
    }
}
