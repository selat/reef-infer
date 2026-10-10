//! 2-D convolution on a 1x16x16xCin uint8 tensor, one 4x4 pixel patch per
//! tile (tile `4 * (y / 4) + x / 4`).
//!
//! Follows `edgetpu_compiler` 16.0's CONV_2D programs
//! (`conv_Cin16_Cout32_H16_W16_..._k1_padsame_stride1` in the ISA corpus): a
//! PARAMETER_CACHING executable stores the parameter blob in tile 0's wide
//! memory; per inference the input is spread over the tiles as in
//! [`super::elementwise`], tile 0 multicasts the blob to all 16 tiles over
//! the ring (`dest = 0xffff`, no scalar-memory hop), and one TensorOp per
//! tile computes its 16 pixels.
//!
//! The parameter blob is the FC layout ([`super::fc`]) with `K = k^2 Cin`
//! and `N = Cout`: three `N x 4` byte header rows (multiplier, bias, weight
//! zero point) and `K / 4` weight rows of `N x 4` bytes (4 input channels
//! per output channel). The receiving ringConsumer puts the header rows in
//! its aux rows (0x1fe8..) and the weight rows in slots below them.
//!
//! For k > 1 each tile stores its patch with a halo of `k / 2` pixels
//! ([`Patch`]), filled by four mesh moves between neighbouring tiles
//! (north, south, then west and east over the full padded height, so the
//! corners follow); edge tiles fill theirs with the input zero point. The
//! TensorOp then runs one pass per 4 input channels over the `k x k` taps,
//! accumulating in partial-sum memory, and the weight rows are streamed in
//! that order (`[Cin / 4][ky][kx]`).
//!
//! Supported (byte-identical to the compiler, `examples/conv_check.rs`):
//! same padding, stride 1, Cin = 16, and k = 1 with Cout = 32 or k = 3 with
//! Cout in {8, 16, 32}. (At Cout = 64 a tile's output is 1 KiB and the
//! compiler arranges narrow memory differently.)

use super::fc::Fc;
use super::*;

/// Spatial size (y, x) the layout is fixed to.
pub const HW: (usize, usize) = (16, 16);
/// Pixels per tile (a 4x4 patch).
const PX: i64 = 16;

/// A quantized CONV_2D layer (same padding, stride 1). `weights` is
/// `[Cout][k][k][Cin]` int8 with per-channel scales (TFLite's OHWI order).
#[derive(Debug, Clone)]
pub struct Conv {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub weights: Vec<i8>,
    pub w_scale: Vec<f32>,
    pub bias: Vec<i32>,
    pub input: Quant,
    pub output: Quant,
    pub relu: bool,
}

impl Conv {
    pub fn check(&self) -> Result<(), String> {
        let ok = match self.k {
            1 => (self.cin, self.cout) == (16, 32),
            3 => self.cin == 16 && [8, 16, 32].contains(&self.cout),
            _ => false,
        };
        if !ok {
            return Err(format!(
                "Cin={} Cout={} k={}: supported are k 1 (Cin 16, Cout 32) and k 3 (Cin 16, Cout 8, 16 or 32)",
                self.cin, self.cout, self.k
            ));
        }
        if self.weights.len() != self.cout * self.k * self.k * self.cin
            || self.w_scale.len() != self.cout
            || self.bias.len() != self.cout
        {
            return Err("weights / w_scale / bias sizes do not match Cin, Cout, k".into());
        }
        Ok(())
    }

    /// The tile patch: a halo of `k / 2` on each side (odd k).
    fn patch(&self) -> Patch {
        let p = (self.k / 2) as i64;
        Patch {
            cin: self.cin as i64,
            top: p,
            bottom: p,
            left: p,
            right: p,
        }
    }

    /// Kernel taps.
    fn taps(&self) -> i64 {
        (self.k * self.k) as i64
    }

    /// The per-channel arithmetic as an FC over one pixel's window, with K
    /// ordered as the weight rows are streamed: `[Cin / 4][ky][kx][4]`.
    fn fc(&self) -> Fc {
        let (k, cin) = (self.k, self.cin);
        let kk = k * k * cin;
        let mut weights = Vec::with_capacity(self.cout * kk);
        for o in 0..self.cout {
            for g in 0..cin / 4 {
                for t in 0..k * k {
                    for j in 0..4 {
                        weights.push(self.weights[(o * k * k + t) * cin + 4 * g + j]);
                    }
                }
            }
        }
        Fc {
            k: kk,
            n: self.cout,
            weights,
            w_scale: self.w_scale.clone(),
            bias: self.bias.clone(),
            input: self.input,
            output: self.output,
            relu: self.relu,
        }
    }

    /// Output channels as stored: at least 16 (the compiler pads with
    /// zero multiplier, bias, zero point and weight bytes).
    fn cout_pad(&self) -> usize {
        self.cout.max(16)
    }

    pub fn parameters(&self) -> Vec<u8> {
        let p = self.fc().parameters();
        let (row, pad) = (4 * self.cout, 4 * self.cout_pad());
        p.chunks(row)
            .flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(0, pad - row)))
            .collect()
    }

    /// Host reference over a y-x-z input: each output pixel is the FC of its
    /// `k x k` window, zero-point padded.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        let (h, w) = (HW.0 as i64, HW.1 as i64);
        let (k, cin) = (self.k as i64, self.cin);
        let p = k / 2;
        let fc = self.fc();
        let pad = self.input.zero_point as u8;
        let mut out = Vec::with_capacity(HW.0 * HW.1 * self.cout);
        for y in 0..h {
            for xx in 0..w {
                // window in FC order: [Cin / 4][ky][kx][4]
                let mut win = Vec::with_capacity(fc.k);
                for g in 0..cin / 4 {
                    for ky in 0..k {
                        for kx in 0..k {
                            let (sy, sx) = (y + ky - p, xx + kx - p);
                            for j in 0..4 {
                                win.push(if (0..h).contains(&sy) && (0..w).contains(&sx) {
                                    x[((sy * w + sx) as usize) * cin + 4 * g + j]
                                } else {
                                    pad
                                });
                            }
                        }
                    }
                }
                out.extend(fc.reference(&win));
            }
        }
        out
    }

    pub fn build(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.check()?;
        let params = self.parameters();
        let caching = ExecInfo {
            exec_type: ExecutableType::PARAMETER_CACHING,
            parameters: params.clone(),
            instruction_chunks: vec![self.caching_program(params.len())?],
            input_layers: vec![],
            output_layers: vec![],
            scratch_size_bytes: 0,
        };
        let (h, w) = HW;
        let exec = ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![self.execution_program()?],
            input_layers: vec![layer("input", (h, w, self.cin), self.input)],
            output_layers: vec![LayerInfo {
                output_layout: Some(output_layout(self.cout)),
                ..layer("output", (h, w, self.cout), self.output)
            }],
            scratch_size_bytes: 0,
        };
        Ok(TfliteModel {
            executables: vec![exec, caching],
        })
    }

    /// Parameter rows of `Cout x 4` bytes: 3 header rows + `K / 4` weight rows.
    fn param_words(&self) -> i64 {
        3 + (self.k * self.k * self.cin / 4) as i64
    }

    fn caching_program(&self, bytes: usize) -> Result<ChunkInfo, IsaError> {
        let words = self.param_words();
        let mut a = Asm::new();
        param_caching(
            &mut a,
            bytes,
            (self.cout_pad() / 16) as i64 - 1,
            &[
                ("tile_mask", 1),
                ("addr", 2),
                ("inc0", 1),
                ("cnt0", words - 1),
                ("inc1", -(words - 1)),
                ("inc2", -(words - 1)),
                ("inc3", -(words - 1)),
                ("sdims", 1),
            ],
        )?;
        a.finish()
    }

    fn execution_program(&self) -> Result<ChunkInfo, IsaError> {
        const IN_NAME: &str = "input";
        const OUT_NAME: &str = "output";
        let pt = self.patch();
        let (tb, tp, cq) = (pt.tb(), pt.tp(), pt.cq());
        let ob = PX * self.cout as i64; // output bytes per tile
        // 256 B wide words per row of 4 tiles of input
        let row_words = 4 * tb / 256;
        let mut a = Asm::new();
        prologue(&mut a)?;

        // ── Constant setup (as in elementwise): one-hot words at narrow
        // tp + 2 tb, copied to wide just below the input rows at 0x2078.
        let const_narrow = tp + 2 * tb;
        let const_wide = 0x2078 - row_words;
        for (i, fill) in [1i64, 0x100, 0x10000, 0x1000000].into_iter().enumerate() {
            let mut f = vec![
                ("tile_mask", ALL_TILES),
                ("i_addr", const_narrow + 4 * i as i64),
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
                ("narrow_addr", const_narrow / 4),
                ("n_inc0", 1),
                ("n_lim0", 3),
                ("n_inc1", -3),
                ("n_inc2", -3),
                ("n_inc3", -3),
                ("n_inc4", -3),
                ("n_inc5", -3),
                ("narrow_lvl_mask", 1),
                ("wide_addr", const_wide),
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
        copy_op(&mut a, &pt, const_wide & 0xff, true)?;

        // ── Input: 4 tb bytes per row of 4 tiles, linear (one slot of
        // `row_words` words at wide 0x2078), landing at narrow tp; column c
        // skips Cin * c words.
        load_input_addr(&mut a, IN_NAME)?;
        let size = 16 * tb;
        let units8 = size / 8;
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
        descriptor_send(&mut a, size, TAG_INPUT)?;
        a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("counters", 0x800)])?;
        let (nw, land) = (tb / 4, tp / 4); // narrow words per tile, landing word
        for col in 0..4i64 {
            let mut f: Vec<(&str, i64)> = vec![
                ("tile_mask", 0x1111 << col),
                ("wide_addr", 0x2078),
                ("w_lim0", row_words - 1),
                ("wide_rows", 1),
                ("narrow_addr", land),
                ("n_inc0", 1),
                ("n_lim0", nw - 1),
                ("n_lim1", 3),
                ("narrow_lvl_mask", 1),
                ("mode", 1),
                ("size64", tb / 64),
            ];
            for n in ["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"] {
                f.push((n, -(nw - 1)));
            }
            if col > 0 {
                let skip = self.cin as i64 * col;
                f.extend([
                    ("head_words_m1", nw - 1 - skip),
                    ("head_en", 1),
                    ("tail_words", skip),
                    ("skip_en0", 1),
                    ("skip_base", land),
                    ("skip_m1", skip - 1),
                    ("skip_end", land + skip - 1),
                    ("rsv543", 1),
                ]);
            }
            f.extend([
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
            a.emit(OP_WIDE_TO_NARROW, &f)?;
        }
        let per_row = 4 * tb / 8; // infeed units of 8 B
        for row in 0..4i64 {
            a.emit(
                OP_RING_CONSUMER,
                &[
                    ("tile_mask", 0xf << (4 * row)),
                    ("addr", 0x2078),
                    ("cnt0", row_words - 1),
                    ("slots", 1),
                    ("mode", 1),
                    ("s_id", 0xb),
                    ("s_val", -(4 * tb / 16)),
                    ("s_cnt", 1),
                    ("s_en_b", 1),
                    ("s_y", 1),
                ],
            )?;
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
        copy_op(&mut a, &pt, const_wide & 0xff, false)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        if self.k > 1 {
            halo(&mut a, &pt, self.input.zero_point)?;
        }
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Weights: tile 0 multicasts the cached blob to every tile. Header
        // rows land in the aux rows (0x1fe8..), weight rows in `slots` slots
        // of 8 address units below them: one per input word (k = 1), or one
        // per tap, refilled for each of the Cin / 4 passes (k = 3).
        let taps = self.taps();
        let oq = (self.cout / 4) as i64;
        let (slots, passes) = if self.k == 1 { (cq, 1) } else { (taps, cq) };
        let slots_addr = 0x1fe8 - 8 * slots;
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("addr", slots_addr),
            ("inc0", 1),
            ("cnt0", slots - 1),
            ("inc1", -(slots - 1)),
            ("inc2", -(slots - 1)),
            ("inc3", -(slots - 1)),
            ("sdims", 1),
            ("cbuf", 1),
            ("slots", slots),
            ("aux_en0", 1),
            ("aux_addr0", if passes > 1 { 0x1fea } else { 0x1fe9 }),
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
        ];
        if passes > 1 {
            f.push(("cnt1", passes - 1));
        }
        a.emit(OP_RING_CONSUMER2, &f)?;
        // header rows -> scaling memory (mode 2), as in fc
        let half = (self.cout_pad() / 2) as i64;
        let mut f = vec![
            ("tile_mask", ALL_TILES),
            ("wide_addr", 0x1fe8),
            ("w_inc0", 1),
            ("w_lim0", 1),
            ("w_inc1", -1),
            ("w_inc2", -1),
            ("w_inc3", -1),
            ("wide_lvl_mask", 1),
            ("wide_circ", 1),
            ("wide_rows", 3),
            ("n_inc0", 1),
            ("n_lim0", half - 1),
            ("n_inc1", 65 - half),
            ("n_lim1", 1),
            ("narrow_lvl_mask", 3),
            ("mode", 2),
            ("size64", 2),
            ("sync_en0", 1),
            ("sync_f2", 1),
            ("sync_id", 0xf),
            ("sync_wait_lvl", 1),
            ("sync_en1", 1),
            ("sync_val", if passes > 1 { 0x20000 } else { 0x8000 }),
            ("sync_en2", 1),
            ("sync_dec_lvl", 2),
            ("sync_dec", -1),
            ("sync_dec_mode", 3),
            ("tail_f867", 1),
            ("tail_en870", 1),
        ];
        for n in ["n_inc2", "n_inc3", "n_inc4", "n_inc5"] {
            f.push((n, -(half - 1 + 64)));
        }
        a.emit(OP_WIDE_TO_NARROW, &f)?;

        // ── The conv TensorOp.
        let (lo, hi) = self.fc().clamp();
        let mut f: Vec<(String, i64)> = Vec::new();
        let rs = pt.pw() * cq; // padded row stride in words
        let (in_dims, loops): (Vec<(i64, i64)>, [i64; 3]) = if self.k == 1 {
            // per pixel: Cin / 4 words against the slots
            (vec![(cq, 1), (1, 0), (PX, cq)], [cq - 1, PX - 1, 0])
        } else {
            // per pass (4 channels): taps (x, y), then pixels (x, y)
            let k = self.k as i64;
            (
                vec![(1, 0), (k, cq), (k, rs), (4, cq), (4, rs), (cq, 1)],
                [taps - 1, PX - 1, cq - 1],
            )
        };
        f.extend(walk("in", &in_dims, 8));
        for (i, l) in loops.iter().enumerate() {
            if *l > 0 {
                f.push((format!("loop{i}"), *l));
            }
        }
        let fixed: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("in_mode7", 3),
            ("in_tflags", if self.k == 1 { 1 } else { 0xf }),
            ("out_base", tp),
            ("out_inc0", 1),
            ("out_cnt0", oq - 1),
            ("out_inc1", 1),
            ("out_cnt1", PX - 1),
            ("out_inc2", -(oq * (PX - 1) - 1)),
            ("out_mode7", 3),
            ("out_tflags", 1),
            ("par_base", slots_addr),
            ("par_inc0", 1),
            ("par_cnt0", slots - 1),
            ("par_inc1", -(slots - 1)),
            ("par_cnt1", PX - 1),
            ("par_mode7", 1),
            ("par_tflags", 0x80),
            ("par_fifo", slots),
            ("cfg0", 0xa5),
            ("cfg1", 0x14f),
            ("sync0", 0x4000),
            ("sync1", 0x4000),
            ("sync2", if self.k == 1 { 0xa6 } else { 0xb6 }),
            ("sync3", 0x80),
            ("sync4", 0x80),
            ("cfg2", if self.k == 1 { 7 } else { 3 }),
            ("cfg3", 8),
            ("in_zp", self.input.zero_point as i64),
            ("row_a", 0x7ff),
            ("row_m", 3),
            ("row_c", 0x7fc),
            ("dp_mode", if self.k == 1 { 2 } else { 3 }),
            ("reduce_mask", if self.k == 1 { 1 } else { 5 }),
            ("out_ch_last", self.cout as i64 - 1),
            ("out_ch", self.cout as i64 - 1),
            ("mult_bits", 1.0f32.to_bits() as i64),
            ("clamp_max_bits", hi.to_bits() as i64),
            ("clamp_min_bits", lo.to_bits() as i64),
            ("out_zp", self.output.zero_point as i64),
            ("cfg4", 1),
            ("row_cfg", 1),
        ];
        f.extend(fixed.iter().map(|&(n, v)| (n.to_string(), v)));
        for j in 3..8 {
            f.push((format!("out_inc{j}"), -(PX * oq - 1)));
        }
        for j in 2..8 {
            f.push((format!("par_inc{j}"), -(slots - 1)));
        }
        if self.k == 1 {
            f.extend([
                ("psum_cnt0".into(), cq - 1),
                ("psum_cnt1".into(), PX - 1),
                ("psum_tflags".into(), 0x840),
            ]);
        } else {
            // partial sums: one per pixel, revisited each pass
            f.extend([
                ("par_cnt2".into(), cq - 1),
                ("psum_base".into(), slots_addr - 64),
                ("psum_cnt0".into(), taps - 1),
                ("psum_inc1".into(), 1),
                ("psum_cnt1".into(), PX - 1),
                ("psum_cnt2".into(), cq - 1),
                ("psum_mode7".into(), 2),
            ]);
            for j in 2..8 {
                f.push((format!("psum_inc{j}"), -(PX - 1)));
            }
        }
        let f: Vec<(&str, i64)> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        a.emit(OP_TENSOR, &f)?;
        let words = self.param_words();
        a.emit(
            OP_RING_PRODUCER,
            &[
                ("tile_mask", 1),
                ("inc0", 1),
                ("cnt0", words - 1),
                ("inc1", -(words - 1)),
                ("inc2", -(words - 1)),
                ("inc3", -(words - 1)),
                ("sdims", 1),
                ("pcfg", 4),
                ("r0_id", 0x32),
                ("r0_en_a", 1),
                ("r0_en_b", 1),
                ("to_c1", 1),
                ("dest", ALL_TILES), // ring multicast to every tile
            ],
        )?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Output: narrow tp -> wide 0x2078 on every tile, then each tile
        // in turn produces its ob bytes to the outfeed.
        load_output_addr(&mut a, OUT_NAME)?;
        descriptor_addr(&mut a)?;
        let mut f = vec![
            ("tile_mask", ALL_TILES),
            ("narrow_addr", tp / 4),
            ("n_inc0", 1),
            ("n_lim0", ob / 64 - 1),
            ("n_inc1", 1),
            ("n_lim1", 3),
            ("n_inc2", 1),
            ("n_lim2", 3),
            ("n_inc3", -(ob / 4 - 1)),
            ("n_inc4", -(ob / 4 - 1)),
            ("n_inc5", -(ob / 4 - 1)),
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
        if ob <= 256 {
            // one wide word per tile: a circular single-row target
            f.extend([
                ("w_inc0", 1),
                ("wide_circ", 1),
                ("sync_f1", 1),
                ("sync_wait_lvl", 1),
            ]);
        } else {
            f.push(("w_lim0", ob / 256 - 1));
        }
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        descriptor_send(&mut a, 16 * ob, TAG_OUTPUT)?;
        for t in 0..16i64 {
            a.emit(
                OP_OUTFEED,
                &[("d0_limit", ob / 8 - 1), ("k173", 3), ("f222", 0x7f)],
            )?;
            a.emit(
                OP_RING_PRODUCER,
                &[
                    ("tile_mask", 1 << t),
                    ("addr", 0x2078),
                    ("inc0", 1),
                    ("cnt1", (ob / 256 - 1).max(0)),
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
        epilogue(&mut a, 16, words)?;
        a.finish()
    }
}

/// Address-generator fields `{p}_inc{j}` / `{p}_cnt{j}` for a loop nest
/// `dims` (innermost first) of `(count, stride)` in words. An increment is
/// applied from the last element of the inner loops, so it is the stride
/// minus their span; a loop of count 1 gets increment 1. The levels after
/// the last loop (up to `levels`) rewind to the start.
fn walk(p: &str, dims: &[(i64, i64)], levels: usize) -> Vec<(String, i64)> {
    let mut f = Vec::new();
    let mut span = 0;
    for (j, &(n, stride)) in dims.iter().enumerate() {
        f.push((format!("{p}_inc{j}"), if n > 1 { stride - span } else { 1 }));
        if n > 1 {
            f.push((format!("{p}_cnt{j}"), n - 1));
            span += (n - 1) * stride;
        }
    }
    for j in dims.len()..levels {
        f.push((format!("{p}_inc{j}"), -span));
    }
    f
}

/// One mesh walk (`o` or `i` side): `dims` with count 1 dropped, the
/// rewind on the remaining levels up to 3, `{p}_sdims` = active levels.
fn mesh_walk(p: &str, addr: i64, dims: &[(i64, i64)]) -> (Vec<(String, i64)>, i64) {
    let dims: Vec<(i64, i64)> = dims.iter().copied().filter(|d| d.0 > 1).collect();
    let mut f = vec![(format!("{p}_addr"), addr)];
    f.extend(walk(p, &dims, 4));
    f.push((format!("{p}_sdims"), (1 << dims.len()) - 1));
    (f, dims.len() as i64)
}

/// One halo move: interior part sent (`o_addr`) into the neighbour's halo
/// (`i_addr`), over `dims`. `groups` are the tile masks that only receive,
/// that send and receive, and that send and fill their own halo with the
/// zero point (no neighbour on that side). Column moves wait on sync
/// flags 6 / 0xa (value 6 for one-pixel halos, Cin >= 8), after the rows.
struct Move {
    op: u8,
    o_addr: i64,
    i_addr: i64,
    dims: Vec<(i64, i64)>,
    groups: [i64; 3],
    synced: bool,
}

/// Fills each tile's halo: rows to the north and south neighbours, then
/// columns (over the full padded height, so the corners follow) to the
/// west and east ones.
fn halo(a: &mut Asm, pt: &Patch, zp: i32) -> Result<(), IsaError> {
    let (cq, pw) = (pt.cq(), pt.pw());
    let rs = pw * cq;
    let fill = (zp as u8 as i64) * 0x0101_0101;
    let rows = |n: i64| vec![(cq, 1), (4, cq), (n, rs)];
    let cols = |n: i64| vec![(cq, 1), (n, cq), (pw, rs)];
    let (t, b, l, r) = (pt.top, pt.bottom, pt.left, pt.right);
    let moves = [
        // north: top interior rows -> the northern tile's bottom halo
        Move {
            op: OP_MESH_N,
            o_addr: pt.at(0, 0),
            i_addr: pt.at(4, 0),
            dims: rows(b),
            groups: [0x000f, 0x0ff0, 0xf000],
            synced: false,
        },
        // south: bottom interior rows -> the southern tile's top halo
        Move {
            op: OP_MESH_S,
            o_addr: pt.at(4 - t, 0),
            i_addr: pt.at(-t, 0),
            dims: rows(t),
            groups: [0xf000, 0x0ff0, 0x000f],
            synced: false,
        },
        // west: left interior columns -> the western tile's right halo
        Move {
            op: OP_MESH_W,
            o_addr: pt.at(-t, 0),
            i_addr: pt.at(-t, 4),
            dims: cols(r),
            groups: [0x1111, 0x6666, 0x8888],
            synced: true,
        },
        // east: right interior columns -> the eastern tile's left halo
        Move {
            op: OP_MESH_E,
            o_addr: pt.at(-t, 4 - l),
            i_addr: pt.at(-t, -l),
            dims: cols(l),
            groups: [0x8888, 0x6666, 0x1111],
            synced: true,
        },
    ];
    for Move {
        op,
        o_addr,
        i_addr,
        dims,
        groups,
        synced,
    } in moves
    {
        let (o, _) = mesh_walk("o", o_addr, &dims);
        let (i, n) = mesh_walk("i", i_addr, &dims);
        // the compiler lists the tile groups in ascending tile order
        let mut order = [(groups[0], 0), (groups[1], 1), (groups[2], 2)];
        order.sort();
        for (mask, role) in order {
            let mut f: Vec<(String, i64)> = vec![("tile_mask".into(), mask)];
            if role > 0 {
                f.extend(o.iter().cloned());
            }
            f.extend(i.iter().cloned());
            f.push(("in_mode".into(), 2 * n + 1));
            if synced && role > 0 {
                for s in ["s0", "s1"] {
                    f.extend([
                        (format!("{s}_id"), if s == "s0" { 6 } else { 0xa }),
                        (format!("{s}_val"), 6),
                        (format!("{s}_en_a"), 1),
                        (format!("{s}_en_b"), 1),
                    ]);
                }
            }
            if role == 2 {
                f.extend([("fill_en".into(), 1), ("fill".into(), fill)]);
            }
            let f: Vec<(&str, i64)> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
            a.emit(op, &f)?;
        }
    }
    Ok(())
}

/// Tile `4 * (y / 4) + x / 4` holds a 4x4 patch, rows of 4 pixels x `nz`
/// bytes; tiles are sent in order.
fn output_layout(nz: usize) -> OutputLayoutInfo {
    let (ny, nx) = HW;
    OutputLayoutInfo {
        y_coordinate_to_linear_tile_id_map: (0..ny).map(|y| (4 * (y / 4)) as i32).collect(),
        x_coordinate_to_linear_tile_id_map: (0..nx).map(|x| (x / 4) as i32).collect(),
        linearized_tile_byte_offset: (0..16).map(|t| (t * 16 * nz) as i32).collect(),
        x_coordinate_to_local_byte_offset: (0..nx).map(|x| ((x % 4) * nz) as i32).collect(),
        y_coordinate_to_local_y_offset: (0..ny).map(|y| (y % 4) as i32).collect(),
        x_coordinate_to_local_y_row_size: vec![(4 * nz) as i32; nx],
    }
}
