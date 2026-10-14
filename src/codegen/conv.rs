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
//! **Larger images** (width / height 32 or 64) are derived without the
//! compiler (bit-exact on the TPU, not compared with its output): a row of
//! `width` pixels is loaded as 16 superpixels of `m = width / 16` pixels,
//! i.e. the 16x16 loader, copy ops and halo exchange with `m Cin` channels.
//! The halo is then whole superpixels (a superset of the 1-pixel halo), and
//! the conv op walks real pixels inside that layout from `in_base`, the
//! first window's origin. Each tile computes `4 x 4m` output pixels (or
//! `2 x 2m` at stride 2).
//!
//! Taller images (height 32 / 64) give each tile `height / 4` rows. The
//! loader moves them as consecutive 4-row bands: each band goes through the
//! verified 16-row loader and copy ops unchanged (its infeeds read that
//! band's rows of the staged image, every byte once), and the copy ops place
//! it 4 rows further down the patch. Verified on the TPU up to 64x64.
//!
//! Supported (byte-identical to the compiler on 16 corpus models,
//! `examples/conv_check.rs`; bit-exact on the TPU): same padding, 16x16,
//! stride 1 or 2, and k = 1 with Cin 16, Cout 32, or k = 3 with Cin in
//! {1, 3, 4, 16, 32, 64, 128} and Cout in {8, 16, 32}. With stride 2 the
//! padding is TensorFlow's (1 pixel after, none before, for k = 3), so the
//! halo exists only on the bottom / right and two of the four mesh moves
//! remain; each tile computes a 2x2 block of the 8x8 output.
//!
//! One tile's input (16 Cin bytes) lands through a wide-memory buffer: for
//! Cin <= 4 a row of 4 tiles shares one 256 B word, for Cin 16 a tile is
//! one word, above that a circular buffer of Cin / 16 slots. Cin 1 and 3
//! are widened on chip to one 4-byte word per pixel (zero weights in the
//! extra channels); from there the program is Cin 4's. Cin 4 and k = 1 take
//! all taps in one TensorOp pass, Cin >= 16 one pass per 4 input channels.
//! Not yet: Cin 8 (another scheme), Cout >= 64 (a tile's output is 1 KiB
//! and the compiler arranges narrow memory differently).

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
    /// 1 or 2 (same padding: TensorFlow pads the far side more).
    pub stride: usize,
    /// Image width: 16, 32 or 64.
    pub width: usize,
    /// Image height: 16, 32 or 64.
    pub height: usize,
    pub weights: Vec<i8>,
    pub w_scale: Vec<f32>,
    pub bias: Vec<i32>,
    pub input: Quant,
    pub output: Quant,
    pub relu: bool,
}

impl Conv {
    pub fn check(&self) -> Result<(), String> {
        let m = self.width / 16;
        let wide_ok = match self.width {
            16 => true,
            // superpixels of m pixels must be a loader-verified channel count
            32 | 64 => self.cin >= 16 && [32, 64, 128].contains(&(self.cin * m)),
            _ => false,
        };
        let tall_ok = match self.height {
            16 => true,
            // bands of 4 rows through the verified loader: >= 16 superpixel
            // channels (the linear and circular input forms)
            32 | 64 => self.cin_pad() * m >= 16,
            _ => false,
        };
        let ob = (self.height / 4 / self.stride) * (4 * m / self.stride) * self.cout;
        let ok = [1, 2].contains(&self.stride)
            && wide_ok
            && tall_ok
            && ob <= 2048
            && match self.k {
                1 => (self.cin, self.cout) == (16, 32),
                3 => {
                    [1, 3, 4, 16, 32, 64, 128].contains(&self.cin)
                        && [8, 16, 32].contains(&self.cout)
                }
                _ => false,
            };
        if !ok {
            return Err(format!(
                "Cin={} Cout={} k={} width={}: supported are k 1 (Cin 16, Cout 32) and k 3 (Cin 1, 3, 4, 16, 32, 64, 128; Cout 8, 16 or 32), width 16, or 32 / 64 with Cin x width / 16 in 32..=128 and at most 2 KiB of output per tile",
                self.cin, self.cout, self.k, self.width
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

    /// Same padding (TensorFlow): `(before, after)` pixels on each axis,
    /// `total = max((out - 1) s + k - in, 0)` split with the extra pixel
    /// after.
    fn pad(&self) -> (i64, i64) {
        let (n, k, s) = (HW.0 as i64, self.k as i64, self.stride as i64);
        let out = (n + s - 1) / s;
        let total = ((out - 1) * s + k - n).max(0);
        (total / 2, total - total / 2)
    }

    /// Pixels per superpixel: a row of `width` pixels is loaded as 16
    /// superpixels of `width / 16` pixels (the 16x16 loader with
    /// `Cin x width / 16` channels).
    fn spx(&self) -> i64 {
        (self.width / 16) as i64
    }

    /// Output pixels per tile along y and x.
    fn opx(&self) -> (i64, i64) {
        let s = self.stride as i64;
        ((self.height / 4) as i64 / s, 4 * self.spx() / s)
    }

    /// The tile patch in superpixels: 4x4 of them, the padding as a halo
    /// (rounded up to whole superpixels along x; the conv op addresses real
    /// pixels inside).
    fn patch(&self) -> Patch {
        let (before, after) = self.pad();
        let m = self.spx();
        Patch {
            cin: self.cin_pad() as i64 * m,
            rows: (self.height / 4) as i64,
            top: before,
            bottom: after,
            left: (before + m - 1) / m,
            right: (after + m - 1) / m,
        }
    }

    /// Whether the TensorOp takes all taps and words in one pass (k = 1, or
    /// one word per pixel) instead of one pass per 4 input channels.
    fn single_pass(&self) -> bool {
        self.k == 1 || self.cin_pad() == 4
    }

    /// The two input copy ops ([`copy_op`]); for Cin 1 / 3 they also widen
    /// each pixel to one 4-byte word.
    fn copy(
        &self,
        a: &mut Asm,
        pt: &Patch,
        par_base: i64,
        first: bool,
        row0: i64,
    ) -> Result<(), IsaError> {
        if self.cin < 4 {
            copy_op_small(a, self.cin as i64, pt, par_base, first, row0)
        } else {
            copy_op(a, pt, par_base, first, row0)
        }
    }

    /// Kernel taps.
    fn taps(&self) -> i64 {
        (self.k * self.k) as i64
    }

    /// The per-channel arithmetic as an FC over one pixel's window, with K
    /// ordered as the weight rows are streamed: `[Cin / 4][ky][kx][4]`.
    fn fc(&self) -> Fc {
        let (k, cin, cp) = (self.k, self.cin, self.cin_pad());
        let kk = k * k * cp;
        let mut weights = Vec::with_capacity(self.cout * kk);
        for o in 0..self.cout {
            for g in 0..cp / 4 {
                for t in 0..k * k {
                    for j in 0..4 {
                        let c = 4 * g + j;
                        weights.push(if c < cin {
                            self.weights[(o * k * k + t) * cin + c]
                        } else {
                            0
                        });
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

    /// Input channels as computed: at least 4 (Cin 1 and 3 are widened to
    /// one 4-byte word per pixel on chip, see [`Conv::input_small`]).
    fn cin_pad(&self) -> usize {
        self.cin.max(4)
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
        let (h, w) = (self.height as i64, self.width as i64);
        let (k, cin) = (self.k as i64, self.cin);
        let p = self.pad().0;
        let fc = self.fc();
        let pad = self.input.zero_point as u8;
        let mut out = Vec::new();
        for y in (0..h).step_by(self.stride) {
            for xx in (0..w).step_by(self.stride) {
                // window in FC order: [Cin / 4][ky][kx][4]
                let mut win = Vec::with_capacity(fc.k);
                for g in 0..self.cin_pad() / 4 {
                    for ky in 0..k {
                        for kx in 0..k {
                            let (sy, sx) = (y + ky - p, xx + kx - p);
                            for j in 0..4 {
                                let c = 4 * g + j;
                                let inside = (0..h).contains(&sy) && (0..w).contains(&sx);
                                win.push(if inside && c < cin {
                                    x[((sy * w + sx) as usize) * cin + c]
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
        self.build_unchecked()
    }

    /// [`Conv::build`] without [`Conv::check`], for hardware experiments on
    /// shapes outside the verified set.
    #[doc(hidden)]
    pub fn build_unchecked(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        let params = self.parameters();
        let caching = ExecInfo {
            exec_type: ExecutableType::PARAMETER_CACHING,
            parameters: params.clone(),
            instruction_chunks: vec![self.caching_program(params.len())?],
            input_layers: vec![],
            output_layers: vec![],
            scratch_size_bytes: 0,
        };
        let (h, w) = (self.height, self.width);
        let (oh, ow) = (h / self.stride, w / self.stride);
        let exec = ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![self.execution_program()?],
            input_layers: vec![layer("input", (h, w, self.cin), self.input)],
            output_layers: vec![LayerInfo {
                output_layout: Some(output_layout(
                    self.cout,
                    self.opx().0 as usize,
                    self.opx().1 as usize,
                )),
                ..layer("output", (oh, ow, self.cout), self.output)
            }],
            scratch_size_bytes: 0,
        };
        Ok(TfliteModel {
            executables: vec![exec, caching],
        })
    }

    /// Parameter rows of `Cout x 4` bytes: 3 header rows + `K / 4` weight rows.
    fn param_words(&self) -> i64 {
        3 + (self.k * self.k * self.cin_pad() / 4) as i64
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
        // `cq`: words per real pixel (conv op, weights); the patch, loader
        // and halo work in superpixels of `pt.cq()` words
        let (tp, cq) = (pt.tp(), self.cin_pad() as i64 / 4);
        let ((oy, ox), st) = (self.opx(), self.stride as i64);
        let npx = oy * ox; // output pixels per tile
        let ob = npx * self.cout as i64; // output bytes per tile
        // Input buffer in wide memory: one tile's input is `sw` 256 B words.
        // One word (Cin 16) lands in a single slot at 0x2078, overwritten
        // for each of the row's 4 x 1 words; more use a circular buffer of
        // `sw` slots below 0x2078, refilled once per tile of the row.
        let lb = pt.lb();
        let sw = lb / 256;
        // Cin 1 / 3: the host sends Cin bytes per pixel (`tbi` per tile) and
        // two constant words take 0x2078..; the buffer moves below them.
        let small = self.cin < 4;
        // (the host's bytes per tile: superpixel channels for wide images)
        let tbi = if small { PX * self.cin as i64 } else { pt.tb() };
        let buf = if small { 0x2070 } else { 0x2078 - 8 * (sw - 1) };
        let mut a = Asm::new();
        prologue(&mut a)?;

        // ── Constant setup (as in elementwise): one-hot words at narrow
        // tp + 2 tb, copied to wide just below the input buffer.
        let const_narrow = tp + 2 * lb;
        let const_wide = if small { 0x2078 } else { buf - 4 };
        for (i, fill) in [1i64, 0x100, 0x10000, 0x1000000].into_iter().enumerate() {
            let mut f = vec![
                ("tile_mask", ALL_TILES),
                ("i_addr", const_narrow + 4 * i as i64),
                ("i_inc0", 1),
                ("fill_en", 1),
                ("fill", fill),
            ];
            if i == 3 && !small {
                f.push(("in_mode", 3));
            }
            a.emit(OP_MESH_N, &f)?;
        }
        if small {
            // a zero word after the one-hot words; both go to wide 0x2078
            let mut f = vec![("tile_mask", ALL_TILES), ("fill_en", 1)];
            let w = mesh_walk("i", const_narrow + 16, &[(64, 1)]).0;
            let w: Vec<(&str, i64)> = w.iter().map(|(n, v)| (n.as_str(), *v)).collect();
            f.extend(w);
            a.emit(OP_MESH_N, &f)?;
            sync(&mut a, 0, 0x3ff, false)?;
            a.emit(
                OP_NARROW_TO_WIDE,
                &[
                    ("tile_mask", ALL_TILES),
                    ("narrow_addr", const_narrow / 4),
                    ("n_inc0", 1),
                    ("n_lim0", 0x43),
                    ("n_inc1", -0x43),
                    ("n_inc2", -0x43),
                    ("n_inc3", -0x43),
                    ("n_inc4", -0x43),
                    ("n_inc5", -0x43),
                    ("narrow_lvl_mask", 1),
                    ("wide_addr", const_wide),
                    ("w_inc0", 1),
                    ("w_lim0", 1),
                    ("w_inc1", -1),
                    ("w_inc2", -1),
                    ("w_inc3", -1),
                    ("wide_lvl_mask", 1),
                    ("tail_lvl", 1),
                    ("sync_en0", 0),
                    ("sync_en1", 0),
                    ("sync_en2", 0),
                ],
            )?;
        } else {
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
        }
        a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("units", 0x20)])?;
        self.copy(&mut a, &pt, const_wide & 0xff, true, 0)?;

        // ── Input: 4 tb bytes per row of 4 tiles through the input buffer,
        // landing at narrow tp; column c skips Cin * c words.
        load_input_addr(&mut a, IN_NAME)?;
        // the whole image is popped once; bands of 4 rows are infed in turn
        let bands = pt.rows / 4;
        let size = 16 * tbi * bands;
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
        if tbi >= 256 {
            a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("counters", 0x800)])?;
        }
        let (nw, land) = (tbi / 4, tp / 4); // narrow words per tile, landing word
        for h in 0..bands {
            // band h: the verified 4-row loader and copy ops, landing in the
            // same place, copied to rows 4 h.. of the patch
            if h > 0 {
                self.copy(&mut a, &pt, const_wide & 0xff, true, 4 * h)?;
            }
            // the buffer's wide-side walk: one slot, 4 words in turn, or `sw`
            // circular slots, 4 rounds
            let wide: Vec<(&str, i64)> = if tbi < 256 {
                // a row of 4 tiles is one word: one circular slot
                vec![
                    ("wide_addr", buf),
                    ("w_inc0", 1),
                    ("wide_circ", 1),
                    ("wide_rows", 1),
                    ("sync_wait_lvl", 1),
                ]
            } else if sw == 1 {
                vec![("wide_addr", buf), ("w_lim0", 3), ("wide_rows", 1)]
            } else {
                vec![
                    ("wide_addr", buf),
                    ("w_inc0", 1),
                    ("w_lim0", sw - 1),
                    ("w_inc1", -(sw - 1)),
                    ("w_lim1", 3),
                    ("w_inc2", -(sw - 1)),
                    ("w_inc3", -(sw - 1)),
                    ("wide_lvl_mask", 1),
                    ("wide_circ", 1),
                    ("wide_rows", sw),
                    ("sync_wait_lvl", 1),
                ]
            };
            for col in 0..4i64 {
                let mut f: Vec<(&str, i64)> = wide.clone();
                f.extend([
                    ("tile_mask", 0x1111 << col),
                    ("narrow_addr", land),
                    ("n_inc0", 1),
                    ("n_lim0", nw - 1),
                    ("n_lim1", if small { 4 } else { 3 }),
                    ("narrow_lvl_mask", 1),
                    ("mode", 1),
                    ("size64", lb / 64),
                ]);
                for n in ["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"] {
                    f.push((n, -(nw - 1)));
                }
                let skip = if small { self.cin as i64 } else { pt.cin } * col;
                // head of the first narrow row; for Cin 1 / 3 the compiler's
                // values (not derived; tail_words follows from them)
                let head = match self.cin {
                    1 => [43, 46, 45, 44][col as usize],
                    3 => [15, 12, 9, 6][col as usize],
                    _ => nw - 1 - skip,
                };
                if small {
                    f.extend([
                        ("head_words_m1", head),
                        ("head_en", 1),
                        ("tail_words", nw - (head + 1)),
                    ]);
                }
                if col > 0 {
                    if !small {
                        f.extend([
                            ("head_words_m1", head),
                            ("head_en", 1),
                            ("tail_words", skip),
                        ]);
                    }
                    f.extend([
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
            let per_row = 4 * tbi / 8; // infeed units of 8 B
            let consumer: Vec<(&str, i64)> = if tbi < 256 {
                vec![
                    ("addr", buf),
                    ("inc0", 1),
                    ("cbuf", 1),
                    ("slots", 1),
                    ("mode", 3),
                    ("s_id", 0x2b),
                ]
            } else if sw == 1 {
                vec![
                    ("addr", buf),
                    ("cnt0", 3),
                    ("slots", 1),
                    ("mode", 1),
                    ("s_id", 0xb),
                ]
            } else {
                vec![
                    ("addr", buf),
                    ("inc0", 1),
                    ("cnt0", sw - 1),
                    ("inc1", -(sw - 1)),
                    ("cnt1", 3),
                    ("inc2", -(sw - 1)),
                    ("inc3", -(sw - 1)),
                    ("sdims", 1),
                    ("cbuf", 1),
                    ("slots", sw),
                    ("mode", 3),
                    ("s_id", 0x2b),
                ]
            };
            for row in 0..4i64 {
                let mut f = consumer.clone();
                f.extend([
                    ("tile_mask", 0xf << (4 * row)),
                    ("s_val", -(lb / 4)),
                    ("s_cnt", sw),
                    ("s_en_b", 1),
                    ("s_y", 1),
                ]);
                a.emit(OP_RING_CONSUMER, &f)?;
                a.emit(
                    OP_INFEED,
                    &[
                        ("stream", 1),
                        ("buf_off", per_row * (row * bands + h)),
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
                        ("pop_wait", per_row * (row * bands + h) + 1),
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
            self.copy(&mut a, &pt, const_wide & 0xff, false, 4 * h)?;
            sync(&mut a, 0x1ffff, 0x3ff, false)?;
        }
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
        // Single pass over all taps and words (k = 1, or Cin = 4: one word
        // per pixel), or one pass per 4 input channels over the taps.
        let single = self.single_pass();
        let (slots, passes) = if single { (taps * cq, 1) } else { (taps, cq) };
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
            ("sync_val", 0x8000 * passes),
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
        let rs = pt.pw() * pt.cq(); // padded row stride in words
        // first window's origin: padded row 0 is the top halo; along x the
        // halo is whole superpixels, so start `left m - pad` real pixels in
        let (pad_y, pad_x) = (pt.top - self.pad().0, pt.left * self.spx() - self.pad().0);
        let in_base = 4 * (pad_y * rs + pad_x * cq);
        let k = self.k as i64;
        let (in_dims, loops): (Vec<(i64, i64)>, [i64; 3]) = if self.k == 1 {
            // per pixel: Cin / 4 words against the slots
            if st == 1 {
                (vec![(cq, 1), (1, cq), (npx, cq)], [cq - 1, npx - 1, 0])
            } else {
                (
                    vec![(cq, 1), (1, cq), (ox, st * cq), (oy, st * rs)],
                    [cq - 1, npx - 1, 0],
                )
            }
        } else if single {
            // per pixel: the k x k window, a row of k pixels contiguous
            (
                vec![(k * cq, 1), (k, rs), (1, cq), (ox, st * cq), (oy, st * rs)],
                [taps * cq - 1, npx - 1, 0],
            )
        } else {
            // per pass (4 channels): taps (x, y), then pixels (x, y)
            (
                vec![
                    (1, 1),
                    (k, cq),
                    (k, rs),
                    (ox, st * cq),
                    (oy, st * rs),
                    (cq, 1),
                ],
                [taps - 1, npx - 1, cq - 1],
            )
        };
        f.extend(walk("in", &in_dims, 8));
        if in_base != 0 {
            f.push(("in_base".into(), in_base));
        }
        for (i, l) in loops.iter().enumerate() {
            if *l > 0 {
                f.push((format!("loop{i}"), *l));
            }
        }
        let fixed: Vec<(&str, i64)> = vec![
            ("tile_mask", ALL_TILES),
            ("in_mode7", 3),
            (
                "in_tflags",
                if self.k == 1 {
                    if st == 1 { 1 } else { 3 }
                } else if single {
                    7
                } else {
                    0xf
                },
            ),
            ("out_base", tp),
            ("out_inc0", 1),
            ("out_cnt0", oq - 1),
            ("out_inc1", 1),
            ("out_cnt1", npx - 1),
            ("out_inc2", -(oq * (npx - 1) - 1)),
            ("out_mode7", 3),
            ("out_tflags", 1),
            ("par_base", slots_addr),
            ("par_inc0", 1),
            ("par_cnt0", slots - 1),
            ("par_inc1", -(slots - 1)),
            ("par_cnt1", npx - 1),
            ("par_mode7", 1),
            ("par_tflags", 0x80),
            ("par_fifo", slots),
            ("cfg0", 0xa5),
            ("cfg1", 0x14f),
            ("sync0", 0x4000),
            ("sync1", 0x4000),
            ("sync2", if single { 0xa6 } else { 0xb6 }),
            ("sync3", 0x80),
            ("sync4", 0x80),
            ("cfg2", if single { 7 } else { 3 }),
            ("cfg3", 8),
            ("in_zp", self.input.zero_point as i64),
            ("row_a", 0x7ff),
            ("row_m", 3),
            ("row_c", 0x7fc),
            ("dp_mode", if single { 2 } else { 3 }),
            ("reduce_mask", if single { 1 } else { 5 }),
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
            f.push((format!("out_inc{j}"), -(npx * oq - 1)));
        }
        for j in 2..8 {
            f.push((format!("par_inc{j}"), -(slots - 1)));
        }
        if single {
            f.extend([
                ("psum_cnt0".into(), slots - 1),
                ("psum_cnt1".into(), npx - 1),
                ("psum_tflags".into(), 0x840),
            ]);
        } else {
            // partial sums: one per pixel, revisited each pass
            f.extend([
                ("par_cnt2".into(), cq - 1),
                ("psum_base".into(), slots_addr - 4 * npx),
                ("psum_cnt0".into(), taps - 1),
                ("psum_inc1".into(), 1),
                ("psum_cnt1".into(), npx - 1),
                ("psum_cnt2".into(), cq - 1),
                ("psum_mode7".into(), 2),
            ]);
            for j in 2..8 {
                f.push((format!("psum_inc{j}"), -(npx - 1)));
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
            ("n_lim0", oq - 1),
            ("n_inc1", 1),
            ("n_lim1", ox - 1),
            ("n_inc2", 1),
            ("n_lim2", oy - 1),
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
/// minus their span; a loop of count 1 still gets one, from its nominal
/// stride. The levels after the last loop (up to `levels`) rewind to the
/// start.
fn walk(p: &str, dims: &[(i64, i64)], levels: usize) -> Vec<(String, i64)> {
    let mut f = Vec::new();
    let mut span = 0;
    for (j, &(n, stride)) in dims.iter().enumerate() {
        f.push((format!("{p}_inc{j}"), stride - span));
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

/// [`copy_op`] for Cin 1 / 3 with k = 3 (a 6x6 patch of 4-byte pixels):
/// the input walk takes Cin bytes per pixel (`in_hmode`, per-level
/// `in_mode` 1) and the parameter walk the one-hot constants per byte.
/// Transcribed from the compiler (Cin 1 and 3, 6x6 patch; the output walk
/// is copy_op's, so other patch sizes follow); the per-level modes and
/// `hmode` fields are not decoded.
fn copy_op_small(
    a: &mut Asm,
    cin: i64,
    pt: &Patch,
    par_base: i64,
    first: bool,
    row0: i64,
) -> Result<(), IsaError> {
    let (pw, tp) = (pt.pw(), pt.tp());
    let mut f: Vec<(String, i64)> = Vec::new();
    let mut push = |n: &str, v: i64| f.push((n.to_string(), v));
    for (n, v) in [
        ("tile_mask", ALL_TILES),
        ("loop0", cin - 1),
        ("loop1", 3),
        ("in_base", if first { pt.tp() } else { pt.tp() + pt.lb() }),
        ("in_hmode", 1),
        ("in_cnt0", cin - 1),
        ("in_cnt1", 3),
        ("in_inc2", 3 * cin),
        ("in_mode7", 3),
        ("in_tflags", 0xc1),
        ("in_twait", pt.lb() / 64),
        ("out_base", pt.at(row0 + if first { 0 } else { 3 }, 0)),
        ("out_inc0", 1),
        ("out_inc1", 1),
        ("out_cnt1", 3),
        ("out_inc2", pw - 3),
        ("out_mode7", 3),
        ("out_tflags", if first { 7 } else { 3 }),
        ("par_base", par_base),
        ("par_sel", 1),
        ("par_hmode", 1),
        ("par_cnt0", cin - 1),
        ("par_cnt1", 3),
        ("par_mode7", 1),
        ("psum_cnt0", cin - 1),
        ("psum_cnt1", 3),
        ("psum_tflags", 0x840),
        ("cfg0", 7),
        ("cfg1", if first { 0x6b } else { 0x6b + 0x100 * cin }),
        ("sync0", if first { cin } else { 3 * cin }),
        ("sync1", 0x4000 + cin),
        ("cfg2", 6),
        ("cfg3", 7),
        ("dp_mode", 3),
        ("reduce_mask", 1),
        ("out_ch_last", 3),
        ("out_ch", 3),
        ("mult_bits", 1.0f32.to_bits() as i64),
        ("clamp_max_bits", f32::INFINITY.to_bits() as i64),
        ("clamp_min_bits", f32::NEG_INFINITY.to_bits() as i64),
    ] {
        push(n, v);
    }
    if cin == 1 {
        push("psum_hmode", 1);
    }
    for j in 0..7 {
        push(&format!("in_mode{j}"), 1);
        if cin > 1 {
            push(&format!("par_mode{j}"), 2);
        }
    }
    for j in 3..8 {
        push(&format!("in_inc{j}"), -cin);
    }
    for j in 1..8 {
        // -1 for Cin 3, 0 for Cin 1 (two samples)
        push(&format!("par_inc{j}"), -((cin > 1) as i64));
    }
    if first {
        // rows 0-2: a third loop
        // copy_op's walk for one-word pixels, with the rows one level up
        for (n, v) in [
            ("loop3", 2),
            ("in_cnt3", 2),
            ("out_inc3", pw - 3),
            ("out_cnt3", 2),
            ("out_inc4", tp / 4 - 2 * pw - 3),
            ("par_cnt2", 2),
            ("psum_cnt2", 2),
        ] {
            push(n, v);
        }
        for j in 5..8 {
            push(&format!("out_inc{j}"), -11 - 2 * (pw - 4));
        }
    } else {
        push("out_inc3", tp / 4 - 3);
        for j in 4..8 {
            push(&format!("out_inc{j}"), -3);
        }
    }
    let f: Vec<(&str, i64)> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    a.emit(OP_TENSOR, &f)?;
    Ok(())
}

/// One halo move: interior part sent (`o_addr`) into the neighbour's halo
/// (`i_addr`), over `dims`. `groups` are the tile masks that only receive,
/// that send and receive, and that send and fill their own halo with the
/// zero point (no neighbour on that side). Column moves wait on sync
/// flags 6 / 0xa after the rows.
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
    // value the column moves wait for (not decoded): 5 for 1-word pixels,
    // 6 from 2 words up, with 1-pixel halos (corpus; 7 for deeper halos)
    let sync_val = if cq == 1 { 5 } else { 6 };
    // a pixel's words move in beats of at most 16; the beats of a wider
    // pixel are the outermost loop
    let (beat, beats) = (cq.min(16), (cq / 16).max(1));
    let rows = |n: i64| vec![(beat, 1), (4, cq), (n, rs), (beats, 16)];
    let ph = pt.ph();
    let cols = |n: i64| vec![(beat, 1), (n, cq), (ph, rs), (beats, 16)];
    let (t, b, l, r) = (pt.top, pt.bottom, pt.left, pt.right);
    let moves = [
        // north: top interior rows -> the northern tile's bottom halo
        Move {
            op: OP_MESH_N,
            o_addr: pt.at(0, 0),
            i_addr: pt.at(pt.rows, 0),
            dims: rows(b),
            groups: [0x000f, 0x0ff0, 0xf000],
            synced: false,
        },
        // south: bottom interior rows -> the southern tile's top halo
        Move {
            op: OP_MESH_S,
            o_addr: pt.at(pt.rows - t, 0),
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
    // column moves wait for the row moves that ran: flag 6 for the south
    // move, 0xa for the north one (both: 6 then 0xa)
    let flags: Vec<i64> = [(t > 0, 6), (b > 0, 0xa)]
        .iter()
        .filter(|f| f.0)
        .map(|f| f.1)
        .collect();
    for Move {
        op,
        o_addr,
        i_addr,
        dims,
        groups,
        synced,
    } in moves
    {
        // no halo on that side
        if dims[1].0 == 0 || dims[2].0 == 0 {
            continue;
        }
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
            // the beat split does not count as a level here
            let levels = n - (beats > 1) as i64;
            f.push(("in_mode".into(), 2 * levels + 1));
            if synced && role > 0 {
                for (j, id) in flags.iter().enumerate() {
                    f.extend([
                        (format!("s{j}_id"), *id),
                        (format!("s{j}_val"), sync_val),
                        (format!("s{j}_en_a"), 1),
                        (format!("s{j}_en_b"), 1),
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

/// Tile `4 * (y / qy) + x / qx` holds a `qy x qx` patch of the output,
/// rows of `qx` pixels x `nz` bytes; tiles are sent in order.
fn output_layout(nz: usize, qy: usize, qx: usize) -> OutputLayoutInfo {
    let (ny, nx) = (4 * qy, 4 * qx);
    OutputLayoutInfo {
        y_coordinate_to_linear_tile_id_map: (0..ny).map(|y| (4 * (y / qy)) as i32).collect(),
        x_coordinate_to_linear_tile_id_map: (0..nx).map(|x| (x / qx) as i32).collect(),
        linearized_tile_byte_offset: (0..16).map(|t| (t * qy * qx * nz) as i32).collect(),
        x_coordinate_to_local_byte_offset: (0..nx).map(|x| ((x % qx) * nz) as i32).collect(),
        y_coordinate_to_local_y_offset: (0..ny).map(|y| (y % qy) as i32).collect(),
        x_coordinate_to_local_y_row_size: vec![(qx * nz) as i32; nx],
    }
}
