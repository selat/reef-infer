//! Fully connected layer `y = W x + b` for `K` inputs and `N` outputs:
//! `K` a multiple of 8 up to 1024, `N` from 4 to 64 (parameters padded to
//! a multiple of 16 channels) or a multiple of 64 up to 1024. (N < 4 uses a
//! different op form in the compiler.)
//!
//! Follows `edgetpu_compiler` 16.0's program for a uint8 FULLY_CONNECTED
//! (`fc_K*_N*` in the ISA corpus): a PARAMETER_CACHING executable streams
//! the weights into the wide memory of the compute tiles once, and an
//! EXECUTION_ONLY executable runs one TensorOp per compute tile per
//! inference.
//!
//! Tiles: the input is spread 64 B per tile over `ceil(K / 64)` tiles (the
//! infeed lands in rows of 4 tiles, each tile keeping its 64 B with a
//! wideToNarrow skip), then gathered over the ring into every compute tile
//! in order. Output channels are split 64 per compute tile (tiles 0, 1, ..),
//! each with its own parameter blob, and each tile outfeeds its channels.
//!
//! Parameter blob per compute tile (decoded from the corpus), `n` its
//! channels padded to 16:
//!
//! | bytes          | content                                             |
//! |----------------|-----------------------------------------------------|
//! | `4n`           | f32 per-channel multiplier `s_in s_w[n] (1 / s_out)` |
//! | `4n`           | i32 bias                                            |
//! | `4n`           | i32 per-channel weight zero point (128 for int8 W)   |
//! | `n K`          | `W + 128` as uint8, ordered `[K/4][n][4]`             |
//!
//! In wide memory each block is one 256 B row (for n = 64): multipliers in
//! row 0, biases in row 1, zero points in row 2 (`op.row_a` / `row_c`), and
//! weights from 64 B unit 12 (`op.par_base`). `op.cfg4` enables the bias.
//! Verified on hardware: any per-channel weight zero point works when the
//! weights are stored with it, `row_a` selects the zero-point row, and with
//! `cfg4 = 0` the result equals a zero-bias layer.

use super::*;

/// A quantized FC layer. `weights` is `[N][K]` int8 with per-channel scales.
#[derive(Debug, Clone)]
pub struct Fc {
    pub k: usize,
    pub n: usize,
    pub weights: Vec<i8>,
    pub w_scale: Vec<f32>,
    pub bias: Vec<i32>,
    pub input: Quant,
    pub output: Quant,
    pub relu: bool,
}

impl Fc {
    pub fn check(&self) -> Result<(), String> {
        if !(self.k % 8 == 0 && (8..=4096).contains(&self.k) && !(2049..=3072).contains(&self.k)) {
            return Err(format!(
                "K={} must be a multiple of 8 in 8..=2048 or 3073..=4096",
                self.k
            ));
        }
        if !((4..=64).contains(&self.n) || (self.n % 64 == 0 && self.n <= 1024)) {
            return Err(format!(
                "N={} must be in 4..=64 or a multiple of 64 up to 1024",
                self.n
            ));
        }
        if self.weights.len() != self.n * self.k
            || self.w_scale.len() != self.n
            || self.bias.len() != self.n
        {
            return Err("weights / w_scale / bias sizes do not match K, N".into());
        }
        Ok(())
    }

    /// Output channels per compute tile (before padding).
    fn tile_n(&self) -> usize {
        self.n.min(64)
    }

    /// Compute tiles (64 output channels each).
    fn out_tiles(&self) -> usize {
        self.n.div_ceil(64)
    }

    /// Channels per compute tile padded to 16: the parameter blob and the
    /// output bytes per tile.
    fn tile_np(&self) -> usize {
        self.tile_n().next_multiple_of(16)
    }

    /// Output bytes per compute tile: its channels rounded up to 8 (the
    /// outfeed moves 8 B units).
    fn tile_ob(&self) -> usize {
        self.tile_n().next_multiple_of(8)
    }

    /// Bytes the host receives (padded channels included).
    pub fn out_bytes(&self) -> usize {
        self.out_tiles() * self.tile_ob()
    }

    /// Per-channel f32 multiplier, computed in f32 as the compiler does:
    /// `(s_in * s_w) * (1 / s_out)` (64/64 channels; `s_in * s_w / s_out`
    /// matches only 40).
    pub fn mult(&self, n: usize) -> f32 {
        self.input.scale * self.w_scale[n] * (1.0 / self.output.scale)
    }

    /// Clamp bounds as the compiler computes them, in f32:
    /// `((q - out_zp) * s) * (1 / s)` for q = 0 and 255 (RELU: 0). Matches
    /// every FC, conv and depthwise model in the corpus; the result sits on
    /// or one ulp inside the integer bound.
    pub fn clamp(&self) -> (f32, f32) {
        let (s, zp) = (self.output.scale, self.output.zero_point as f32);
        let bound = |q: f32| ((q - zp) * s) * (1.0 / s);
        (if self.relu { 0.0 } else { bound(0.0) }, bound(255.0))
    }

    pub fn parameters(&self) -> Vec<u8> {
        (0..self.out_tiles())
            .flat_map(|t| self.blob(64 * t..(64 * t + 64).min(self.n), self.tile_np()))
            .collect()
    }

    /// One parameter blob for output channels `chans`, padded to `np`
    /// channels. As the compiler pads: up to the next multiple of 4 the
    /// extra channels are computed (the first channel's multiplier, zero
    /// bias and weights, zero point 128); beyond that everything is 0.
    pub(crate) fn blob(&self, chans: std::ops::Range<usize>, np: usize) -> Vec<u8> {
        let live = (chans.end - chans.start).next_multiple_of(4);
        let ch = |c: usize| Some(chans.start + c).filter(|&n| n < chans.end);
        let mut p = Vec::with_capacity(np * (12 + self.k));
        for c in 0..np {
            let m = if c < live {
                self.mult(ch(c).unwrap_or(chans.start))
            } else {
                0.0
            };
            p.extend(m.to_le_bytes());
        }
        for c in 0..np {
            p.extend(ch(c).map_or(0, |n| self.bias[n]).to_le_bytes());
        }
        for c in 0..np {
            p.extend(if c < live { 128i32 } else { 0 }.to_le_bytes());
        }
        for k4 in 0..self.k / 4 {
            for c in 0..np {
                for j in 0..4 {
                    p.push(match ch(c) {
                        Some(n) => (self.weights[n * self.k + 4 * k4 + j] as i16 + 128) as u8,
                        None if c < live => 128,
                        None => 0,
                    });
                }
            }
        }
        p
    }

    /// Host reference: int32 accumulate, f32 multiply, round half to even,
    /// clamp, add the output zero point.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        let (lo, hi) = self.clamp();
        (0..self.n)
            .map(|n| {
                let acc: i32 = (0..self.k)
                    .map(|k| {
                        (x[k] as i32 - self.input.zero_point) * self.weights[n * self.k + k] as i32
                    })
                    .sum::<i32>()
                    + self.bias[n];
                let v = (acc as f32 * self.mult(n)).clamp(lo, hi).round_ties_even();
                (v + self.output.zero_point as f32).clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Builds the PARAMETER_CACHING and EXECUTION_ONLY executables.
    pub fn build(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.build_at(0)
    }

    /// [`Fc::build`] with the parameters cached from wide address `base`
    /// (64 B units, a whole number of 256 B rows) instead of 0, so that
    /// several layers' parameters can stay cached at once.
    pub fn build_at(&self, base: i64) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.build_with_at(self.parameters(), base)
    }

    /// Wide address units (64 B) the parameters take on each compute tile
    /// from their base: the 3 header rows and a row per 4 inputs.
    pub fn wide_units(&self) -> i64 {
        4 * (3 + self.k as i64 / 4)
    }

    /// [`Fc::build`] with a caller-supplied parameter blob (for experiments).
    pub fn build_with(&self, params: Vec<u8>) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.build_with_at(params, 0)
    }

    fn build_with_at(
        &self,
        params: Vec<u8>,
        base: i64,
    ) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.check()?;
        check_base(base)?;
        let caching = ExecInfo {
            exec_type: ExecutableType::PARAMETER_CACHING,
            parameters: params.clone(),
            instruction_chunks: vec![self.caching_program(params.len(), base)?],
            input_layers: vec![],
            output_layers: vec![],
            scratch_size_bytes: 0,
        };
        let mut output = layer("output", (1, 1, self.n), self.output);
        output.size_bytes = self.out_bytes() as i32;
        // channels of compute tile t at t x row; one 1x1 pixel
        let row = self.tile_n().next_multiple_of(4);
        output.output_layout = Some(OutputLayoutInfo {
            y_coordinate_to_linear_tile_id_map: vec![0],
            x_coordinate_to_linear_tile_id_map: vec![0],
            linearized_tile_byte_offset: (0..16)
                .map(|t| (t.min(self.out_tiles()) * row) as i32)
                .collect(),
            x_coordinate_to_local_byte_offset: vec![0],
            y_coordinate_to_local_y_offset: vec![0],
            x_coordinate_to_local_y_row_size: vec![row as i32],
        });
        let exec = ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![self.execution_program(base)?],
            input_layers: vec![layer("input", (1, 1, self.k), self.input)],
            output_layers: vec![output],
            scratch_size_bytes: 0,
        };
        Ok(TfliteModel {
            executables: vec![exec, caching],
        })
    }

    /// Streams each compute tile's parameter blob into its wide memory from
    /// `base`: header rows at `base`, `base + 4`, `base + 8`, weights from
    /// `base + 12` (the consumer's `addr` is 2 above where its rows land,
    /// as in conv).
    fn caching_program(&self, bytes: usize, base: i64) -> Result<ChunkInfo, IsaError> {
        let kq = (self.k / 4) as i64;
        let tiles = self.out_tiles();
        let consumer = |t: usize| {
            vec![
                ("tile_mask", 1i64 << t),
                ("addr", base + 0xe),
                ("inc0", 1),
                ("cnt0", kq - 1),
                ("inc1", -(kq - 1)),
                ("inc2", -(kq - 1)),
                ("inc3", -(kq - 1)),
                ("sdims", 1),
                ("aux_en0", 1),
                ("aux_addr0", base),
                ("aux_addr1", base + 8),
                ("aux_en1", 1),
                // unexplained bits, set by the compiler with aux_en0/aux_en1
                ("rsv286", 2),
                ("rsv319", 1),
            ]
        };
        let per_tile: Vec<_> = (0..tiles)
            .map(|t| (1i64 << t, bytes / tiles, consumer(t)))
            .collect();
        let mut a = Asm::new();
        // 16-channel groups per tile - 1
        param_caching_tiles(&mut a, &per_tile, (self.tile_np() / 16) as i64 - 1)?;
        a.finish()
    }

    fn execution_program(&self, base: i64) -> Result<ChunkInfo, IsaError> {
        let k = self.k as i64;
        let kq = k / 4;
        let nq = self.tile_n().div_ceil(4) as i64;
        let ctiles = self.out_tiles() as i64;
        let cmask = (1i64 << ctiles) - 1; // compute tiles
        // Input tiles take `ch` bytes each: 64 up to K = 1024, then 128 / 256
        // so that 16 tiles hold up to 4096 (K 2049..=3072 not derived).
        let ch = 64 * self.k.div_ceil(1024).next_power_of_two() as i64;
        let itiles = (k + ch - 1) / ch;
        let in_bytes = |t: i64| (k - ch * t).min(ch);
        // narrow memory placement (bytes); the compiler moves the input up
        // when it does not fill the first 64 B
        let (in_base, out_base) = if k < 64 { (64, 0) } else { (0, k) };
        // Staging: `slots` wide words (256 B) hold the whole input, below
        // 0x2080; it lands in narrow memory above the input and output.
        let rows = (itiles + 3) / 4;
        let slots = (k + 255) / 256;
        let stage_w = 0x2080 - 8 * slots;
        let stage_n = ((k + 255) / 256 * 256 + 512) / 4;
        let mut a = Asm::new();
        prologue(&mut a)?;
        a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("units", 0x20)])?;

        // ── Input: K bytes from the host, 64 B per input tile.
        load_input_addr(&mut a, "input")?;
        let units8 = k / 8;
        // A one-unit transfer is encoded with the loop fields cleared
        // (count_m1 = 0, no d0 limit / strides, k157 / k203 = 0); a hang
        // otherwise (K = 8 on hardware).
        let multi = units8 > 1;
        let pop: Vec<(&str, i64)> = [
            ("stream", 1),
            ("d0_stride", 1),
            ("k121", 1),
            ("k124", 1),
            ("av140", 1),
            ("count_m1", units8 - 1),
            ("k174", 1),
            ("k178", 1),
            ("k199", 1),
            ("k215", 1),
        ]
        .into_iter()
        .chain(
            [
                ("d0_limit", 0x3fff),
                ("d1_stride", 0xc001),
                ("d2_stride", 0xc001),
                ("k157", 1),
                ("count_neg", 0x4000 - units8),
            ]
            .into_iter()
            .filter(|_| multi),
        )
        .collect();
        a.emit(OP_POP, &pop)?;
        descriptor_addr(&mut a)?;
        descriptor_send(&mut a, k, TAG_INPUT)?;
        // Each tile of a row receives the row's bytes and keeps its own `ch`:
        // column c reads wide word `c ch / 256` of the row, skipping the
        // words before its bytes. Full tiles go by column, a partial last
        // tile on its own.
        let full = if k % ch == 0 { itiles } else { itiles - 1 };
        let mut classes: Vec<(i64, i64, i64)> = (0..4.min(itiles))
            .map(|c| {
                let mask = (c..full).step_by(4).fold(0, |m, t| m | 1 << t);
                (mask, c, ch)
            })
            .filter(|&(m, _, _)| m != 0)
            .collect();
        if full < itiles {
            classes.push((1 << full, full % 4, in_bytes(full)));
        }
        for (mask, c, bytes) in classes {
            let skip = (c * ch % 256) / 4;
            let mut f = vec![
                ("tile_mask", mask),
                ("wide_addr", stage_w + 4 * (c * ch / 256)),
                ("w_inc0", 1),
                ("narrow_addr", stage_n),
                ("mode", 1),
                ("size64", 4),
                ("sync_en0", 1),
                ("sync_id", 0xe),
                ("sync_wait_lvl", 1),
                ("sync_en1", 1),
                ("sync_val", 0x8000),
                ("sync_en2", 1),
                ("sync_dec_lvl", 1),
                ("sync_dec", -1),
                ("sync_dec_mode", 3),
            ];
            if skip > 0 {
                f.extend([
                    ("skip_en0", 1),
                    ("skip_base", stage_n),
                    ("skip_m1", skip - 1),
                    ("skip_end", stage_n + skip - 1),
                    ("tail_en870", 0),
                    ("tail_f871", 1),
                    ("rsv543", 1),
                ]);
            }
            f.extend(walk(bytes / 4 - 1));
            a.emit(OP_WIDE_TO_NARROW, &f)?;
        }
        for r in 0..rows {
            let mask = (4 * r..(4 * r + 4).min(itiles)).fold(0, |m, t| m | 1 << t);
            let mut f = vec![
                ("tile_mask", mask),
                ("addr", stage_w),
                ("inc0", 1),
                ("cbuf", 1),
                ("slots", slots),
                ("mode", 3),
                ("s_id", 0x2b),
                ("s_val", -64 * slots),
                ("s_cnt", slots),
                ("s_en_b", 1),
                ("s_y", 1),
            ];
            if slots > 1 {
                f.push(("sdims", 1));
            }
            // above 1024 B a full row of 4 tiles walks all slots, in groups
            // (`grp` = words per tile - 1; `gstride` fits K 2048 / 3136 / 4096)
            if ch > 64 && in_bytes(4 * r + 3) == ch {
                f.extend([
                    ("cnt0", slots - 1),
                    ("inc1", -(slots - 1)),
                    ("inc2", -(slots - 1)),
                    ("inc3", -(slots - 1)),
                    ("grp", ch / 64 - 1),
                    ("gstride", 3 * (k / 256) + 1),
                ]);
            }
            a.emit(OP_RING_CONSUMER, &f)?;
            let units8 = (k - 4 * ch * r).min(4 * ch) / 8;
            let infeed: Vec<(&str, i64)> = [
                ("stream", 1),
                ("buf_off", ch / 2 * r),
                ("d0_stride", 1),
                ("k157", 1),
                ("k160", 1),
                ("av176", 1),
                ("count_m1", units8 - 1),
                ("k311", 1),
                ("pop_wait", ch / 2 * r + 1),
                ("k336", 1),
                ("k352", 1),
                ("k392", 1),
                ("tiles", mask),
                ("k434", 0xf),
                ("f438", 3),
            ]
            .into_iter()
            .chain(
                [
                    ("d0_limit", 0x3fff),
                    ("d1_stride", 0x1fc001),
                    ("d2_stride", 0x1fc001),
                    ("k203", 1),
                    ("count_neg", 0x4000 - units8),
                ]
                .into_iter()
                .filter(|_| multi),
            )
            .collect();
            a.emit(OP_INFEED, &infeed)?;
        }
        sync(&mut a, 0, 0x3ff, false)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Gather: input tile t sends its 64 B over the ring to every
        // compute tile, landing at narrow `in_base + 64 t`.
        for t in 0..itiles {
            let mut f = vec![
                ("tile_mask", cmask),
                ("wide_addr", 0x2078),
                ("w_inc0", 1),
                ("wide_circ", 1),
                ("wide_rows", 1),
                ("narrow_addr", (in_base + ch * t) / 4),
                ("sync_en0", 1),
                ("sync_id", 0xe),
                ("sync_wait_lvl", 1),
                ("sync_en1", (t % 2 == 0) as i64),
                ("sync_val", 0x8000 + (t + 1) / 2),
                ("sync_en2", 1),
            ];
            f.extend(walk(in_bytes(t) / 4 - 1));
            a.emit(OP_WIDE_TO_NARROW, &f)?;
            ring_consumer(&mut a, cmask, -64 + ch / 4 * t)?;
            ring_producer(&mut a, 1 << t, 0x2068, t, cmask)?;
            let mut f = vec![("tile_mask", 1 << t), ("narrow_addr", stage_n)];
            f.extend(walk(in_bytes(t) / 4 - 1));
            f.extend(n2w_tail(0x2068));
            a.emit(OP_NARROW_TO_WIDE, &f)?;
        }
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        a.emit(OP_SCSYNC, &[("sunits", 0xc)])?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── Per-channel multipliers / biases: wide `base` -> scaling memory
        // (mode 2).
        a.emit(
            OP_WIDE_TO_NARROW,
            &[
                ("tile_mask", cmask),
                ("wide_addr", base),
                ("w_inc0", 1),
                ("w_lim0", 1),
                ("w_inc1", -1),
                ("w_inc2", -1),
                ("w_inc3", -1),
                ("wide_lvl_mask", 1),
                ("n_inc0", 1),
                ("n_lim0", 0x1f),
                ("n_inc1", 0x21),
                ("n_lim1", 1),
                ("n_inc2", -95),
                ("n_inc3", -95),
                ("n_inc4", -95),
                ("n_inc5", -95),
                ("narrow_lvl_mask", 3),
                ("mode", 2),
                ("size64", 2),
                ("sync_en0", 1),
                ("sync_f2", 1),
                ("sync_wait_lvl", 2),
                ("sync_en1", 1),
                ("sync_val", 0xffff),
                ("sync_en2", 1),
                ("sync_f45", 1),
                ("tail_f867", 1),
            ],
        )?;

        // ── The FC TensorOp, on every compute tile.
        let (lo, hi) = self.clamp();
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", cmask),
            ("loop0", kq - 1),
            ("in_base", in_base),
            ("in_inc0", 1),
            ("in_cnt0", kq - 1),
            ("in_inc1", 1),
            ("in_inc2", 1),
            ("in_mode7", 3),
            ("in_tflags", 1),
            ("out_base", out_base),
            ("out_inc0", 1),
            ("out_cnt0", nq - 1),
            ("out_inc1", 1),
            ("out_inc2", 1),
            ("out_mode7", 3),
            ("out_tflags", 1),
            ("par_base", base + 0xc), // weights start after the 3 header rows
            ("par_inc0", 1),
            ("par_cnt0", kq - 1),
            ("par_mode7", 1),
            ("psum_cnt0", kq - 1),
            ("cfg0", 0x65),
            ("cfg1", 0x12c),
            ("sync0", 0x4000),
            ("sync1", 0x4000),
            ("cfg2", 7),
            ("cfg3", 8),
            ("in_zp", self.input.zero_point as i64),
            ("row_a", base / 4 + 2), // wide row of the weight zero points
            ("row_m", 1),
            ("row_c", base / 4 + 2),
            ("dp_mode", 1),
            ("reduce_mask", 1),
            ("out_ch_last", 4 * nq - 1),
            ("out_ch", 4 * nq - 1),
            ("mult_bits", 1.0f32.to_bits() as i64),
            ("clamp_max_bits", hi.to_bits() as i64),
            ("clamp_min_bits", lo.to_bits() as i64),
            ("out_zp", self.output.zero_point as i64),
            ("cfg4", 1), // add bias
            ("row_cfg", 1),
        ];
        for k in 3..8 {
            f.push((
                [
                    "", "", "", "in_inc3", "in_inc4", "in_inc5", "in_inc6", "in_inc7",
                ][k],
                -(kq - 1),
            ));
            f.push((
                [
                    "", "", "", "out_inc3", "out_inc4", "out_inc5", "out_inc6", "out_inc7",
                ][k],
                -(nq - 1),
            ));
        }
        for k in 1..8 {
            f.push((
                [
                    "", "par_inc1", "par_inc2", "par_inc3", "par_inc4", "par_inc5", "par_inc6",
                    "par_inc7",
                ][k],
                -(kq - 1),
            ));
        }
        a.emit(OP_TENSOR, &f)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0x1ffff, 0x3ff, true)?;
        wait_tiles(&mut a)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Output: narrow `out_base` -> wide row 0x2078 -> ring -> host,
        // one outfeed per compute tile.
        load_output_addr(&mut a, "output")?;
        descriptor_addr(&mut a)?;
        let mut f = vec![("tile_mask", cmask), ("narrow_addr", out_base / 4)];
        f.extend(walk(nq - 1));
        f.extend(n2w_tail(0x2078));
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        descriptor_send(&mut a, self.out_bytes() as i64, TAG_OUTPUT)?;
        // a one-unit outfeed clears the loop (as the one-unit input pop)
        let ou = (self.tile_ob() / 8) as i64;
        for t in 0..ctiles {
            a.emit(
                OP_OUTFEED,
                &[
                    ("d0_stride", (ou == 1) as i64),
                    ("d0_limit", ou - 1),
                    ("k173", 3),
                    ("f222", 0x7f),
                ],
            )?;
            ring_producer(&mut a, 1 << t, 0x2078, itiles + t, 0x10000)?;
        }
        epilogue(&mut a, itiles + ctiles, 0)?;
        a.finish()
    }
}

/// Narrow-side loop of a one-level move of `n_lim + 1` words.
fn walk(n_lim: i64) -> Vec<(&'static str, i64)> {
    let mut f = vec![
        ("n_inc0", 1),
        ("n_lim0", n_lim),
        ("narrow_lvl_mask", (n_lim > 0) as i64),
    ];
    f.extend(["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"].map(|n| (n, -n_lim)));
    f
}

fn ring_consumer(a: &mut Asm, tiles: i64, s_val: i64) -> Result<(), IsaError> {
    a.emit(
        OP_RING_CONSUMER,
        &[
            ("tile_mask", tiles),
            ("addr", 0x2078),
            ("inc0", 1),
            ("cbuf", 1),
            ("slots", 1),
            ("mode", 3),
            ("s_id", 0x2b),
            ("s_val", s_val),
            ("s_cnt", 1),
            ("s_en_b", 1),
            ("s_y", 1),
        ],
    )?;
    Ok(())
}

/// `tiles` produce one row from wide `addr` to `dest` (bit t = tile t,
/// bit 16 = scalar core / outfeed); `r0_val` counts the program's rows.
fn ring_producer(
    a: &mut Asm,
    tiles: i64,
    addr: i64,
    r0_val: i64,
    dest: i64,
) -> Result<(), IsaError> {
    a.emit(
        OP_RING_PRODUCER,
        &[
            ("tile_mask", tiles),
            ("addr", addr),
            ("inc0", 1),
            ("sdims", 1),
            ("cbuf", 1),
            ("slots", 1),
            ("mode", 3),
            ("pcfg", 0x14),
            ("r0_id", 0x51),
            ("r0_val", r0_val),
            ("r0_en_a", 1),
            ("r0_en_b", 1),
            ("r1_id", 0x2d),
            ("r1_val", 1),
            ("r1_en_a", 1),
            ("r1_en_b", 1),
            ("dest", dest),
        ],
    )?;
    Ok(())
}

/// Wide-side fields of a narrowToWide into one circular row at `wide_addr`.
fn n2w_tail(wide_addr: i64) -> [(&'static str, i64); 13] {
    [
        ("tail_lvl", 1),
        ("wide_addr", wide_addr),
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
    ]
}
