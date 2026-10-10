//! Fully connected layer `y = W x + b` on tile 0, for `K` inputs and `N`
//! outputs (`K` a multiple of 8 and `N` a multiple of 16, both up to 64).
//!
//! Follows `edgetpu_compiler` 16.0's program for a uint8 FULLY_CONNECTED
//! (`fc_K64_N64_..._iouint8` in the ISA corpus): a PARAMETER_CACHING
//! executable streams the weights into tile 0's wide memory once, and an
//! EXECUTION_ONLY executable runs one TensorOp per inference.
//!
//! Parameter blob (decoded from the corpus):
//!
//! | bytes          | content                                             |
//! |----------------|-----------------------------------------------------|
//! | `4N`           | f32 per-channel multiplier `s_in s_w[n] (1 / s_out)` |
//! | `4N`           | i32 bias                                            |
//! | `4N`           | i32 per-channel weight zero point (128 for int8 W)   |
//! | `N K`          | `W + 128` as uint8, ordered `[K/4][N][4]`             |
//!
//! In wide memory each block is one 256 B row (for N = 64): multipliers in
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
        if !(self.k % 8 == 0 && (8..=64).contains(&self.k)) {
            return Err(format!("K={} must be a multiple of 8 in 8..=64", self.k));
        }
        if !(self.n % 16 == 0 && (16..=64).contains(&self.n)) {
            return Err(format!("N={} must be a multiple of 16 in 16..=64", self.n));
        }
        if self.weights.len() != self.n * self.k
            || self.w_scale.len() != self.n
            || self.bias.len() != self.n
        {
            return Err("weights / w_scale / bias sizes do not match K, N".into());
        }
        Ok(())
    }

    /// Per-channel f32 multiplier, computed in f32 as the compiler does:
    /// `(s_in * s_w) * (1 / s_out)` (64/64 channels; `s_in * s_w / s_out`
    /// matches only 40).
    pub fn mult(&self, n: usize) -> f32 {
        self.input.scale * self.w_scale[n] * (1.0 / self.output.scale)
    }

    /// Clamp bounds: the uint8 range relative to `out_zp`, pulled one ulp
    /// towards zero as the compiler emits them (RELU clamps at 0).
    pub fn clamp(&self) -> (f32, f32) {
        let zp = self.output.zero_point as f32;
        let toward_zero = |v: f32| {
            if v == 0.0 {
                v
            } else {
                f32::from_bits(v.to_bits() - 1)
            }
        };
        let lo = if self.relu { 0.0 } else { toward_zero(-zp) };
        (lo, toward_zero(255.0 - zp))
    }

    pub fn parameters(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(self.n * (12 + self.k));
        for n in 0..self.n {
            p.extend(self.mult(n).to_le_bytes());
        }
        for &b in &self.bias {
            p.extend(b.to_le_bytes());
        }
        for _ in 0..self.n {
            p.extend(128i32.to_le_bytes());
        }
        for k4 in 0..self.k / 4 {
            for n in 0..self.n {
                for j in 0..4 {
                    p.push((self.weights[n * self.k + 4 * k4 + j] as i16 + 128) as u8);
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
        self.build_with(self.parameters())
    }

    /// [`Fc::build`] with a caller-supplied parameter blob (for experiments).
    pub fn build_with(&self, params: Vec<u8>) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.check()?;
        let caching = ExecInfo {
            exec_type: ExecutableType::PARAMETER_CACHING,
            parameters: params.clone(),
            instruction_chunks: vec![self.caching_program(params.len())?],
            input_layers: vec![],
            output_layers: vec![],
            scratch_size_bytes: 0,
        };
        let exec = ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![self.execution_program()?],
            input_layers: vec![layer("input", (1, 1, self.k), self.input)],
            output_layers: vec![layer("output", (1, 1, self.n), self.output)],
            scratch_size_bytes: 0,
        };
        Ok(TfliteModel {
            executables: vec![exec, caching],
        })
    }

    /// Streams the parameter blob from the host into tile 0's wide memory.
    fn caching_program(&self, bytes: usize) -> Result<ChunkInfo, IsaError> {
        use Description as D;
        use Position as P;
        let units = (bytes / 64) as i64; // the parameter stream moves 64 B units
        let kq = (self.k / 4) as i64;
        let mut a = Asm::new();
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
        mov_reg(&mut a, 3, 1)?;
        mov_reg(&mut a, 2, 0)?;
        param_descriptor(&mut a, bytes as i64, TAG_PARAMS)?;
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
        a.emit(
            OP_RING_CONSUMER,
            &[
                ("tile_mask", 1),
                ("addr", 0xe),
                ("inc0", 1),
                ("cnt0", kq - 1),
                ("inc1", -(kq - 1)),
                ("inc2", -(kq - 1)),
                ("inc3", -(kq - 1)),
                ("sdims", 1),
                ("aux_en0", 1),
                ("aux_addr1", 8),
                ("aux_en1", 1),
                // unexplained bits, set by the compiler with aux_en0/aux_en1
                ("rsv286", 2),
                ("rsv319", 1),
            ],
        )?;
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
                ("f438", (self.n / 16) as i64 - 1), // 16-channel groups - 1
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
        wait_tiles(&mut a)?;
        a.mov(2, 0)?;
        a.mov(3, 0)?;
        param_descriptor(&mut a, 0, TAG_DONE)?;
        a.finish()
    }

    fn execution_program(&self) -> Result<ChunkInfo, IsaError> {
        let (kq, nq) = ((self.k / 4) as i64, (self.n / 4) as i64);
        // narrow memory placement (bytes); the compiler moves the input up
        // when it does not fill the first 64 B
        let (in_base, out_base) = if self.k == 64 { (0, 64) } else { (64, 0) };
        let mut a = Asm::new();
        prologue(&mut a)?;
        a.emit(OP_SYNC, &[("tiles", ALL_TILES), ("units", 0x20)])?;

        // ── Input: K bytes from the host into narrow word 0xc0 of tile 0.
        load_input_addr(&mut a, "input")?;
        let units8 = (self.k / 8) as i64;
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
        descriptor_send(&mut a, self.k as i64, TAG_INPUT)?;
        let walk = |n_lim: i64| -> Vec<(&'static str, i64)> {
            let mut f = vec![("n_inc0", 1), ("n_lim0", n_lim), ("narrow_lvl_mask", 1)];
            f.extend(["n_inc1", "n_inc2", "n_inc3", "n_inc4", "n_inc5"].map(|n| (n, -n_lim)));
            f
        };
        let mut f = vec![
            ("tile_mask", 1),
            ("wide_addr", 0x2078),
            ("w_inc0", 1),
            ("narrow_addr", 0xc0),
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
        f.extend(walk(kq - 1));
        a.emit(OP_WIDE_TO_NARROW, &f)?;
        ring_consumer(&mut a)?;
        let infeed: Vec<(&str, i64)> = [
            ("stream", 1),
            ("d0_stride", 1),
            ("k157", 1),
            ("k160", 1),
            ("av176", 1),
            ("count_m1", units8 - 1),
            ("k311", 1),
            ("pop_wait", 1),
            ("k336", 1),
            ("k352", 1),
            ("k392", 1),
            ("tiles", 1),
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
        sync(&mut a, 0, 0x3ff, false)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;

        // ── Loopback: tile 0 sends the staged input over the ring to itself,
        // landing at narrow `in_base`.
        let mut f = vec![
            ("tile_mask", 1),
            ("wide_addr", 0x2078),
            ("w_inc0", 1),
            ("wide_circ", 1),
            ("wide_rows", 1),
            ("narrow_addr", in_base / 4),
            ("sync_en0", 1),
            ("sync_id", 0xe),
            ("sync_wait_lvl", 1),
            ("sync_en1", 1),
            ("sync_val", 0x8000),
            ("sync_en2", 1),
        ];
        f.extend(walk(kq - 1));
        a.emit(OP_WIDE_TO_NARROW, &f)?;
        ring_consumer(&mut a)?;
        ring_producer(&mut a, 0x2068, 0, 1)?;
        let mut f = vec![("tile_mask", 1), ("narrow_addr", 0xc0)];
        f.extend(walk(kq - 1));
        f.extend(n2w_tail(0x2068));
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        sync(&mut a, 0, 0x3ff, true)?;
        wait_tiles(&mut a)?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;
        a.emit(OP_SCSYNC, &[("sunits", 0xc)])?;
        sync(&mut a, 0x1ffff, 0x3ff, false)?;

        // ── Per-channel multipliers / biases: wide 0 -> scaling memory (mode 2).
        a.emit(
            OP_WIDE_TO_NARROW,
            &[
                ("tile_mask", 1),
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

        // ── The FC TensorOp.
        let (lo, hi) = self.clamp();
        let mut f: Vec<(&str, i64)> = vec![
            ("tile_mask", 1),
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
            ("par_base", 0xc), // weights start after the 768 B header
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
            ("row_a", 2), // wide row of the weight zero points
            ("row_m", 1),
            ("row_c", 2),
            ("dp_mode", 1),
            ("reduce_mask", 1),
            ("out_ch_last", self.n as i64 - 1),
            ("out_ch", self.n as i64 - 1),
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

        // ── Output: narrow `out_base` -> wide row 0x2078 -> ring -> host.
        load_output_addr(&mut a, "output")?;
        descriptor_addr(&mut a)?;
        let mut f = vec![("tile_mask", 1), ("narrow_addr", out_base / 4)];
        f.extend(walk(nq - 1));
        f.extend(n2w_tail(0x2078));
        a.emit(OP_NARROW_TO_WIDE, &f)?;
        descriptor_send(&mut a, self.n as i64, TAG_OUTPUT)?;
        a.emit(
            OP_OUTFEED,
            &[
                ("d0_limit", (self.n / 8) as i64 - 1),
                ("k173", 3),
                ("f222", 0x7f),
            ],
        )?;
        ring_producer(&mut a, 0x2078, 1, 0x10000)?;
        epilogue(&mut a, 2, 0)?;
        a.finish()
    }
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
            ("s_op", 0x2f),
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
            ("s_op", 0x2f),
            ("s_x", 7),
            ("imm_scalar", tag),
        ],
    )?;
    a.emit(OP_SCALAR, &[("v_op", 0xa), ("v_offset", 3), ("vs_reg", 7)])?;
    a.emit(OP_SCALAR, &[("v_op", 0xc)])?;
    Ok(())
}

fn ring_consumer(a: &mut Asm) -> Result<(), IsaError> {
    a.emit(
        OP_RING_CONSUMER,
        &[
            ("tile_mask", 1),
            ("addr", 0x2078),
            ("inc0", 1),
            ("cbuf", 1),
            ("slots", 1),
            ("mode", 3),
            ("s_id", 0x2b),
            ("s_val", -64),
            ("s_cnt", 1),
            ("s_en_b", 1),
            ("s_y", 1),
        ],
    )?;
    Ok(())
}

/// Tile 0 produces one row from wide `addr` to `dest` (bit t = tile t,
/// bit 16 = scalar core / outfeed).
fn ring_producer(a: &mut Asm, addr: i64, r0_val: i64, dest: i64) -> Result<(), IsaError> {
    a.emit(
        OP_RING_PRODUCER,
        &[
            ("tile_mask", 1),
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
