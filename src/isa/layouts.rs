//! Field tables for every decoded opcode, ported from tinycoral `coral/isa/`
//! (`op.py`, `wide_narrow.py`, `ring_mesh.py`, `scalar.py`, `scalar_core.py`,
//! `eltwise.py`, `__init__.py`). Field names match tinycoral so its docs
//! (`docs/isa/*.md`) apply directly.

use super::Layout;

/// Tile sync counters; bit `k` of a `sync.counters` mask is `TILE_COUNTERS[k]`.
pub const TILE_COUNTERS: [&str; 19] = [
    "AVDATA",
    "PARAMETERS",
    "PARTIAL_SUMS",
    "MESH_NORTH_IN",
    "MESH_EAST_IN",
    "MESH_SOUTH_IN",
    "MESH_WEST_IN",
    "MESH_NORTH_OUT",
    "MESH_EAST_OUT",
    "MESH_SOUTH_OUT",
    "MESH_WEST_OUT",
    "WIDE_TO_NARROW",
    "WIDE_TO_SCALING",
    "NARROW_TO_WIDE",
    "RING_READ_A",
    "RING_READ_B",
    "RING_WRITE",
    "RING_PRODUCER_A",
    "RING_PRODUCER_B",
];

/// Scalar sync counters; bit `k` of a `scsync.smask` is `SCALAR_COUNTERS[k]`.
pub const SCALAR_COUNTERS: [&str; 9] = [
    "AVDATA_POP",
    "PARAMETER_POP",
    "AVDATA_INFEED",
    "PARAMETER_INFEED",
    "SCALAR_INFEED",
    "PRODUCER_A",
    "PRODUCER_B",
    "RING_OUTFEED",
    "SCALAR_PIPELINE",
];

/// Field list builder.
#[derive(Default)]
struct F(Vec<(String, u32, u32, bool)>);

impl F {
    /// Predication and opcode, shared by every instruction except scalar bundles.
    /// If `gate` is set the instruction runs iff `p[pred_reg] == pred_pol`.
    fn header() -> F {
        F::default()
            .u("gate", 0, 1)
            .u("pred_reg", 1, 3)
            .u("pred_pol", 4, 1)
            .u("opcode", 6, 6)
    }
    fn u(mut self, name: impl Into<String>, lo: u32, w: u32) -> F {
        self.0.push((name.into(), lo, w, false));
        self
    }
    fn s(mut self, name: impl Into<String>, lo: u32, w: u32) -> F {
        self.0.push((name.into(), lo, w, true));
        self
    }
    fn tile(self) -> F {
        self.u("tile_mask", 12, 16)
    }
}

/// Tensor-op address generator: base address (+ a select bit for wide
/// memory), a 2-bit header mode, then 8 loop levels of
/// (signed increment, count - 1, 2-bit mode).
fn op_ttu(mut f: F, p: &str, lo: u32, base: u32, sel: bool, sw: u32, cw: u32) -> F {
    f = f.u(format!("{p}_base"), lo, base);
    let mut pos = lo + base;
    if sel {
        f = f.u(format!("{p}_sel"), pos, 1);
        pos += 1;
    }
    f = f.u(format!("{p}_hmode"), pos, 2);
    pos += 2;
    for k in 0..8 {
        let r = pos + (sw + cw + 2) * k;
        f = f
            .s(format!("{p}_inc{k}"), r, sw)
            .u(format!("{p}_cnt{k}"), r + sw, cw)
            .u(format!("{p}_mode{k}"), r + sw + cw, 2);
    }
    f
}

/// 17 words, opcodes 0x01 (TensorOp), 0x02 (MAX_POOL_2D), 0x03 (SUM/MEAN
/// reduction). See tinycoral `docs/isa/op.md`.
fn op() -> Layout {
    let mut f = F::header().tile().u("seq", 46, 16);
    for k in 0..8 {
        f = f.u(format!("loop{k}"), 62 + 14 * k, 14);
    }
    f = op_ttu(f, "in", 191, 18, false, 17, 18)
        .u("in_tflags", 507, 8)
        .u("in_twait", 522, 14);
    f = op_ttu(f, "out", 574, 18, false, 17, 18)
        .u("out_tflags", 890, 3)
        .s("out_last_inc", 901, 17)
        .u("out_last_cnt", 918, 18)
        .u("out_last_mode", 936, 2)
        .u("out_last_skip", 941, 16);
    f = op_ttu(f, "par", 957, 13, true, 13, 14)
        .u("par_tflags", 1205, 8)
        .u("par_fifo", 1216, 14);
    f = op_ttu(f, "psum", 1260, 13, true, 13, 14).u("psum_tflags", 1508, 15);
    f = f.u("cfg0", 1563, 9).u("cfg1", 1578, 10);
    for j in 0..5 {
        f = f.u(format!("sync{j}"), 1588 + 16 * j, 16);
    }
    f = f
        .u("cfg2", 1843, 5)
        .u("cfg3", 1850, 4)
        .u("w_zp", 1854, 8)
        // In SUB ops (16.0) `w_zp | w_zp_hi << 8` is a 16-bit operand weight:
        // the numerator of Fraction(s_b / s_a).limit_denominator(32767)
        // (4/4 models). Zero elsewhere except layernorm.
        .u("w_zp_hi", 1862, 8)
        .u("in_zp", 1870, 8)
        // Compiler 16.0 only (never set by 14.1); nonzero in FC/conv/dwconv
        // ops. Values fit a tile's 2048 wide-memory rows of 256 B and fall as
        // kernel area and input width grow, so they look like wide-memory row
        // bounds. Observed: row_c = row_a when row_m = 1, else row_a - row_m
        // (633/634). Names and meaning are provisional.
        .u("row_a", 1886, 11)
        .u("row_m", 1898, 2)
        .u("row_c", 1910, 11)
        .u("dp_mode", 1923, 3)
        .u("reduce_mask", 1926, 8)
        .u("out_ch_last", 1934, 6)
        .u("out_ch", 1940, 6)
        .u("mult_bits", 1949, 32)
        .u("clamp_max_bits", 1981, 32)
        .u("clamp_min_bits", 2013, 32)
        .s("offset", 2045, 32)
        .u("out_zp", 2077, 8)
        .u("cfg4", 2093, 1)
        // 1 exactly when row_a/row_m/row_c are used (1803/1805).
        .u("row_cfg", 2094, 3);
    Layout::new("op", &[0x01, 0x02, 0x03], 17, f.0, &[])
}

/// Tile-local DMA address generator: `levels` x (signed increment, limit).
fn dma_ttu(mut f: F, p: &str, lo: u32, levels: u32) -> F {
    for d in 0..levels {
        f = f.s(format!("{p}_inc{d}"), lo + 33 * d, 17).u(
            format!("{p}_lim{d}"),
            lo + 33 * d + 17,
            16,
        );
    }
    f
}

/// 88-bit sync block of the tile DMAs; only wideToNarrow has the decrement fields.
fn dma_sync(f: F, lo: u32, w2n: bool) -> F {
    let f = f
        .u("sync_en0", lo, 1)
        .u("sync_f1", lo + 1, 1)
        .u("sync_f2", lo + 2, 1)
        .u("sync_id", lo + 4, 5)
        .u("sync_wait_lvl", lo + 9, 2)
        .u("sync_en1", lo + 12, 1)
        .u("sync_val", lo + 13, 19)
        .u("sync_en2", lo + 44, 1)
        .u("sync_f45", lo + 45, 1);
    if w2n {
        f.u("sync_dec_lvl", lo + 51, 2)
            .s("sync_dec", lo + 54, 17)
            .u("sync_dec_mode", lo + 86, 2)
    } else {
        f
    }
}

/// 0x14: wide memory / wide ring FIFO -> narrow memory (or the bias store, mode 2).
fn wide_to_narrow() -> Layout {
    let mut f = F::header().tile().u("seq", 46, 14).u("wide_addr", 60, 14);
    f = dma_ttu(f, "w", 78, 4)
        .u("wide_lvl_mask", 210, 4)
        .u("wide_circ", 214, 1)
        .u("wide_rows", 216, 11)
        .u("narrow_addr", 267, 16);
    f = dma_ttu(f, "n", 283, 6)
        .u("narrow_lvl_mask", 481, 6)
        .u("mode", 487, 2)
        .u("size64", 494, 13)
        .u("head_words_m1", 507, 16)
        .u("head_en", 523, 1)
        // Words in the innermost narrow row after the head: (n_lim0 + 1) -
        // (head_words_m1 + 1), 0 without a head. Verified on 1375/1375
        // instances (corpus sweep, 2026-10-08); tinycoral's `head_tail16` is
        // bits 4-7 of this field.
        .s("tail_words", 526, 16)
        .u("skip_en0", 542, 1)
        .u("skip_en1", 544, 1)
        .u("skip_base", 546, 16)
        .u("skip_m1", 562, 16)
        .u("skip_end", 578, 16);
    f = dma_sync(f, 611, true)
        .u("tail_f867", 867, 1)
        .u("tail_en870", 870, 1)
        .u("tail_f871", 871, 1);
    Layout::new(
        "wideToNarrow",
        &[0x14],
        7,
        f.0,
        &[
            ("n_inc0", 1),
            ("sync_en0", 1),
            ("sync_en1", 1),
            ("sync_en2", 1),
            ("tail_en870", 1),
        ],
    )
}

/// 0x13: narrow memory -> wide memory / wide ring FIFO.
fn narrow_to_wide() -> Layout {
    let mut f = F::header().tile().u("seq", 46, 14).u("narrow_addr", 62, 16);
    f = dma_ttu(f, "n", 78, 6)
        .u("narrow_lvl_mask", 276, 6)
        .u("wide_addr", 335, 14);
    f = dma_ttu(f, "w", 353, 4)
        .u("wide_lvl_mask", 485, 4)
        .u("wide_circ", 489, 1)
        .u("wide_rows", 491, 11);
    f = dma_sync(f, 543, false)
        .u("tail_f799", 799, 1)
        .u("tail_lvl", 801, 2);
    Layout::new(
        "narrowToWide",
        &[0x13],
        7,
        f.0,
        &[
            ("n_inc0", 1),
            ("sync_en0", 1),
            ("sync_en1", 1),
            ("sync_en2", 1),
        ],
    )
}

/// Ring/mesh address generator: 18-bit address, 4 loop levels, buffer block.
fn ring_desc(mut f: F, p: &str, off: u32) -> F {
    f = f.u(format!("{p}addr"), 60 + off, 18);
    for j in 0..4 {
        f = f.s(format!("{p}inc{j}"), 78 + off + 33 * j, 17).u(
            format!("{p}cnt{j}"),
            95 + off + 33 * j,
            16,
        );
    }
    f.u(format!("{p}sdims"), 210 + off, 4)
        .u(format!("{p}cbuf"), 214 + off, 2)
        .u(format!("{p}slots"), 216 + off, 11)
        .u(format!("{p}rsv_s"), 227 + off, 6)
        .u(format!("{p}grp"), 233 + off, 10)
}

/// 41-bit sync record: id, signed value, enable a, x, enable b (+ y if room).
fn ring_rec(f: F, p: &str, lo: u32, idw: u32, valw: u32) -> F {
    let q = lo + idw + valw;
    let f = f
        .u(format!("{p}id"), lo, idw)
        .s(format!("{p}val"), lo + idw, valw)
        .u(format!("{p}en_a"), q, 1)
        .u(format!("{p}x"), q + 1, 15)
        .u(format!("{p}en_b"), q + 16, 1);
    if q + 17 < lo + 41 {
        f.u(format!("{p}y"), q + 17, lo + 41 - (q + 17))
    } else {
        f
    }
}

/// 0x10: narrow/wide memory -> ring, multicast to `dest` (bit 16 = scalar core).
fn ring_producer() -> Layout {
    let mut f = ring_desc(F::header().tile().u("seq", 46, 14), "", 0)
        .u("mode", 335, 2)
        .u("pcfg", 337, 6);
    for (k, lo) in [343, 384, 425].into_iter().enumerate() {
        f = ring_rec(f, &format!("r{k}_"), lo, 7, 16);
    }
    f = f.u("to_c1", 466, 1).u("dest", 467, 17);
    Layout::new("ringProducer", &[0x10], 4, f.0, &[])
}

/// 0x11 / 0x12: ring -> memory (consumer 0 / consumer 1).
fn ring_consumer() -> Layout {
    let f = ring_desc(F::header().tile().u("seq", 46, 14), "", 0)
        .u("gstride", 249, 18)
        .u("aux_en0", 267, 1)
        .u("aux_addr0", 268, 18)
        .u("aux_addr1", 300, 18)
        .u("aux_en1", 318, 1)
        .u("mode", 335, 2)
        .u("s_id", 339, 7)
        .s("s_val", 346, 16)
        .u("s_en_a", 362, 1)
        .u("s_cnt", 368, 10)
        .u("s_en_b", 378, 1)
        .u("s_y", 379, 1);
    Layout::new("ringConsumer", &[0x11, 0x12], 4, f.0, &[])
}

/// 0x15 south, 0x16 west, 0x17 north, 0x18 east: neighbour tile moves with an
/// outbound (`o_`) and an inbound (`i_`) half.
fn mesh() -> Layout {
    let mut f = F::header().tile().u("seq", 46, 14);
    f = ring_desc(f, "o_", 0);
    f = ring_desc(f, "i_", 205)
        .u("out_mode", 473, 2)
        .u("in_mode", 477, 3);
    for k in 0..5 {
        f = ring_rec(f, &format!("s{k}_"), 480 + 41 * k, 6, 18);
    }
    f = f.u("fill_en", 727, 1).u("fill", 728, 32);
    Layout::new("mesh", &[0x15, 0x16, 0x17, 0x18], 6, f.0, &[])
}

/// 0x19: NLU spline coefficient load (8 x 5 f32 coefficients, 7 f32 breakpoints, mode).
fn nlu() -> Layout {
    let mut f = F::header().tile().u("seq", 46, 16);
    for k in 0..50 {
        f = f.u(format!("slot{k}"), 62 + 32 * k, 32);
    }
    Layout::new("nlu", &[0x19], 13, f.0, &[])
}

fn start() -> Layout {
    // 128 * (words between start and end, exclusive)
    Layout::new(
        "start",
        &[0x3e],
        1,
        F::header().u("length_bits", 19, 32).0,
        &[],
    )
}

fn end() -> Layout {
    Layout::new(
        "end",
        &[0x3f],
        1,
        F::header().u("length_bits", 19, 32).0,
        &[],
    )
}

fn halt() -> Layout {
    // code 4; 1 ends a bitstream that continues in the next one
    Layout::new("halt", &[0x21], 1, F::header().u("code", 14, 5).0, &[])
}

/// 0x1a: tile fence, dispatched to the tiles with a sequence number.
fn sync() -> Layout {
    let f = F::header()
        .u("tiles", 12, 16)
        .u("seq", 28, 15)
        .u("b43", 43, 1)
        .u("counters", 44, 19)
        .u("count", 63, 16)
        .u("units", 79, 10)
        .u("signal", 107, 1)
        .u("f109", 109, 2);
    Layout::new("sync", &[0x1a], 1, f.0, &[])
}

/// 0x24: scalar fence / scalar sync-counter op.
fn scsync() -> Layout {
    let mut f = F::header()
        .u("tiles", 12, 16)
        .u("smask", 32, 9)
        .u("value", 43, 13)
        .u("tokwait", 57, 1)
        .u("tmask", 73, 16);
    for k in 0..9 {
        f = f.u(format!("thr{k}"), 93 + 16 * k, 16);
    }
    f = f.u("wait", 237, 9).u("sunits", 246, 5);
    Layout::new("scsync", &[0x24], 2, f.0, &[])
}

/// 0x25: avDataPop (stream=1) / parameterPop (stream=0): host -> staging buffer.
fn pop() -> Layout {
    let f = F::header()
        .u("stream", 12, 1)
        // Staging-buffer offset this pop writes to; the next parameter infeed
        // reads from the same `buf_off` (39/39, SSDLite streamed weights).
        // Zero for activation pops and non-streaming programs.
        .u("buf_off", 13, 11)
        .u("d0_stride", 28, 16)
        .u("d0_limit", 44, 15)
        .u("d1_stride", 59, 16)
        .u("d1_limit", 75, 15)
        .u("d2_stride", 90, 16)
        .u("d2_limit", 106, 15)
        .u("k121", 121, 1)
        .u("k124", 124, 1)
        .u("par136", 136, 1)
        .u("av140", 140, 1)
        .u("count_m1", 142, 14)
        .u("k157", 157, 1)
        .u("count_neg", 159, 14)
        .u("k174", 174, 1)
        .u("par177", 177, 1)
        .u("k178", 178, 1)
        .u("b183", 183, 1)
        .u("rows_neg", 184, 15)
        .u("k199", 199, 1)
        .u("k215", 215, 1);
    Layout::new("pop", &[0x25], 2, f.0, &[])
}

/// 0x26: staging buffer -> ring bus -> the tiles in `tiles`.
fn infeed() -> Layout {
    let f = F::header()
        .u("stream", 12, 1)
        // On-chip loopback: this infeed reads rows written by a later outfeed
        // with `loop_en` (same d0/d1/d2 limits, 92/92 paired) and sends them to
        // the tiles. `loop_k168`, `loop_k308`, `loop_k315` are constant (1, 3, 3) with it.
        .u("loop_en", 13, 1)
        .u("buf_off", 14, 20)
        .u("d0_stride", 34, 21)
        .u("d0_limit", 55, 20)
        .u("d1_stride", 75, 21)
        .u("d1_limit", 96, 20)
        .u("d2_stride", 116, 21)
        .u("d2_limit", 137, 20)
        .u("k157", 157, 1)
        .u("k160", 160, 1)
        .u("loop_k168", 168, 1)
        .u("par172", 172, 1)
        .u("av176", 176, 1)
        .u("count_m1", 183, 14)
        .u("k203", 203, 1)
        .u("count_neg", 205, 14)
        .u("loop_k308", 308, 2)
        .u("k311", 311, 1)
        .u("par314", 314, 1)
        .u("loop_k315", 315, 2)
        .u("pop_wait", 320, 16)
        .u("k336", 336, 1)
        .u("k352", 352, 1)
        .u("k392", 392, 1)
        .u("tiles", 409, 16)
        .u("k434", 434, 4)
        .u("f438", 438, 2);
    Layout::new("infeed", &[0x26], 4, f.0, &[])
}

/// 0x27: ring -> host stream (DMA tag 3), or ring -> scalar memory.
fn outfeed() -> Layout {
    let f = F::header()
        .u("base", 12, 15)
        .u("d0_stride", 27, 16)
        .u("d0_limit", 43, 15)
        .u("d1_stride", 58, 16)
        .u("d1_limit", 74, 15)
        .u("d2_stride", 89, 16)
        .u("d2_limit", 105, 15)
        .u("d3_stride", 120, 16)
        .u("k173", 173, 2)
        // Loopback outfeed: writes to a ring of rows 1024..2047 (first use
        // 2046) instead of the host; each one advances `loop_row` by its row
        // count (d1_limit + 1), wrapping to 1024 (64/66). `loop_cfg` is 17 in
        // 90/92 instances.
        .u("loop_cfg", 178, 10)
        .u("loop_row", 188, 11)
        .u("loop_en", 214, 1)
        .u("f222", 222, 7);
    Layout::new("outfeed", &[0x27], 2, f.0, &[])
}

/// 0x20: scalar-core bundle (geohot's edgetpuxray field view). Bits 6-10
/// (`branch`) and 11 (`enable_scalar`) form the opcode.
fn scalar() -> Layout {
    let f = F::default()
        .u("gate", 0, 1)
        .u("pred_reg", 1, 3)
        .u("yes_pred", 4, 1)
        .u("unk_0", 5, 1)
        .u("branch", 6, 5)
        .u("enable_scalar", 11, 1)
        .u("enable_vector", 12, 2)
        .u("vs_reg_v1", 14, 5)
        .u("imm_size", 19, 12)
        .u("v_op", 31, 5)
        .u("v_offset", 36, 8)
        .u("v_cmd", 44, 5)
        .u("vs_reg", 49, 5)
        .u("s_op", 54, 6)
        .u("s_x", 60, 5)
        .u("s_y", 65, 5)
        .u("imm_scalar", 70, 32)
        .u("v_op_2", 102, 3)
        .u("vs_reg_w", 105, 5)
        .u("unk_3", 110, 18);
    Layout::new("scalar", &[0x20], 1, f.0, &[("enable_scalar", 1)])
}

/// 0x23: repeat the next `body` words `count` times (from SOFTMAX programs).
fn loop_() -> Layout {
    Layout::new(
        "loop",
        &[0x23],
        1,
        F::header().u("count", 12, 16).u("body", 28, 10).0,
        &[],
    )
}

/// 0x22: relative branch, taken while p0 is set. Only one form has been seen
/// (`cond` = 0x11, `flag` = 1), so the field split is a hypothesis.
fn branch() -> Layout {
    let f = F::header()
        .u("cond", 12, 5)
        .s("off", 17, 15)
        .u("flag", 33, 1);
    Layout::new("branch", &[0x22], 1, f.0, &[])
}

pub(super) fn all() -> Vec<Layout> {
    vec![
        op(),
        wide_to_narrow(),
        narrow_to_wide(),
        ring_producer(),
        ring_consumer(),
        mesh(),
        nlu(),
        start(),
        end(),
        halt(),
        sync(),
        scsync(),
        pop(),
        infeed(),
        outfeed(),
        scalar(),
        loop_(),
        branch(),
    ]
}
