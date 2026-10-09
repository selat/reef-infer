//! Hardware perturbation experiments: start from a generated program that is
//! known to be correct, change one thing, and classify what the TPU does.
//!
//!     cargo run --features usb --example isa_probe [-- filter]

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::elementwise::{self, Requant};
use reef_infer::codegen::fc::Fc;
use reef_infer::isa;
use reef_infer::model::{ChunkInfo, TfliteModel};
use reef_infer::runner::Device;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Sets `field` of the `nth` instruction whose layout is `layout`.
fn patch(chunk: &mut ChunkInfo, layout: &str, nth: usize, field: &str, value: i64) {
    let mut word = 0;
    let mut seen = 0;
    for ins in isa::split(&chunk.bitstream).unwrap() {
        let mut d = ins.decode().unwrap();
        if d.layout.name == layout {
            if seen == nth {
                d.set(field, value).unwrap();
                for (j, w) in d.encode().unwrap().iter().enumerate() {
                    chunk.bitstream[(word + j) * 16..(word + j + 1) * 16]
                        .copy_from_slice(&w.to_le_bytes());
                }
                return;
            }
            seen += 1;
        }
        word += ins.words.len();
    }
    panic!("no {layout} #{nth}");
}

fn base_fc(rng: &mut Rng) -> Fc {
    let (k, n) = (64, 64);
    Fc {
        k,
        n,
        weights: (0..n * k).map(|_| rng.next() as i8).collect(),
        w_scale: (0..n).map(|i| 0.004 + 0.0001 * i as f32).collect(),
        bias: (0..n).map(|i| 300 * i as i32 - 9000).collect(),
        input: Quant {
            scale: 0.03,
            zero_point: 128,
        },
        output: Quant {
            scale: 0.25,
            zero_point: 124,
        },
        relu: false,
    }
}

enum Outcome {
    Correct,
    Wrong(usize, usize, Vec<(u8, u8)>),
    Hang,
}

struct Probe {
    slot: Option<Device>,
    rng: Rng,
}

impl Probe {
    async fn reopen(&mut self) {
        self.slot = None;
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            if let Ok(d) = Device::open().await {
                self.slot = Some(d);
                return;
            }
        }
        panic!("device did not come back");
    }

    /// Runs `model` on `reps` random inputs and compares with `want(x)`.
    async fn run(
        &mut self,
        model: TfliteModel,
        in_len: usize,
        reps: usize,
        want: impl Fn(&[u8]) -> Vec<u8>,
    ) -> Outcome {
        let model = Arc::new(model);
        let (mut bad, mut total, mut examples) = (0, 0, Vec::new());
        for _ in 0..reps {
            let x = self.rng.bytes(in_len);
            let dev = self.slot.as_mut().unwrap();
            let r = match dev.load_params(Arc::clone(&model)).await {
                Ok(l) => dev.run_inference(&l, x.clone()).await,
                Err(e) => Err(e),
            };
            match r {
                Ok(got) => {
                    let w = want(&x);
                    total += w.len();
                    for (&g, &e) in got.iter().zip(&w) {
                        if g != e {
                            bad += 1;
                            if examples.len() < 4 {
                                examples.push((g, e));
                            }
                        }
                    }
                }
                Err(_) => {
                    self.reopen().await;
                    return Outcome::Hang;
                }
            }
        }
        if bad == 0 {
            Outcome::Correct
        } else {
            Outcome::Wrong(bad, total, examples)
        }
    }
}

fn report(name: &str, o: Outcome) {
    match o {
        Outcome::Correct => println!("{name:58} correct"),
        Outcome::Hang => println!("{name:58} HANG"),
        Outcome::Wrong(b, t, ex) => println!("{name:58} wrong {b}/{t} (got, want) {ex:?}"),
    }
}

type FcMut = Box<dyn Fn(&mut TfliteModel)>;

#[tokio::main]
async fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let mut p = Probe {
        slot: Some(Device::open().await.expect("open device")),
        rng: Rng(0x1234_5678_9abc_def1),
    };

    // ── FC TensorOp perturbations (EXECUTION_ONLY = executables[0], the op is
    // the only "op" instruction).
    let fc_cases: Vec<(&str, FcMut)> = vec![
        ("fc baseline", Box::new(|_| {})),
        (
            "fc row_a/row_m/row_c/row_cfg = 0 (as 14.1)",
            Box::new(|m| {
                let c = &mut m.executables[0].instruction_chunks[0];
                for f in ["row_a", "row_m", "row_c", "row_cfg"] {
                    patch(c, "op", 0, f, 0);
                }
            }),
        ),
        (
            "fc row_a = row_c = 2047",
            Box::new(|m| {
                let c = &mut m.executables[0].instruction_chunks[0];
                patch(c, "op", 0, "row_a", 2047);
                patch(c, "op", 0, "row_c", 2047);
            }),
        ),
        (
            "fc row_a = row_c = 1",
            Box::new(|m| {
                let c = &mut m.executables[0].instruction_chunks[0];
                patch(c, "op", 0, "row_a", 1);
                patch(c, "op", 0, "row_c", 1);
            }),
        ),
        (
            "fc row_cfg = 0 only",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "op",
                    0,
                    "row_cfg",
                    0,
                )
            }),
        ),
        (
            "fc cfg4 = 0",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "op",
                    0,
                    "cfg4",
                    0,
                )
            }),
        ),
        (
            "fc 14.1 form: cfg0=0xa5 dp_mode=3 reduce_mask=0b101, rows 0",
            Box::new(|m| {
                let c = &mut m.executables[0].instruction_chunks[0];
                for (f, v) in [
                    ("cfg0", 0xa5),
                    ("dp_mode", 3),
                    ("reduce_mask", 5),
                    ("row_a", 0),
                    ("row_m", 0),
                    ("row_c", 0),
                    ("row_cfg", 0),
                ] {
                    patch(c, "op", 0, f, v);
                }
            }),
        ),
        (
            "fc mult_bits = 2.0 (per-channel mult x2?)",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "op",
                    0,
                    "mult_bits",
                    2.0f32.to_bits() as i64,
                )
            }),
        ),
        (
            "fc caching ringConsumer rsv286 = rsv319 = 0",
            Box::new(|m| {
                let c = &mut m.executables[1].instruction_chunks[0];
                patch(c, "ringConsumer", 0, "rsv286", 0);
                patch(c, "ringConsumer", 0, "rsv319", 0);
            }),
        ),
        (
            "fc output narrowToWide tail_lvl = 0",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "narrowToWide",
                    1,
                    "tail_lvl",
                    0,
                )
            }),
        ),
        (
            "fc loopback narrowToWide tail_lvl = 0",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "narrowToWide",
                    0,
                    "tail_lvl",
                    0,
                )
            }),
        ),
        (
            "fc input wideToNarrow tail_en870 = 0",
            Box::new(|m| {
                patch(
                    &mut m.executables[0].instruction_chunks[0],
                    "wideToNarrow",
                    0,
                    "tail_en870",
                    0,
                )
            }),
        ),
    ];
    for (name, mutate) in fc_cases {
        if !name.contains(&filter) {
            continue;
        }
        let fc = base_fc(&mut p.rng);
        let mut m = fc.build().unwrap();
        mutate(&mut m);
        let o = p.run(m, fc.k, 4, |x| fc.reference(x)).await;
        report(name, o);
    }

    // ── Parameter block 3 (i32 128 per channel): weight zero point? Store
    // weights as W + c with block 3 = c. If block 3 is the weight zero point,
    // the result is unchanged for any c.
    for c in [128i32, 100, 0, 200] {
        let name = format!("fc weights stored as W+{c}, block 3 = {c}");
        if !name.contains(&filter) {
            continue;
        }
        let fc = base_fc(&mut p.rng);
        let fc = Fc {
            weights: fc.weights.iter().map(|&w| w.clamp(-100, 55)).collect(),
            ..fc
        };
        let mut m = fc.build().unwrap();
        let (n, k) = (fc.n, fc.k);
        let params = &mut m.executables[1].parameters;
        for i in 0..n {
            params[8 * n + 4 * i..8 * n + 4 * i + 4].copy_from_slice(&c.to_le_bytes());
        }
        for k4 in 0..k / 4 {
            for o in 0..n {
                for j in 0..4 {
                    params[12 * n + (k4 * n + o) * 4 + j] =
                        (fc.weights[o * k + 4 * k4 + j] as i32 + c) as u8;
                }
            }
        }
        let o = p.run(m, k, 4, |x| fc.reference(x)).await;
        report(&name, o);
    }

    // ── row_a: wide row of the weight zero-point vector? Zero the vector at
    // row 2, append it as row 19 (after 4096 B of weights) and point row_a /
    // row_c there.
    for (name, row) in [
        ("fc zp vector moved to row 19, row_a = row_c = 19", 19),
        ("fc zp vector moved to row 19, row_a left at 2", 2),
    ] {
        if !name.contains(&filter) {
            continue;
        }
        let fc = base_fc(&mut p.rng);
        let fc = Fc {
            weights: fc.weights.iter().map(|&w| w.clamp(-100, 55)).collect(),
            ..fc
        };
        let (n, k, c) = (fc.n, fc.k, 100i32);
        let mut params = fc.parameters();
        for i in 0..n {
            params[8 * n + 4 * i..8 * n + 4 * i + 4].copy_from_slice(&0i32.to_le_bytes());
        }
        for k4 in 0..k / 4 {
            for o in 0..n {
                for j in 0..4 {
                    params[12 * n + (k4 * n + o) * 4 + j] =
                        (fc.weights[o * k + 4 * k4 + j] as i32 + c) as u8;
                }
            }
        }
        assert_eq!(params.len(), 19 * 256);
        for _ in 0..n {
            params.extend(c.to_le_bytes());
        }
        let mut m = fc.build_with(params).unwrap();
        let ch = &mut m.executables[0].instruction_chunks[0];
        patch(ch, "op", 0, "row_a", row);
        patch(ch, "op", 0, "row_c", row);
        let o = p.run(m, k, 4, |x| fc.reference(x)).await;
        report(name, o);
    }

    // Size-preserving version: bias = 100 for every channel, so row 1 (bias)
    // holds the same i32 values a zero-point row with c = 100 would. Zero
    // row 2 and point row_a / row_c at row 1.
    for (name, row) in [
        ("fc zp row 2 zeroed, bias row = zps, row_a = row_c = 1", 1),
        ("fc zp row 2 zeroed, bias row = zps, row_a = row_c = 2", 2),
    ] {
        if !name.contains(&filter) {
            continue;
        }
        let fc = base_fc(&mut p.rng);
        let fc = Fc {
            weights: fc.weights.iter().map(|&w| w.clamp(-100, 55)).collect(),
            bias: vec![100; 64],
            ..fc
        };
        let (n, k, c) = (fc.n, fc.k, 100i32);
        let mut params = fc.parameters();
        for i in 0..n {
            params[8 * n + 4 * i..8 * n + 4 * i + 4].copy_from_slice(&0i32.to_le_bytes());
        }
        for k4 in 0..k / 4 {
            for o in 0..n {
                for j in 0..4 {
                    params[12 * n + (k4 * n + o) * 4 + j] =
                        (fc.weights[o * k + 4 * k4 + j] as i32 + c) as u8;
                }
            }
        }
        let mut m = fc.build_with(params).unwrap();
        let ch = &mut m.executables[0].instruction_chunks[0];
        patch(ch, "op", 0, "row_a", row);
        patch(ch, "op", 0, "row_c", row);
        let o = p.run(m, k, 4, |x| fc.reference(x)).await;
        report(name, o);
    }

    // ── cfg4 = 0 compared with a zero-bias reference.
    let name = "fc cfg4 = 0 vs reference with bias = 0";
    if name.contains(&filter) {
        let fc = base_fc(&mut p.rng);
        let mut m = fc.build().unwrap();
        patch(
            &mut m.executables[0].instruction_chunks[0],
            "op",
            0,
            "cfg4",
            0,
        );
        let nobias = Fc {
            bias: vec![0; fc.n],
            ..fc.clone()
        };
        let o = p.run(m, fc.k, 4, |x| nobias.reference(x)).await;
        report(name, o);
    }

    // ── Per-channel weight zero points: channel o stored as W + 60 + o.
    let name = "fc per-channel weight zp (60 + channel)";
    if name.contains(&filter) {
        let fc = base_fc(&mut p.rng);
        let fc = Fc {
            weights: fc.weights.iter().map(|&w| w.clamp(-60, 60)).collect(),
            ..fc
        };
        let (n, k) = (fc.n, fc.k);
        let mut params = fc.parameters();
        for o in 0..n {
            params[8 * n + 4 * o..8 * n + 4 * o + 4]
                .copy_from_slice(&(60 + o as i32).to_le_bytes());
        }
        for k4 in 0..k / 4 {
            for o in 0..n {
                for j in 0..4 {
                    params[12 * n + (k4 * n + o) * 4 + j] =
                        (fc.weights[o * k + 4 * k4 + j] as i32 + 60 + o as i32) as u8;
                }
            }
        }
        let o = p
            .run(fc.build_with(params).unwrap(), k, 4, |x| fc.reference(x))
            .await;
        report(name, o);
    }

    // ── cfg4 = 0 added 4 with out_zp 124. Same with other output zero points.
    for zp in [0, 200, 124] {
        let name = format!("fc cfg4 = 0, out_zp {zp}: output - reference");
        if !name.contains(&filter) {
            continue;
        }
        let fc = base_fc(&mut p.rng);
        let fc = Fc {
            output: Quant {
                scale: 0.5,
                zero_point: zp,
            },
            ..fc
        };
        let mut m = fc.build().unwrap();
        patch(
            &mut m.executables[0].instruction_chunks[0],
            "op",
            0,
            "cfg4",
            0,
        );
        let x = p.rng.bytes(fc.k);
        let dev = p.slot.as_mut().unwrap();
        let l = dev.load_params(Arc::new(m)).await.unwrap();
        let got = dev.run_inference(&l, x.clone()).await.unwrap();
        let mut d: Vec<i32> = got
            .iter()
            .zip(fc.reference(&x))
            .map(|(&g, w)| g as i32 - w as i32)
            .collect();
        d.sort();
        d.dedup();
        println!("{name:58} distinct diffs {d:?}");
    }

    // ── Requant stage order: clamp then round, or round then clamp?
    // With clamp_max = 40.7: clamp-then-round gives 41, round-then-clamp 40.7 -> 40.
    let q = Quant {
        scale: 1.0,
        zero_point: 0,
    };
    for (name, rq) in [
        (
            "requant clamp_max 40.7 (x1, zp 0)",
            Requant {
                in_zp: 0,
                mult: 1.0,
                clamp_min: 0.0,
                clamp_max: 40.7,
                out_zp: 0,
            },
        ),
        (
            "requant clamp_min 10.3 (x1, zp 0)",
            Requant {
                in_zp: 0,
                mult: 1.0,
                clamp_min: 10.3,
                clamp_max: 255.0,
                out_zp: 0,
            },
        ),
        (
            "requant clamp_max 300, out_zp 100 (saturation?)",
            Requant {
                in_zp: 0,
                mult: 1.0,
                clamp_min: 0.0,
                clamp_max: 300.0,
                out_zp: 100,
            },
        ),
        (
            "requant clamp_min -50, x-1 (negative result?)",
            Requant {
                in_zp: 0,
                mult: -1.0,
                clamp_min: -50.0,
                clamp_max: 255.0,
                out_zp: 0,
            },
        ),
    ] {
        if !name.contains(&filter) {
            continue;
        }
        let prog = elementwise::build(&rq, q, q, true).unwrap();
        let x: Vec<u8> = (0..elementwise::SIZE).map(|i| i as u8).collect();
        let dev = p.slot.as_mut().unwrap();
        let l = dev.load_params(Arc::new(prog.into_model())).await.unwrap();
        match dev.run_inference(&l, x.clone()).await {
            Ok(got) => {
                // value per input byte 0..=255
                let mut seen = vec![None; 256];
                for (&xi, &g) in x.iter().zip(&got) {
                    seen[xi as usize].get_or_insert(g);
                }
                let show: Vec<String> = [
                    0usize, 1, 9, 10, 11, 39, 40, 41, 42, 100, 155, 156, 200, 255,
                ]
                .iter()
                .map(|&i| format!("{i}->{}", seen[i].unwrap()))
                .collect();
                println!("{name:58} {}", show.join(" "));
            }
            Err(e) => {
                println!("{name:58} {e}");
                p.reopen().await;
            }
        }
    }

    // ── wideToNarrow bit 543 (set with skip_en0 by the compiler).
    let name = "relu wideToNarrow rsv543 = 0";
    if name.contains(&filter) {
        let (qi, qo) = (
            Quant {
                scale: 0.0627,
                zero_point: 128,
            },
            Quant {
                scale: 0.0314,
                zero_point: 0,
            },
        );
        let rq = Requant::from_quant(qi, qo, true);
        let prog = elementwise::build(&rq, qi, qo, false).unwrap();
        let o = p
            .run(prog.into_model(), elementwise::SIZE, 1, |x| {
                x.iter().map(|&v| rq.apply(v)).collect()
            })
            .await;
        report(name, o);
    }
}
