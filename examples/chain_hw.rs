//! On-chip chaining experiments: several layer programs in one call.
//!
//!     cargo run --release --example chain_hw
//!
//! 1. Two complete conv programs spliced into one bitstream
//!    (`codegen::chain::splice`), each with its own host input and output.
//! 2. conv -> conv with the activation kept on chip: A's conv op writes
//!    straight into B's padded patch (`conv::Io`), B reads it from there;
//!    only A's input comes from the host and only B's output goes back.
//! 3. Three convs 16 -> 32 -> 32 -> 16 (ReLU in the middle) on chip, the
//!    patches alternating between two narrow buffers, timed against three
//!    separate calls.
//! 4. `codegen::chain::convs` at 32x32 and across stride 2 (32x32 ->
//!    16x16), 3x3 and 1x1: ResNet-8's first two stages' conv shapes.

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::chain::{rename, splice};
use reef_infer::codegen::conv::{Conv, Io};
use reef_infer::executable_generated::platforms::darwinn::ExecutableType;
use reef_infer::model::{ExecInfo, TfliteModel};
use reef_infer::runner::Device;

fn rng(seed: u64) -> impl FnMut() -> u8 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as u8
    }
}

fn conv(cin: usize, cout: usize, seed: u64, in_zp: i32, out_zp: i32) -> Conv {
    let mut next = rng(seed);
    Conv {
        cin,
        cout,
        k: 3,
        stride: 1,
        width: 16,
        height: 16,
        weights: (0..cout * 9 * cin).map(|_| next() as i8).collect(),
        w_scale: (0..cout)
            .map(|i| (0.0007 + 0.0001 * i as f32) * 16.0 / cin as f32)
            .collect(),
        bias: (0..cout).map(|i| 300 * i as i32 - 4000).collect(),
        input: Quant {
            scale: 0.03,
            zero_point: in_zp,
        },
        output: Quant {
            scale: 0.2,
            zero_point: out_zp,
        },
        relu: false,
        pad_before: None,
    }
}

/// A conv of any supported shape (same quantization rules as `conv`).
#[allow(clippy::too_many_arguments)]
fn conv_at(
    k: usize,
    stride: usize,
    hw: usize,
    cin: usize,
    cout: usize,
    seed: u64,
    zp: (i32, i32),
    relu: bool,
) -> Conv {
    let mut next = rng(seed);
    Conv {
        k,
        stride,
        width: hw,
        height: hw,
        weights: (0..cout * k * k * cin).map(|_| next() as i8).collect(),
        relu,
        ..conv(cin, cout, seed, zp.0, zp.1)
    }
}

/// Loads `layers` at consecutive wide bases, runs them as one chained
/// call on 10 inputs against the host references, and times it against
/// one call per layer.
async fn chain_case(dev: &mut Device, title: &str, layers: &[Conv]) {
    let mut bases = Vec::new();
    let mut next = 0;
    let mut separate = Vec::new();
    for c in layers {
        bases.push(next);
        separate.push(
            dev.load_params(Arc::new(c.build_at(next).unwrap()))
                .await
                .unwrap(),
        );
        next += c.wide_units();
    }
    let pairs: Vec<(&Conv, i64)> = layers.iter().zip(bases.iter().copied()).collect();
    let chunk = match reef_infer::codegen::chain::convs(&pairs) {
        Ok(c) => c,
        Err(e) => {
            println!("{title}: {e}");
            return;
        }
    };
    let first = exec(layers[0].build().unwrap(), "x");
    let last = exec(layers[layers.len() - 1].build().unwrap(), "y");
    let mut model = TfliteModel {
        executables: vec![ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![chunk],
            input_layers: first.input_layers,
            output_layers: last.output_layers,
            scratch_size_bytes: 0,
        }],
    };
    let e = &mut model.executables[0];
    e.input_layers[0].name = "input".into();
    e.output_layers[0].name = "output".into();
    let chained = dev.load_params(Arc::new(model)).await.unwrap();
    let reference = |x: &[u8]| layers.iter().fold(x.to_vec(), |y, c| c.reference(&y));
    let n_in = layers[0].width * layers[0].height * layers[0].cin;
    let mut bad = 0;
    for seed in 40..50 {
        let x = input(n_in, seed);
        match dev.run_inference(&chained, x.clone()).await {
            Ok(y) => bad += (y != reference(&x)) as usize,
            Err(e) => {
                println!("{title}: {e}");
                return;
            }
        }
    }
    let x = input(n_in, 99);
    let want = reference(&x);
    let reps = 100;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let mut y = x.clone();
        for l in &separate {
            y = dev.run_inference(l, y).await.unwrap();
        }
        assert_eq!(y, want);
    }
    let sep = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        dev.run_inference(&chained, x.clone()).await.unwrap();
    }
    let one = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "{title}: {}; {} separate calls {sep:.3} ms, one chained call {one:.3} ms",
        if bad == 0 {
            "10/10 exact".to_string()
        } else {
            format!("{bad}/10 WRONG")
        },
        layers.len()
    );
}

fn input(n: usize, seed: u64) -> Vec<u8> {
    let mut next = rng(seed);
    (0..n).map(|_| next()).collect()
}

/// The EXECUTION_ONLY executable of a built layer, its layers renamed
/// `{p}_in` / `{p}_out`.
fn exec(m: TfliteModel, p: &str) -> ExecInfo {
    let mut e = m
        .executables
        .into_iter()
        .find(|e| e.exec_type == ExecutableType::EXECUTION_ONLY)
        .unwrap();
    for c in &mut e.instruction_chunks {
        rename(c, "input", &format!("{p}_in"));
        rename(c, "output", &format!("{p}_out"));
    }
    e.input_layers[0].name = format!("{p}_in");
    e.output_layers[0].name = format!("{p}_out");
    e
}

#[tokio::main]
async fn main() {
    let mut dev = Device::open().await.expect("no Edge TPU");
    let a = conv(16, 16, 1, 100, 120);
    let b = conv(16, 16, 2, 120, 130);
    let ua = a.wide_units();
    let la = dev
        .load_params(Arc::new(a.build_at(0).unwrap()))
        .await
        .unwrap();
    let lb = dev
        .load_params(Arc::new(b.build_at(ua).unwrap()))
        .await
        .unwrap();
    let (xa, xb) = (input(4096, 3), input(4096, 4));
    let ok_a = dev.run_inference(&la, xa.clone()).await.unwrap() == a.reference(&xa);
    let ok_b = dev.run_inference(&lb, xb.clone()).await.unwrap() == b.reference(&xb);
    println!("separate calls: A {ok_a}, B {ok_b}");

    // ── 1. A and B spliced, each with its own host input and output.
    let ea = exec(a.build_at(0).unwrap(), "a");
    let eb = exec(b.build_at(ua).unwrap(), "b");
    let chunk = splice(&[
        ea.instruction_chunks.into_iter().next().unwrap(),
        eb.instruction_chunks.into_iter().next().unwrap(),
    ])
    .unwrap();
    let both = TfliteModel {
        executables: vec![ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![chunk],
            input_layers: ea.input_layers.into_iter().chain(eb.input_layers).collect(),
            output_layers: ea
                .output_layers
                .into_iter()
                .chain(eb.output_layers)
                .collect(),
            scratch_size_bytes: 0,
        }],
    };
    let l = dev.load_params(Arc::new(both)).await.unwrap();
    match dev
        .run_inference(&l, [xa.clone(), xb.clone()].concat())
        .await
    {
        Ok(y) => {
            let (ya, yb) = y.split_at(4096);
            println!(
                "1. spliced A+B, one call: A {}, B {}",
                ya == a.reference(&xa),
                yb == b.reference(&xb)
            );
        }
        Err(e) => println!("1. spliced A+B: {e}"),
    }

    // ── 2. A -> B on chip: B's patch at narrow NB on every tile.
    const NB: i64 = 0x1000;
    let mut pa = a
        .chain_program(
            0,
            Io {
                output_to: Some((NB, b.input_patch())),
                ..Io::default()
            },
        )
        .unwrap();
    rename(&mut pa, "input", "a_in");
    let mut pb = b
        .chain_program(
            ua,
            Io {
                input_at: Some((NB, b.input_patch())),
                ..Io::default()
            },
        )
        .unwrap();
    rename(&mut pb, "output", "b_out");
    let ea = exec(a.build_at(0).unwrap(), "a");
    let eb = exec(b.build_at(ua).unwrap(), "b");
    let chained = TfliteModel {
        executables: vec![ExecInfo {
            exec_type: ExecutableType::EXECUTION_ONLY,
            parameters: vec![],
            instruction_chunks: vec![splice(&[pa, pb]).unwrap()],
            input_layers: ea.input_layers,
            output_layers: eb.output_layers,
            scratch_size_bytes: 0,
        }],
    };
    let l = dev.load_params(Arc::new(chained)).await.unwrap();
    for seed in [5, 6, 7] {
        let x = input(4096, seed);
        let want = b.reference(&a.reference(&x));
        match dev.run_inference(&l, x).await {
            Ok(y) => {
                let bad = y.iter().zip(&want).filter(|(u, v)| u != v).count();
                println!(
                    "2. A -> B on chip, one call: {bad} of {} bytes differ",
                    want.len()
                );
            }
            Err(e) => {
                println!("2. A -> B on chip: {e}");
                break;
            }
        }
    }

    // ── 3. Three layers, two patch buffers in turn.
    let c1 = conv(16, 32, 11, 110, 125);
    let mut c2 = conv(32, 32, 12, 125, 0);
    c2.relu = true;
    let c3 = conv(32, 16, 13, 0, 128);
    let (u1, u2) = (c1.wide_units(), c2.wide_units());
    let bases = [0, u1, u1 + u2];
    let layers = [&c1, &c2, &c3];
    let mut separate = Vec::new();
    for (c, &base) in layers.iter().zip(&bases) {
        separate.push(
            dev.load_params(Arc::new(c.build_at(base).unwrap()))
                .await
                .unwrap(),
        );
    }
    const BUF: [i64; 2] = [0x1000, 0x2000];
    let io = |i: usize| Io {
        input_at: (i > 0).then(|| (BUF[(i + 1) % 2], layers[i].input_patch())),
        output_to: (i < 2).then(|| (BUF[i % 2], layers[i + 1].input_patch())),
        output_ch: None,
    };
    let mut parts = Vec::new();
    for (i, (c, &base)) in layers.iter().zip(&bases).enumerate() {
        parts.push(c.chain_program(base, io(i)).unwrap());
    }
    let e1 = exec(c1.build_at(0).unwrap(), "input");
    let e3 = exec(c3.build_at(bases[2]).unwrap(), "output");
    let mut chunk = splice(&parts).unwrap();
    rename(&mut chunk, "input", "input_in");
    rename(&mut chunk, "output", "output_out");
    let three = dev
        .load_params(Arc::new(TfliteModel {
            executables: vec![ExecInfo {
                exec_type: ExecutableType::EXECUTION_ONLY,
                parameters: vec![],
                instruction_chunks: vec![chunk],
                input_layers: e1.input_layers,
                output_layers: e3.output_layers,
                scratch_size_bytes: 0,
            }],
        }))
        .await
        .unwrap();
    let reference = |x: &[u8]| c3.reference(&c2.reference(&c1.reference(x)));
    let mut all = true;
    for seed in 20..30 {
        let x = input(4096, seed);
        all &= dev.run_inference(&three, x.clone()).await.unwrap() == reference(&x);
    }
    println!(
        "3. three convs on chip, 10 inputs: {}",
        if all { "all exact" } else { "WRONG" }
    );
    let x = input(4096, 99);
    let want = reference(&x);
    let reps = 200;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let mut y = x.clone();
        for l in &separate {
            y = dev.run_inference(l, y).await.unwrap();
        }
        assert_eq!(y, want);
    }
    let sep = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        dev.run_inference(&three, x.clone()).await.unwrap();
    }
    let one = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("   three separate calls {sep:.3} ms, one chained call {one:.3} ms");

    // ── 4. 32x32 and stride 2.
    chain_case(
        &mut dev,
        "4a. 32x32 3x3 16->16 -> 16->16",
        &[
            conv_at(3, 1, 32, 16, 16, 31, (100, 0), true),
            conv_at(3, 1, 32, 16, 16, 32, (0, 120), false),
        ],
    )
    .await;
    chain_case(
        &mut dev,
        "4b. 32x32 16->32 stride 2 -> 16x16 32->32",
        &[
            conv_at(3, 2, 32, 16, 32, 33, (128, 0), true),
            conv_at(3, 1, 16, 32, 32, 34, (0, 125), false),
        ],
    )
    .await;
    chain_case(
        &mut dev,
        "4c. 32x32 1x1 16->32 stride 2 -> 16x16 32->32",
        &[
            conv_at(1, 2, 32, 16, 32, 35, (128, 110), false),
            conv_at(3, 1, 16, 32, 32, 36, (110, 125), false),
        ],
    )
    .await;
    chain_case(
        &mut dev,
        "4d. ResNet-8 stages 1-2 convs: 32x32 x3 -> stride 2 -> 16x16",
        &[
            conv_at(3, 1, 32, 16, 16, 37, (128, 0), true),
            conv_at(3, 1, 32, 16, 16, 38, (0, 0), true),
            conv_at(3, 1, 32, 16, 16, 39, (0, 0), true),
            conv_at(3, 2, 32, 16, 32, 40, (0, 0), true),
            conv_at(3, 1, 16, 32, 32, 41, (0, 125), false),
        ],
    )
    .await;
    chain_case(
        &mut dev,
        "4e. 32x32 16->32 -> 32->32 -> stride 2 32->32 -> 16x16",
        &[
            conv_at(3, 1, 32, 16, 32, 42, (128, 0), true),
            conv_at(3, 1, 32, 32, 32, 43, (0, 0), true),
            conv_at(3, 2, 32, 32, 32, 44, (0, 0), true),
            conv_at(3, 1, 16, 32, 16, 45, (0, 125), false),
        ],
    )
    .await;
}
