//! A test set end to end without `edgetpu_compiler`: an int8 `.tflite`
//! from `tools/mnist/train.py` or `tools/cifar10/train.py` ->
//! `graph::Graph` -> generated layer programs.
//!
//!     cargo run --release --example eval -- [--dir target/mnist] [--model mlp|cnn] [--limit N] [--device]
//!     cargo run --release --example eval -- --dir target/cifar10 --model resnet8 [--device]
//!
//! The directory holds `<model>.tflite`, `test_images.bin` (uint8 pixels),
//! `test_labels.bin` and `<model>_tflite_logits.bin` (int8, 10 per image).
//!
//! Always: the host reference (bit-exact with the TPU) against TFLite's CPU
//! logits over the test set. With `--device`: every image on the TPU,
//! checked against the host reference byte for byte.

use reef_infer::graph::{Graph, i8_to_u8, u8_to_i8};
use reef_infer::runner::Device;

fn argmax(v: &[i8]) -> usize {
    (0..v.len())
        .max_by_key(|&i| (v[i], usize::MAX - i))
        .unwrap()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let dir = opt("--dir").unwrap_or("target/mnist".into());
    let device = args.iter().any(|a| a == "--device");
    let read = |f: &str| {
        std::fs::read(format!("{dir}/{f}"))
            .unwrap_or_else(|e| panic!("{dir}/{f}: {e} (run tools/mnist/train.py first)"))
    };

    let name = opt("--model").unwrap_or("mlp".into());
    let graph = Graph::from_tflite(&format!("{dir}/{name}.tflite")).unwrap();
    for (i, s) in graph.steps.iter().enumerate() {
        let ins: Vec<_> = s.inputs.iter().map(|i| i.shape).collect();
        println!(
            "layer {i}: {} {:?} -> {:?}",
            summary(&s.layer),
            ins,
            s.output_shape
        );
    }
    let images = read("test_images.bin");
    let px = graph.input_len(); // bytes per image
    let labels = read("test_labels.bin");
    let tflite: Vec<i8> = read(&format!("{name}_tflite_logits.bin"))
        .iter()
        .map(|&b| b as i8)
        .collect();
    let n = opt("--limit")
        .map(|s| s.parse().unwrap())
        .unwrap_or(labels.len())
        .min(labels.len());

    // Input: pixel / 255, quantized as TFLite does (int8), then to uint8.
    let (s, zp) = (graph.input.scale, graph.input.zero_point - 128);
    let quantize = |px: &[u8]| -> Vec<u8> {
        let q: Vec<i8> = px
            .iter()
            .map(|&p| {
                ((p as f32 / 255.0 / s).round_ties_even() + zp as f32).clamp(-128.0, 127.0) as i8
            })
            .collect();
        i8_to_u8(&q)
    };

    let (mut ours_ok, mut cpu_ok, mut same_top1, mut bytes_diff, mut max_diff) = (0, 0, 0, 0, 0);
    let mut refs = Vec::with_capacity(n);
    for i in 0..n {
        let x = quantize(&images[px * i..px * (i + 1)]);
        let y = u8_to_i8(&graph.reference(&x));
        let t = &tflite[10 * i..10 * (i + 1)];
        for (a, b) in y.iter().zip(t) {
            let d = (*a as i32 - *b as i32).abs();
            bytes_diff += (d != 0) as usize;
            max_diff = max_diff.max(d);
        }
        ours_ok += (argmax(&y) == labels[i] as usize) as usize;
        cpu_ok += (argmax(t) == labels[i] as usize) as usize;
        same_top1 += (argmax(&y) == argmax(t)) as usize;
        refs.push((x, y));
    }
    let pct = |c: usize| 100.0 * c as f64 / n as f64;
    println!("{n} images");
    println!("  TFLite CPU accuracy:     {:.2}%", pct(cpu_ok));
    println!("  host reference accuracy: {:.2}%", pct(ours_ok));
    println!(
        "  reference vs TFLite: {bytes_diff} of {} logits differ (max {max_diff}), top-1 agrees on {same_top1}",
        10 * n
    );

    if !device {
        return;
    }
    let programs = graph.build().unwrap();
    println!(
        "TPU: {} layers in {} calls per image (on-chip chains: {:?})",
        programs.steps.len(),
        programs.calls(),
        programs
            .chains
            .iter()
            .map(|c| c.0.clone())
            .collect::<Vec<_>>()
    );
    let mut dev = Device::open().await.expect("no Edge TPU");
    let loaded = graph.load(&mut dev, &programs).await.unwrap();
    let (mut tpu_ok, mut mismatched) = (0, 0);
    let t0 = std::time::Instant::now();
    for (i, (x, want)) in refs.iter().enumerate() {
        let y = match graph.run(&mut dev, &loaded, x).await {
            Ok(y) => u8_to_i8(&y),
            Err(e) => panic!("image {i}: {e:?} after {:.1} s", t0.elapsed().as_secs_f64()),
        };
        if &y != want {
            mismatched += 1;
            if mismatched <= 5 {
                println!("  image {i}: TPU {y:?} vs reference {want:?}");
            }
        }
        tpu_ok += (argmax(&y) == labels[i] as usize) as usize;
    }
    let dt = t0.elapsed();
    println!(
        "  TPU accuracy:            {:.2}% ({mismatched} images differ from the reference; {:.2} ms / image)",
        pct(tpu_ok),
        dt.as_secs_f64() * 1e3 / n as f64
    );
    if mismatched > 0 {
        std::process::exit(1);
    }
}

fn summary(l: &reef_infer::graph::Layer) -> String {
    use reef_infer::graph::Layer;
    match l {
        Layer::Fc(f) => format!("FC K={} N={} relu={}", f.k, f.n, f.relu),
        Layer::Conv(c) => format!(
            "CONV {}x{} {}->{} at {}x{} relu={}",
            c.k, c.k, c.cin, c.cout, c.height, c.width, c.relu
        ),
        Layer::MaxPool(p) => format!("MAXPOOL 2x2 at {}x{}x{}", p.height, p.width, p.c),
        Layer::AvgPool(p) => format!("AVGPOOL global at {}x{}x{}", p.height, p.width, p.c),
        Layer::Add(a) => format!(
            "ADD at {}x{}x{} relu={} weights {:?}",
            a.height,
            a.width,
            a.c,
            a.relu,
            a.weights()
        ),
    }
}
