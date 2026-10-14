//! Speed and accuracy of our generated programs against `edgetpu_compiler`'s
//! for the same int8 `.tflite`, on the same TPU and test images.
//!
//!     cargo run --release --example bench_compare -- [--dir target/cifar10] [--model resnet8] [--limit N]
//!
//! Needs `<model>.tflite` (with static shapes) and `<model>_edgetpu.tflite`
//! (`edgetpu_compiler -s <model>.tflite`) next to the test set of
//! `examples/eval.rs`. Reports, per image: the compiled model (weights
//! cached once), our layer programs as `Graph::run` runs them (every layer
//! reloads its weights), and a per-layer breakdown of ours into weight
//! loading and inference with the weights already cached.

use std::sync::Arc;
use std::time::Instant;

use reef_infer::graph::{Graph, i8_to_u8, u8_to_i8};
use reef_infer::model::load_tflite_model;
use reef_infer::runner::Device;

fn argmax(v: &[i8]) -> usize {
    (0..v.len())
        .max_by_key(|&i| (v[i], usize::MAX - i))
        .unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let dir = opt("--dir").unwrap_or("target/cifar10".into());
    let name = opt("--model").unwrap_or("resnet8".into());
    let read =
        |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("{dir}/{f}: {e}"));

    let graph = Graph::from_tflite(&format!("{dir}/{name}.tflite")).unwrap();
    let compiled = Arc::new(load_tflite_model(&format!("{dir}/{name}_edgetpu.tflite")).unwrap());
    let images = read("test_images.bin");
    let labels = read("test_labels.bin");
    let tflite: Vec<i8> = read(&format!("{name}_tflite_logits.bin"))
        .iter()
        .map(|&b| b as i8)
        .collect();
    let px = graph.input_len();
    let n = opt("--limit")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1000)
        .min(labels.len());
    let (s, zp) = (graph.input.scale, graph.input.zero_point - 128);
    let inputs: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let q: Vec<i8> = images[px * i..px * (i + 1)]
                .iter()
                .map(|&p| {
                    ((p as f32 / 255.0 / s).round_ties_even() + zp as f32).clamp(-128.0, 127.0)
                        as i8
                })
                .collect();
            i8_to_u8(&q)
        })
        .collect();
    let programs = graph.build().unwrap();
    let mut dev = Device::open().await.expect("no Edge TPU");
    println!(
        "{name}: {n} images, {} generated programs\n",
        programs.len()
    );

    // ── edgetpu_compiler: weights cached once, then one call per image.
    let t = Instant::now();
    let loaded = dev.load_params(Arc::clone(&compiled)).await.unwrap();
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    for x in inputs.iter().take(10) {
        dev.run_inference(&loaded, x.clone()).await.unwrap(); // warm-up
    }
    let (mut ok, mut same_tflite, mut diff_bytes) = (0, 0, 0);
    let t = Instant::now();
    for (i, x) in inputs.iter().enumerate() {
        let y = u8_to_i8(&dev.run_inference(&loaded, x.clone()).await.unwrap());
        let tl = &tflite[10 * i..10 * (i + 1)];
        ok += (argmax(&y) == labels[i] as usize) as usize;
        same_tflite += (argmax(&y) == argmax(tl)) as usize;
        diff_bytes += y.iter().zip(tl).filter(|(a, b)| a != b).count();
    }
    let comp_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    let pct = |c: usize| 100.0 * c as f64 / n as f64;
    println!("edgetpu_compiler (one program, weights cached on chip):");
    println!("  weight load once:  {load_ms:.2} ms");
    println!(
        "  per image:         {comp_ms:.3} ms  ({:.0} images/s)",
        1e3 / comp_ms
    );
    println!(
        "  accuracy {:.2}%, top-1 = TFLite on {same_tflite}/{n}, {diff_bytes} of {} logits differ from TFLite",
        pct(ok),
        10 * n
    );

    // ── Ours, as Graph::run runs it.
    let (mut ok, mut same_tflite, mut diff_bytes, mut not_ref) = (0, 0, 0, 0);
    for x in inputs.iter().take(3) {
        graph.run(&mut dev, &programs, x).await.unwrap(); // warm-up
    }
    let t = Instant::now();
    for (i, x) in inputs.iter().enumerate() {
        let y = graph.run(&mut dev, &programs, x).await.unwrap();
        not_ref += (y != graph.reference(x)) as usize;
        let y = u8_to_i8(&y);
        let tl = &tflite[10 * i..10 * (i + 1)];
        ok += (argmax(&y) == labels[i] as usize) as usize;
        same_tflite += (argmax(&y) == argmax(tl)) as usize;
        diff_bytes += y.iter().zip(tl).filter(|(a, b)| a != b).count();
    }
    // (the host reference check is inside the timed loop; time it alone)
    let t_ref = Instant::now();
    for x in &inputs {
        std::hint::black_box(graph.reference(x));
    }
    let ref_ms = t_ref.elapsed().as_secs_f64() * 1e3 / n as f64;
    let ours_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64 - ref_ms;
    println!(
        "\nours ({} programs through the host, every layer reloads its weights):",
        programs.len()
    );
    println!(
        "  per image:         {ours_ms:.3} ms  ({:.0} images/s)",
        1e3 / ours_ms
    );
    println!(
        "  accuracy {:.2}%, top-1 = TFLite on {same_tflite}/{n}, {diff_bytes} of {} logits differ from TFLite, {not_ref} images differ from our host reference",
        pct(ok),
        10 * n
    );

    // ── Per layer: weight loading vs inference with cached weights, on the
    // activations of the first image.
    println!("\nours per layer (median of 30):");
    println!("  {:<38} {:>10} {:>12}", "layer", "load ms", "infer ms");
    let mut acts = std::collections::HashMap::from([(graph.input_tensor, inputs[0].clone())]);
    let (mut sum_load, mut sum_infer) = (0.0, 0.0);
    for (st, p) in graph.steps.iter().zip(&programs) {
        let ins: Vec<&[u8]> = st.inputs.iter().map(|i| &acts[&i.tensor][..]).collect();
        let x = st.fit(&ins).concat();
        let mut loads = Vec::new();
        let mut loaded = None;
        for _ in 0..30 {
            let t = Instant::now();
            loaded = Some(dev.load_params(Arc::clone(p)).await.unwrap());
            loads.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let loaded = loaded.unwrap();
        let mut infers = Vec::new();
        let mut y = Vec::new();
        for _ in 0..30 {
            let t = Instant::now();
            y = dev.run_inference(&loaded, x.clone()).await.unwrap();
            infers.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (l, r) = (median(loads), median(infers));
        sum_load += l;
        sum_infer += r;
        let desc = format!("{:?}", st.layer);
        let desc: String = desc
            .chars()
            .take(desc.find('{').unwrap_or(10).min(10))
            .collect();
        println!(
            "  {:<38} {l:>10.3} {r:>12.3}",
            format!("{desc} {:?} -> {:?}", st.inputs[0].shape, st.output_shape)
        );
        acts.insert(st.output, st.crop(&y));
    }
    println!("  {:<38} {sum_load:>10.3} {sum_infer:>12.3}", "total");
    println!(
        "\nsummary: compiled {comp_ms:.2} ms / image; ours {ours_ms:.2} ms ({:.1}x), of which {sum_infer:.2} ms is inference with cached weights ({:.1}x the compiled model)",
        ours_ms / comp_ms,
        sum_infer / comp_ms
    );
}
