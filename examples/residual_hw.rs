//! Residual blocks chained on chip: ResNet-8's first two stages (steps
//! 0-7: six convs and two adds, the shortcuts kept in narrow memory) as
//! one call, against the host reference and against one call per step.
//!
//!     cargo run --release --example residual_hw -- [--dir target/cifar10] [--limit N]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use reef_infer::codegen::chain::{self, Node, Op};
use reef_infer::graph::{Graph, Layer, i8_to_u8};
use reef_infer::runner::Device;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let dir = opt("--dir").unwrap_or("target/cifar10".into());
    let n: usize = opt("--limit").map(|s| s.parse().unwrap()).unwrap_or(200);
    let g = Graph::from_tflite(&format!("{dir}/resnet8.tflite")).unwrap();
    let images = std::fs::read(format!("{dir}/test_images.bin")).unwrap();
    let steps = &g.steps[..8];

    // chain nodes: tensor ids -> 0 (the chain input) or k + 1
    let mut index = HashMap::from([(g.input_tensor, 0usize)]);
    let bases = g.wide_bases().unwrap();
    let mut nodes = Vec::new();
    for (k, s) in steps.iter().enumerate() {
        let op = match &s.layer {
            Layer::Conv(c) => Op::Conv(c, bases[k]),
            Layer::Add(a) => Op::Add(a),
            l => panic!("step {k}: {l:?}"),
        };
        let inputs = s.inputs.iter().map(|i| index[&i.tensor]).collect();
        nodes.push(Node { op, inputs });
        index.insert(s.output, k + 1);
    }
    let plan = chain::plan(&nodes).unwrap();
    println!("narrow plan (scratch at {:#x}):", plan.scratch);
    for (t, p) in plan.at.iter().enumerate() {
        if let Some((a, p)) = p {
            println!(
                "  tensor {t}: {a:#07x}..{:#07x}, pad t{} b{} l{} r{}",
                a + p.tp(),
                p.top,
                p.bottom,
                p.left,
                p.right
            );
        }
    }

    let programs = g.build().unwrap();
    let mut dev = Device::open().await.expect("no Edge TPU");
    let loaded = g.load(&mut dev, &programs).await.unwrap();
    let chained = dev
        .load_params(Arc::new(g.chain(0..8).unwrap()))
        .await
        .unwrap();

    let px = g.input_len();
    let (s, zp) = (g.input.scale, g.input.zero_point - 128);
    let input = |i: usize| {
        let q: Vec<i8> = images[px * i..px * (i + 1)]
            .iter()
            .map(|&p| {
                ((p as f32 / 255.0 / s).round_ties_even() + zp as f32).clamp(-128.0, 127.0) as i8
            })
            .collect();
        i8_to_u8(&q)
    };
    // host reference of steps 0-7 and the program inputs of step 0
    let reference = |x: &[u8]| {
        let mut acts = HashMap::from([(g.input_tensor, x.to_vec())]);
        for s in steps {
            let ins: Vec<&[u8]> = s.inputs.iter().map(|i| &acts[&i.tensor][..]).collect();
            acts.insert(s.output, s.reference(&ins));
        }
        acts.remove(&steps[7].output).unwrap()
    };
    let xs: Vec<Vec<u8>> = (0..n).map(input).collect();
    let feeds: Vec<Vec<u8>> = xs.iter().map(|x| steps[0].fit(&[x]).concat()).collect();
    let refs: Vec<Vec<u8>> = xs.iter().map(|x| reference(x)).collect();
    let mut bad = 0;
    for (f, r) in feeds.iter().zip(&refs) {
        let y = dev.run_inference(&chained, f.clone()).await.unwrap();
        bad += (&steps[7].crop(&y) != r) as usize;
    }
    println!(
        "one call, steps 0-7: {} of {n} images differ from the host reference",
        bad
    );

    // timing: one call per step (host glue as Graph::run) vs one chained call
    let reps = n.min(100);
    let t = Instant::now();
    for x in &xs[..reps] {
        let mut acts = HashMap::from([(g.input_tensor, x.clone())]);
        for (s, p) in steps.iter().zip(&loaded.steps) {
            let ins: Vec<&[u8]> = s.inputs.iter().map(|i| &acts[&i.tensor][..]).collect();
            let y = dev.run_inference(p, s.fit(&ins).concat()).await.unwrap();
            acts.insert(s.output, s.crop(&y));
        }
    }
    let sep = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let t = Instant::now();
    for x in &xs[..reps] {
        let f = steps[0].fit(&[x]).concat();
        let y = dev.run_inference(&chained, f).await.unwrap();
        std::hint::black_box(steps[7].crop(&y));
    }
    let one = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("8 calls {sep:.3} ms / image, one chained call {one:.3} ms / image");
}
