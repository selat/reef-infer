//! Do two models' cached parameters coexist on the TPU? Loads FC A
//! (784 x 128, tiles 0-1), then FC B (128 x 10, tile 0), then runs A with
//! A's handle and checks it against the host reference. Both blobs start at
//! wide address 0, so A is expected to come back wrong if B overwrote it.
//!
//!     cargo run --example param_cache_hw

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::fc::Fc;
use reef_infer::runner::Device;

fn fc(k: usize, n: usize, seed: u64) -> Fc {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as u8
    };
    Fc {
        k,
        n,
        weights: (0..n * k).map(|_| next() as i8).collect(),
        w_scale: vec![0.004; n],
        bias: (0..n).map(|i| 100 * i as i32).collect(),
        input: Quant {
            scale: 0.03,
            zero_point: 120,
        },
        output: Quant {
            scale: 0.3 * (k as f32 / 64.0).sqrt(),
            zero_point: 128,
        },
        relu: false,
    }
}

#[tokio::main]
async fn main() {
    let (a, b) = (fc(784, 128, 1), fc(128, 10, 2));
    let x: Vec<u8> = (0..784).map(|i| (i * 7 % 256) as u8).collect();
    let mut dev = Device::open().await.expect("no Edge TPU");
    let la = dev.load_params(Arc::new(a.build().unwrap())).await.unwrap();
    let ok = dev.run_inference(&la, x.clone()).await.unwrap() == a.reference(&x);
    println!("A after loading A: {}", if ok { "exact" } else { "WRONG" });
    let lb = dev.load_params(Arc::new(b.build().unwrap())).await.unwrap();
    let xb = x[..128].to_vec();
    let ok = dev.run_inference(&lb, xb.clone()).await.unwrap() == b.reference(&xb);
    println!("B after loading B: {}", if ok { "exact" } else { "WRONG" });
    let ok = dev.run_inference(&la, x.clone()).await.unwrap() == a.reference(&x);
    println!(
        "A after loading B: {}",
        if ok {
            "exact (parameters coexist)"
        } else {
            "WRONG (B overwrote A's parameters)"
        }
    );
}
