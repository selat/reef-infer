//! Do several layers' cached parameters coexist on the TPU when each is
//! placed at its own wide address (`build_at`)? Loads every layer of a case
//! in turn, then runs them all again and checks each against its host
//! reference: a layer whose parameters a later load overwrote comes back
//! wrong.
//!
//!     cargo run --release --example param_cache_hw
//!
//! Cases: two FCs (one on 2 tiles), two convs, FC and conv mixed, layers
//! near the top of the parameter range, and placements tighter than
//! `wide_units` (one 64 B unit per parameter row instead of four) to see
//! how much wide memory a row really takes.

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::conv::Conv;
use reef_infer::codegen::fc::Fc;
use reef_infer::model::TfliteModel;
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

fn fc(k: usize, n: usize, seed: u64) -> Fc {
    let mut next = rng(seed);
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

fn conv(cin: usize, cout: usize, seed: u64) -> Conv {
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
            zero_point: 100,
        },
        output: Quant {
            scale: 0.2,
            zero_point: 120,
        },
        relu: false,
        pad_before: None,
    }
}

enum L {
    Fc(Fc),
    Conv(Conv),
}

impl L {
    fn units(&self) -> i64 {
        match self {
            L::Fc(f) => f.wide_units(),
            L::Conv(c) => c.wide_units(),
        }
    }
    fn build_at(&self, base: i64) -> TfliteModel {
        match self {
            L::Fc(f) => f.build_at(base).unwrap(),
            L::Conv(c) => c.build_at(base).unwrap(),
        }
    }
    fn input(&self, seed: u64) -> Vec<u8> {
        let n = match self {
            L::Fc(f) => f.k,
            L::Conv(c) => c.width * c.height * c.cin,
        };
        let mut next = rng(seed);
        (0..n).map(|_| next()).collect()
    }
    fn reference(&self, x: &[u8]) -> Vec<u8> {
        match self {
            L::Fc(f) => f.reference(x),
            L::Conv(c) => c.reference(x),
        }
    }
    fn name(&self) -> String {
        match self {
            L::Fc(f) => format!("fc {}x{}", f.k, f.n),
            L::Conv(c) => format!("conv {}->{} k3", c.cin, c.cout),
        }
    }
}

/// Loads `layers` at `bases` in order, then runs each; true if all exact.
async fn case(dev: &mut Device, title: &str, layers: &[(L, i64)]) -> bool {
    println!("{title}");
    let mut loaded = Vec::new();
    for (l, base) in layers {
        let m = Arc::new(l.build_at(*base));
        let p = dev.load_params(m).await.unwrap();
        // right after its own load
        let x = l.input(*base as u64 + 1);
        let ok = dev.run_inference(&p, x.clone()).await.unwrap() == l.reference(&x);
        println!(
            "  {:<16} base {base:#06x} ({} units): {}",
            l.name(),
            l.units(),
            if ok { "exact" } else { "WRONG" }
        );
        loaded.push(p);
    }
    let mut all = true;
    for ((l, base), p) in layers.iter().zip(&loaded) {
        let x = l.input(*base as u64 + 7);
        let ok = dev.run_inference(p, x.clone()).await.unwrap() == l.reference(&x);
        all &= ok;
        println!(
            "    after all loads, {:<16}: {}",
            l.name(),
            if ok { "exact" } else { "WRONG" }
        );
    }
    all
}

#[tokio::main]
async fn main() {
    let mut dev = Device::open().await.expect("no Edge TPU");
    let mut results = Vec::new();

    let a = fc(784, 128, 1);
    let ua = a.wide_units();
    results.push(
        case(
            &mut dev,
            "FC A (2 tiles) then FC B after A's units",
            &[(L::Fc(a), 0), (L::Fc(fc(128, 10, 2)), ua)],
        )
        .await,
    );
    // B at 0 overwrites A: the reference failure of the old code
    results.push(
        !case(
            &mut dev,
            "FC A and FC B both at 0 (expected: A wrong)",
            &[(L::Fc(fc(784, 128, 1)), 0), (L::Fc(fc(128, 10, 2)), 0)],
        )
        .await,
    );
    let c = conv(16, 32, 3);
    let uc = c.wide_units();
    results.push(
        case(
            &mut dev,
            "conv A then conv B after A's units",
            &[(L::Conv(c), 0), (L::Conv(conv(16, 16, 4)), uc)],
        )
        .await,
    );
    let f = fc(256, 64, 5);
    let uf = f.wide_units();
    let c = conv(32, 32, 6);
    let uc = c.wide_units();
    results.push(
        case(
            &mut dev,
            "FC, conv, FC packed",
            &[
                (L::Fc(f), 0),
                (L::Conv(c), uf),
                (L::Fc(fc(64, 10, 7)), uf + uc),
            ],
        )
        .await,
    );
    results.push(
        case(
            &mut dev,
            "high bases",
            &[
                (L::Conv(conv(16, 32, 8)), 0x1000),
                (L::Fc(fc(128, 64, 9)), 0x1e00 - fc(128, 64, 9).wide_units()),
            ],
        )
        .await,
    );
    let ok = results.iter().all(|&r| r);

    // Tighter than wide_units: one unit per 256 B row. Informational.
    let big = fc(1024, 64, 10);
    let rows: i64 = 3 + 1024 / 4;
    case(
        &mut dev,
        "tight: FC 1024x64 rows at one unit each, then a small FC right after",
        &[(L::Fc(big), 0), (L::Fc(fc(64, 16, 11)), (rows + 3) / 4 * 4)],
    )
    .await;
    let big = conv(64, 32, 12);
    let words: i64 = 3 + 9 * 64 / 4;
    case(
        &mut dev,
        "tight: conv 64->32 words at one unit each, then a small conv right after",
        &[
            (L::Conv(big), 0),
            (L::Conv(conv(16, 16, 13)), (words + 2 + 3) / 4 * 4),
        ],
    )
    .await;
    println!(
        "\nplacement {}",
        if ok {
            "works: every layer exact after all loads"
        } else {
            "FAILED"
        }
    );
}
