//! Convolutions larger than 16x16 (`Conv::width` / `height` 32, 64) on the TPU against
//! the host reference; on a mismatch prints which output pixels are wrong.
//!
//!     cargo run --release --features usb --example conv_wide_hw

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::conv::Conv;
use reef_infer::runner::Device;

fn bytes(seed: &mut u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed >> 32) as u8
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let mut dev = Some(Device::open().await.expect("open device"));
    let mut seed = 0x1357_9bdf_2468_ace0u64;
    let cases: Vec<(usize, usize, usize, usize, usize, usize)> = match std::env::var("CASES") {
        // CASES=k,s,height,width,cin,cout;...  (built without Conv::check)
        Ok(v) => v
            .split(';')
            .map(|c| {
                let n: Vec<usize> = c.split(',').map(|x| x.parse().unwrap()).collect();
                (n[0], n[1], n[2], n[3], n[4], n[5])
            })
            .collect(),
        Err(_) => vec![
            (1, 1, 16, 32, 16, 32),
            (3, 1, 16, 32, 16, 32),
            (3, 1, 16, 32, 32, 16),
            (3, 1, 16, 64, 16, 16),
            (3, 2, 16, 32, 16, 32),
            (3, 2, 16, 64, 32, 16),
            (1, 2, 16, 64, 16, 32),
        ],
    };
    for (k, stride, height, width, cin, cout) in cases {
        let name = format!("k={k} s={stride} {height}x{width} Cin={cin} Cout={cout}");
        let w = bytes(&mut seed, cout * k * k * cin);
        let conv = Conv {
            cin,
            cout,
            k,
            stride,
            width,
            height,
            weights: w.iter().map(|&b| b as i8).collect(),
            w_scale: (0..cout)
                .map(|i| (0.0007 + 0.0001 * i as f32) * 16.0 / cin as f32)
                .collect(),
            bias: (0..cout).map(|i| 300 * i as i32 - 4000).collect(),
            input: Quant {
                scale: 0.03,
                zero_point: 120,
            },
            output: Quant {
                scale: 0.2,
                zero_point: 128,
            },
            relu: false,
            pad_before: None,
        };
        let model = match conv.build_unchecked() {
            Ok(m) => m,
            Err(e) => {
                println!("{name:40} build: {e}");
                continue;
            }
        };
        let d = dev.as_mut().unwrap();
        let x = bytes(&mut seed, height * width * cin);
        let r = match d.load_params(Arc::new(model)).await {
            Ok(l) => d.run_inference(&l, x.clone()).await,
            Err(e) => Err(e),
        };
        let got = match r {
            Ok(g) => g,
            Err(e) => {
                println!("{name:40} {e}");
                dev = None;
                for _ in 0..20 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if let Ok(d) = Device::open().await {
                        dev = Some(d);
                        break;
                    }
                }
                assert!(dev.is_some(), "device did not come back");
                continue;
            }
        };
        let want = conv.reference(&x);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        let distinct = {
            let mut v = got.clone();
            v.sort();
            v.dedup();
            v.len()
        };
        if bad == 0 && got.len() == want.len() {
            println!(
                "{name:40} exact ({} values, {distinct} distinct)",
                want.len()
            );
            continue;
        }
        println!(
            "{name:40} {bad}/{} wrong (len {} vs {})",
            want.len(),
            got.len(),
            want.len()
        );
        // wrong pixels map ('#' = any channel wrong)
        let (oh, ow) = (height / stride, width / stride);
        for y in 0..oh {
            let row: String = (0..ow)
                .map(|x| {
                    let p = (y * ow + x) * cout;
                    if got.get(p..p + cout) == want.get(p..p + cout) {
                        '.'
                    } else {
                        '#'
                    }
                })
                .collect();
            println!("    {row}");
        }
    }
}
