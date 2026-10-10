//! On-chip loopback experiments (docs/EDGE_TPU.md section 6.3) on a compiled
//! depthwise model, whose weights reach the tiles through the loopback every
//! inference. The unmodified model's output on the same input is the
//! reference; a stale-data check runs a different parameter blob first, so a
//! loopback that delivers nothing shows up as the previous model's output.
//!
//!     cargo run --release --features usb --example loopback_probe -- [model] [filter]

use std::sync::Arc;

use reef_infer::isa;
use reef_infer::model::{ChunkInfo, TfliteModel, load_tflite_model};
use reef_infer::runner::Device;

const MODEL: &str = "target/isa_corpus/dwconv/\
    dwconv_C32_H16_W16_actnone_in_range4p0_k3_mult1_padsame_stride1/model_edgetpu.tflite";

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

/// Sets `field` of every instruction with layout `layout` for which `pick`
/// holds (on its decoded form); returns how many were patched.
fn patch(
    chunk: &mut ChunkInfo,
    layout: &str,
    pick: impl Fn(&isa::Decoded<'_>) -> bool,
    field: &str,
    value: i64,
) -> usize {
    let mut word = 0;
    let mut n = 0;
    for ins in isa::split(&chunk.bitstream.clone()).unwrap() {
        let mut d = ins.decode().unwrap();
        if d.layout.name == layout && pick(&d) {
            d.set(field, value).unwrap();
            for (j, w) in d.encode().unwrap().iter().enumerate() {
                chunk.bitstream[(word + j) * 16..(word + j + 1) * 16]
                    .copy_from_slice(&w.to_le_bytes());
            }
            n += 1;
        }
        word += ins.words.len();
    }
    assert!(n > 0, "no {layout} matched");
    n
}

fn looped(d: &isa::Decoded<'_>) -> bool {
    d.get("loop_en").unwrap() == 1
}

fn to_c1(d: &isa::Decoded<'_>) -> bool {
    d.get("to_c1").unwrap() == 1
}

fn exec(m: &mut TfliteModel) -> &mut ChunkInfo {
    &mut m.executables[0].instruction_chunks[0]
}

/// Second parameter blob: the caching blob with its first byte of every
/// 4-byte word bumped, so weights change while floats stay finite.
fn other_params(m: &TfliteModel) -> Vec<u8> {
    let mut p = m.executables[1].parameters.clone();
    for (i, b) in p.iter_mut().enumerate() {
        if i % 4 == 0 {
            *b = b.wrapping_add(37);
        }
    }
    p
}

struct Probe {
    slot: Option<Device>,
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

    /// Loads and runs `m` on `x`; `None` on a hang (device reopened).
    async fn run(&mut self, m: TfliteModel, x: &[u8]) -> Option<Vec<u8>> {
        let dev = self.slot.as_mut().unwrap();
        let r = match dev.load_params(Arc::new(m)).await {
            Ok(l) => dev.run_inference(&l, x.to_vec()).await,
            Err(e) => Err(e),
        };
        match r {
            Ok(v) => Some(v),
            Err(_) => {
                self.reopen().await;
                None
            }
        }
    }
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

type Mutate = Box<dyn Fn(&mut TfliteModel)>;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().map(String::as_str).unwrap_or(MODEL);
    let filter = args.get(1).cloned().unwrap_or_default();
    let base = || load_tflite_model(path).expect("load model");
    let alt = || {
        let mut m = base();
        m.executables[1].parameters = other_params(&m);
        m
    };
    let in_len = base().executables[0].input_layers[0].size_bytes as usize;

    let mut p = Probe {
        slot: Some(Device::open().await.expect("open device")),
    };
    let mut rng = Rng(0x0bad_cafe_1234_5678);
    let x = rng.bytes(in_len);

    // References: both blobs, unmodified program, and determinism.
    let ref_a = p.run(base(), &x).await.expect("baseline hangs");
    let ref_b = p.run(alt(), &x).await.expect("alt params hang");
    let again = p.run(base(), &x).await.unwrap();
    println!("model {path}");
    println!(
        "{:56} {} / {} bytes differ",
        "params A vs B (do the weights matter?)",
        diff(&ref_a, &ref_b),
        ref_a.len()
    );
    println!(
        "{:56} {}",
        "baseline repeatable",
        if again == ref_a { "yes" } else { "NO" }
    );

    let cases: Vec<(&str, Mutate)> = vec![
        ("baseline", Box::new(|_| {})),
        (
            "infeed loop_en = 0",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "loop_en", 0);
            }),
        ),
        (
            "infeed loop_k168/k308/k315 = 0",
            Box::new(|m| {
                for f in ["loop_k168", "loop_k308", "loop_k315"] {
                    patch(exec(m), "infeed", looped, f, 0);
                }
            }),
        ),
        (
            "infeed loop_en + k168/k308/k315 = 0 (tinycoral)",
            Box::new(|m| {
                for f in ["loop_k168", "loop_k308", "loop_k315", "loop_en"] {
                    patch(exec(m), "infeed", looped, f, 0);
                }
            }),
        ),
        (
            "infeed loop_k168 = 0",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "loop_k168", 0);
            }),
        ),
        (
            "infeed loop_k308 = 0",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "loop_k308", 0);
            }),
        ),
        (
            "infeed loop_k315 = 0",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "loop_k315", 0);
            }),
        ),
        (
            "outfeed loop_en = 0",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "loop_en", 0);
            }),
        ),
        (
            "outfeed loop_cfg = 0",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "loop_cfg", 0);
            }),
        ),
        (
            "outfeed bit 120 (d3_stride lsb) = 0",
            Box::new(|m| {
                let c = exec(m);
                let v = isa::split(&c.bitstream)
                    .unwrap()
                    .iter()
                    .map(|i| i.decode().unwrap())
                    .find(|d| d.layout.name == "outfeed" && looped(d))
                    .unwrap()
                    .get("d3_stride")
                    .unwrap();
                patch(c, "outfeed", looped, "d3_stride", v & !1);
            }),
        ),
        (
            "outfeed loop_row = 1024",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "loop_row", 1024);
            }),
        ),
        (
            "outfeed loop_row = 1536",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "loop_row", 1536);
            }),
        ),
        (
            "outfeed loop_row = 0",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "loop_row", 0);
            }),
        ),
        (
            "infeed tiles = 0x0001 (only tile 0)",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "tiles", 1);
            }),
        ),
        (
            "infeed tiles = 0x00ff",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "tiles", 0xff);
            }),
        ),
        (
            "outfeed k173 = 0",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "k173", 0);
            }),
        ),
        (
            "outfeed k173 = 2",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "k173", 2);
            }),
        ),
        (
            "outfeed k173 = 3 (as host outfeeds)",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "k173", 3);
            }),
        ),
        (
            "outfeed f222 = 0x7f (as host outfeeds)",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "f222", 0x7f);
            }),
        ),
        (
            "outfeed f222 = 0",
            Box::new(|m| {
                patch(exec(m), "outfeed", looped, "f222", 0);
            }),
        ),
        (
            "producer to_c1 = 0",
            Box::new(|m| {
                patch(exec(m), "ringProducer", to_c1, "to_c1", 0);
            }),
        ),
        (
            "producer addr = 0x40 (source shifted 64 B)",
            Box::new(|m| {
                patch(exec(m), "ringProducer", to_c1, "addr", 0x40);
            }),
        ),
        (
            "producer dest = 0",
            Box::new(|m| {
                patch(exec(m), "ringProducer", to_c1, "dest", 0);
            }),
        ),
        (
            "producer tile_mask = 0x2 (tile 1 as source)",
            Box::new(|m| {
                patch(exec(m), "ringProducer", to_c1, "tile_mask", 2);
            }),
        ),
        (
            "both loop_en = 0",
            Box::new(|m| {
                patch(exec(m), "infeed", looped, "loop_en", 0);
                patch(exec(m), "outfeed", looped, "loop_en", 0);
            }),
        ),
    ];

    for (name, mutate) in cases {
        if !name.contains(&filter) {
            continue;
        }
        // Run A (unmodified) first so the tiles hold A's broadcast, then the
        // perturbed program with blob B.
        let mut m = alt();
        mutate(&mut m);
        p.run(base(), &x).await.expect("baseline hangs");
        let verdict = match p.run(m, &x).await {
            None => "HANG".to_string(),
            Some(got) if got == ref_b => "correct".to_string(),
            Some(got) if got == ref_a => "STALE (= previous model's output)".to_string(),
            Some(got) => {
                // Per-tile breakdown: the output is 16 tiles' worth of y-x-z.
                let zeros = got.iter().filter(|&&v| v == 0).count();
                // Wrong values per channel (z, innermost) and per 512 B block.
                let c = 32.min(got.len());
                let mut per_ch = vec![0; c];
                let mut per_blk = vec![0; got.len().div_ceil(512)];
                for (i, (g, w)) in got.iter().zip(&ref_b).enumerate() {
                    if g != w {
                        per_ch[i % c] += 1;
                        per_blk[i / 512] += 1;
                    }
                }
                format!(
                    "wrong: {} vs B, {} vs A, {} zero bytes\n    per channel {:?}\n    per 512 B {:?}",
                    diff(&got, &ref_b),
                    diff(&got, &ref_a),
                    zeros,
                    per_ch,
                    per_blk
                )
            }
        };
        println!("{name:56} {verdict}");
    }

    let rows: Vec<i64> = std::env::var("ROWS")
        .map(|v| v.split(',').map(|r| r.parse().unwrap()).collect())
        .unwrap_or_default();
    for row in rows {
        let mut m = alt();
        patch(exec(&mut m), "outfeed", looped, "loop_row", row);
        let v = match p.run(m, &x).await {
            None => "HANG",
            Some(g) if g == ref_b => "correct",
            Some(_) => "wrong",
        };
        println!("{:56} {v}", format!("outfeed loop_row = {row}"));
    }
}
