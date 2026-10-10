//! Generated on-chip loopback (`elementwise::build_broadcast`): tile `src`
//! -> scalar memory -> all 16 tiles. The other tiles hold zeros, and the
//! host producers are pointed at each received 16 B unit's wide address, so
//! every tile's copy is read back and checked byte for byte.
//!
//!     cargo run --release --features usb --example broadcast_hw -- [src...]
//!     PARTS=7f,0f ... (hex, `elementwise::part`) runs subsets for bisecting
//!
//! Tile order: tile `4 * (y / 4) + x / 4` holds a 4x4 pixel patch.

use std::sync::Arc;

use reef_infer::codegen::Quant;
use reef_infer::codegen::elementwise::{self, Requant, SHAPE, SIZE};
use reef_infer::isa;
use reef_infer::model::TfliteModel;
use reef_infer::runner::Device;

const ID: Requant = Requant {
    in_zp: 0,
    mult: 1.0,
    clamp_min: 0.0,
    clamp_max: 255.0,
    out_zp: 0,
};
const Q: Quant = Quant {
    scale: 1.0,
    zero_point: 0,
};

/// Which source patch each output patch matches, or '?' (16 chars, tile order).
fn patch_map(got: &[u8], x: &[u8]) -> String {
    (0..16)
        .map(|t| {
            let (ty, tx) = (4 * (t / 4), 4 * (t % 4));
            let patch = |buf: &[u8], py: usize, px: usize| -> Vec<u8> {
                (0..4)
                    .flat_map(|y| {
                        let r = ((py + y) * SHAPE.1 + px) * SHAPE.2;
                        buf[r..r + 4 * SHAPE.2].to_vec()
                    })
                    .collect()
            };
            let g = patch(got, ty, tx);
            (0..16)
                .find(|&s| g == patch(x, 4 * (s / 4), 4 * (s % 4)))
                .map(|s| char::from_digit(s as u32, 16).unwrap())
                .unwrap_or(if g.iter().all(|&v| v == 0) { '0' } else { '?' })
        })
        .collect()
}

/// Offset in tile `src`'s 512 B of y-x-z index `i`, if `i` is in its patch.
fn tile_local(i: usize, src: usize) -> Option<usize> {
    let (y, xx) = ((i / SHAPE.2) / SHAPE.1, (i / SHAPE.2) % SHAPE.1);
    let (sy, sx) = (4 * (src / 4), 4 * (src % 4));
    (y >= sy && y < sy + 4 && xx >= sx && xx < sx + 4)
        .then(|| ((y - sy) * 4 + xx - sx) * SHAPE.2 + i % SHAPE.2)
}

/// Tile `t`'s 512 B of a y-x-z output, in tile order.
fn tile_bytes(g: &[u8], t: usize) -> Vec<u8> {
    let (ty, tx) = (4 * (t / 4), 4 * (t % 4));
    (0..4)
        .flat_map(|y| {
            let r = ((ty + y) * SHAPE.1 + tx) * SHAPE.2;
            g[r..r + 4 * SHAPE.2].to_vec()
        })
        .collect()
}

/// Broadcasts from `src` (all other tiles hold zeros) and has every tile
/// return the 512 B at wide address `addr` instead of 0x2078. Two passes
/// recover, per tile and byte, the offset in `src`'s 512 B it came from.
async fn readback(
    slot: &mut Option<Device>,
    src: usize,
    addr: i64,
) -> Option<Vec<Vec<Option<usize>>>> {
    let mut passes = Vec::new();
    for pass in 0..2 {
        let xp: Vec<u8> = (0..SIZE)
            .map(|i| match tile_local(i, src) {
                Some(l) if pass == 0 => (l % 255 + 1) as u8,
                Some(l) => (l / 255 + 1) as u8,
                None => 0,
            })
            .collect();
        let mut m = elementwise::build_broadcast(&ID, Q, Q, src)
            .unwrap()
            .into_model();
        let c = &mut m.executables[0].instruction_chunks[0];
        let mut word = 0;
        for ins in isa::split(&c.bitstream.clone()).unwrap() {
            let mut d = ins.decode().unwrap();
            if d.layout.name == "ringProducer" && d.get("r0_id") == Some(0x51) {
                d.set("addr", addr).unwrap();
                for (j, w) in d.encode().unwrap().iter().enumerate() {
                    c.bitstream[(word + j) * 16..(word + j + 1) * 16]
                        .copy_from_slice(&w.to_le_bytes());
                }
            }
            word += ins.words.len();
        }
        passes.push(run(slot, m, &xp).await?);
    }
    Some(
        (0..16)
            .map(|t| {
                tile_bytes(&passes[0], t)
                    .iter()
                    .zip(tile_bytes(&passes[1], t))
                    .map(|(&lo, hi)| {
                        (lo != 0 && hi != 0).then(|| (hi as usize - 1) * 255 + lo as usize - 1)
                    })
                    .collect()
            })
            .collect(),
    )
}

/// Received 16 B units at every tile: reads cover each unit's address.
async fn verify(slot: &mut Option<Device>, src: usize) -> Result<(), String> {
    let mut bad = Vec::new();
    for u in 0..24 {
        let a = elementwise::bcast_unit_addr(u);
        let r = readback(slot, src, a).await.ok_or("hang")?;
        for (t, tile) in r.iter().enumerate() {
            let want: Vec<Option<usize>> = (16 * u..16 * u + 16).map(Some).collect();
            if tile[..16] != want[..] {
                bad.push(format!("tile {t} unit {u} @0x{a:04x}: {:?}", &tile[..4]));
            }
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} wrong: {}",
            bad.len(),
            bad[..bad.len().min(4)].join("; ")
        ))
    }
}

async fn run(slot: &mut Option<Device>, m: TfliteModel, x: &[u8]) -> Option<Vec<u8>> {
    let dev = slot.as_mut().unwrap();
    let r = match dev.load_params(Arc::new(m)).await {
        Ok(l) => dev.run_inference(&l, x.to_vec()).await,
        Err(e) => Err(e),
    };
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            println!("    error: {e}");
            *slot = None;
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if let Ok(d) = Device::open().await {
                    *slot = Some(d);
                    return None;
                }
            }
            panic!("device did not come back");
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let srcs: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let srcs = if srcs.is_empty() {
        vec![0, 5, 15]
    } else {
        srcs
    };
    let mut slot = Some(Device::open().await.expect("open device"));
    let mut seed = 0x5eed_1234_abcd_0001u64;
    let x: Vec<u8> = (0..SIZE)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        })
        .collect();

    let ctl = elementwise::build(&ID, Q, Q, true).unwrap().into_model();
    match run(&mut slot, ctl, &x).await {
        Some(g) => println!(
            "{:40} {}",
            "control (no loopback)",
            if g == x { "identity, exact" } else { "WRONG" }
        ),
        None => println!("{:40} HANG", "control (no loopback)"),
    }
    // PARTS=hex,hex,... runs those piece sets (from tile src[0]) instead.
    if let Ok(v) = std::env::var("PARTS") {
        for p in v.split(',') {
            let parts = u8::from_str_radix(p, 16).unwrap();
            let m = elementwise::build_broadcast_parts(&ID, Q, Q, srcs[0], parts)
                .unwrap()
                .into_model();
            let name = format!("parts 0x{parts:02x} from tile {}", srcs[0]);
            match run(&mut slot, m, &x).await {
                None => println!("{name:40} HANG"),
                Some(g) if g == x => println!("{name:40} identity"),
                Some(g) => println!("{name:40} patches {}", patch_map(&g, &x)),
            }
        }
        return;
    }
    for src in srcs {
        let name = format!("broadcast from tile {src}");
        match verify(&mut slot, src).await {
            Ok(()) => println!("{name:40} 384 B (24 x 16 B) exact at all 16 tiles"),
            Err(e) => println!("{name:40} {e}"),
        }
    }
}
