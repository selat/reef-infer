//! Builds `codegen::pool` programs for the corpus MAX_POOL_2D models it
//! supports, and `codegen::avg_pool` for the global average pool, and diffs
//! them against the compiler's, field by field.
//!
//!     cargo run --example pool_check -- [filter] [--all-diffs] [--unchecked]
//!
//! `--unchecked` also builds the global pools `GlobalAvgPool::check` does
//! not take (to derive new shapes against the compiler).
//!
//! Corpus directory: `target/isa_corpus` or `REEF_ISA_CORPUS`.

use std::path::PathBuf;

use reef_infer::codegen::Quant;
use reef_infer::codegen::avg_pool::GlobalAvgPool;
use reef_infer::codegen::pool::MaxPool;
use reef_infer::isa;
use reef_infer::model::{ExecInfo, LayerInfo, TfliteModel, load_tflite_model};

fn quant(l: &LayerInfo) -> Quant {
    Quant {
        scale: l.dequantization_factor,
        zero_point: l.zero_point,
    }
}

/// Field diffs between two bitstreams; `None` when identical.
fn diff(ours: &ExecInfo, theirs: &ExecInfo, max: usize) -> Option<String> {
    let a = isa::split(&ours.instruction_chunks[0].bitstream).unwrap();
    let b = isa::split(&theirs.instruction_chunks[0].bitstream).unwrap();
    let mut out = Vec::new();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x.words == y.words {
            continue;
        }
        let (dx, dy) = (x.decode().unwrap(), y.decode().unwrap());
        if dx.layout.name != dy.layout.name {
            out.push(format!(
                "  #{i}: ours {} vs theirs {}",
                dx.layout.name, dy.layout.name
            ));
        } else {
            let fields: Vec<String> = dx
                .layout
                .fields
                .iter()
                .zip(dx.values.iter().zip(&dy.values))
                .filter(|(_, (u, v))| u != v)
                .map(|(f, (u, v))| format!("{}: {u:#x} vs {v:#x}", f.name))
                .collect();
            out.push(format!("  #{i} {}: {}", dx.layout.name, fields.join(", ")));
        }
        if out.len() >= max {
            break;
        }
    }
    if a.len() != b.len() {
        out.push(format!("  instruction count {} vs {}", a.len(), b.len()));
    }
    let offs = |e: &ExecInfo| -> Vec<i32> {
        e.instruction_chunks[0]
            .field_offsets
            .iter()
            .map(|f| f.offset_bit)
            .collect()
    };
    if offs(ours) != offs(theirs) {
        out.push(format!(
            "  relocations {:?} vs {:?}",
            offs(ours),
            offs(theirs)
        ));
    }
    (!out.is_empty()).then(|| out.join("\n"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let filter = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_default();
    let unchecked = args.iter().any(|a| a == "--unchecked");
    let max_diffs = if args.iter().any(|a| a == "--all-diffs") {
        usize::MAX
    } else {
        8
    };
    let dir = std::env::var("REEF_ISA_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/isa_corpus"))
        .join("pool");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("corpus")
        .flatten()
        .map(|e| e.file_name().into_string().unwrap())
        .collect();
    names.sort();
    let (mut same, mut total) = (0, 0);
    for name in names.iter().filter(|n| n.contains(&filter)) {
        let p: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(name).join("params.json")).unwrap(),
        )
        .unwrap();
        // (a non-square window is a list: not compared)
        let get = |k: &str| p[k].as_u64().unwrap_or(0) as usize;
        let global = p["op"] == "avg" && get("size") == get("H") && get("size") == get("W");
        let max = p["op"] == "max" && p["pad"] == "valid" && get("size") == 2 && get("stride") == 2;
        if !(global || max) {
            continue;
        }
        let compiled: TfliteModel = load_tflite_model(
            dir.join(name)
                .join("model_edgetpu.tflite")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let e = &compiled.executables[0];
        let q = quant(&e.input_layers[0]);
        let ours = if global {
            // the compiler's form: centred on the zero point, mult 1 / n
            let pool = GlobalAvgPool {
                height: get("H"),
                width: get("W"),
                c: get("C"),
                center: q.zero_point,
                mult: 1.0 / (get("H") * get("W")) as f32,
                quant: q,
            };
            if pool.check().is_err() && !unchecked {
                continue;
            }
            match pool.build_unchecked() {
                Ok(p) => p.into_model(),
                Err(e) => {
                    println!("{name}: build: {e}");
                    continue;
                }
            }
        } else {
            let pool = MaxPool {
                height: get("H"),
                width: get("W"),
                c: get("C"),
                quant: q,
            };
            if pool.check().is_err() {
                continue;
            }
            pool.build().unwrap().into_model()
        };
        total += 1;
        let mut report = Vec::new();
        if let Some(d) = diff(&ours.executables[0], e, max_diffs) {
            report.push(format!(" {:?}:\n{d}", e.exec_type));
        }
        let lay = |m: &ExecInfo| format!("{:?}", m.output_layers[0].output_layout);
        if lay(&ours.executables[0]) != lay(e) {
            report.push(format!(
                " output layout: {} vs {}",
                lay(&ours.executables[0]),
                lay(e)
            ));
        }
        if report.is_empty() {
            same += 1;
            println!("{name}: identical");
        } else {
            println!("{name}:\n{}", report.join("\n"));
        }
    }
    println!("{same}/{total} identical");
}
