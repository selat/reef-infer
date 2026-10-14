//! Builds `codegen::conv` programs for the corpus CONV_2D models it
//! supports and diffs them against the compiler's, field by field.
//!
//!     cargo run --example conv_check -- [filter] [--all-diffs]
//!
//! Corpus directory: `target/isa_corpus` or `REEF_ISA_CORPUS`. With the
//! models' weights exported (`tools/isa_corpus/conv_ref.py`) the parameter
//! blobs are compared too.

use std::path::PathBuf;

use reef_infer::codegen::Quant;
use reef_infer::codegen::conv::Conv;
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
    let max = if args.iter().any(|a| a == "--all-diffs") {
        usize::MAX
    } else {
        8
    };
    let dir = std::env::var("REEF_ISA_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/isa_corpus"))
        .join("conv");
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
        let get = |k: &str| p[k].as_u64().unwrap() as usize;
        let compiled: TfliteModel = load_tflite_model(
            dir.join(name)
                .join("model_edgetpu.tflite")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let e = &compiled.executables[0];
        let (cin, cout, k) = (get("Cin"), get("Cout"), get("k"));
        let r: Option<serde_json::Value> =
            std::fs::read_to_string(dir.join(name).join("conv_ref.json"))
                .ok()
                .map(|s| serde_json::from_str(&s).unwrap());
        let nums = |k: &str| -> Option<Vec<f64>> {
            r.as_ref().map(|r| {
                r[k].as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
        };
        let conv = Conv {
            cin,
            cout,
            k,
            stride: get("stride"),
            width: 16,
            height: 16,
            weights: nums("weights").map_or(vec![1; cout * k * k * cin], |v| {
                v.iter().map(|&x| x as i8).collect()
            }),
            w_scale: nums("w_scale")
                .map_or(vec![0.01; cout], |v| v.iter().map(|&x| x as f32).collect()),
            bias: nums("bias").map_or(vec![0; cout], |v| v.iter().map(|&x| x as i32).collect()),
            input: quant(&e.input_layers[0]),
            output: quant(&e.output_layers[0]),
            relu: p["act"] == "relu",
        };
        if conv.check().is_err() || (get("H"), get("W")) != (16, 16) || p["pad"] != "same" {
            continue;
        }
        total += 1;
        let ours = conv.build().unwrap();
        let mut report = Vec::new();
        for i in 0..2 {
            if let Some(d) = diff(&ours.executables[i], &compiled.executables[i], max) {
                report.push(format!(" {:?}:\n{d}", compiled.executables[i].exec_type));
            }
        }
        let lay =
            |m: &TfliteModel| format!("{:?}", m.executables[0].output_layers[0].output_layout);
        if lay(&ours) != lay(&compiled) {
            report.push(" output layout differs".into());
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
