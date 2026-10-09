//! Decodes every instruction bitstream in compiled Edge TPU models and checks
//! that re-encoding reproduces it byte for byte.
//!
//!     cargo run --example isa_check -- [--disasm] [--dump DIR] model_edgetpu.tflite...
//!
//! `--dump DIR` writes each bitstream (`<n>.bin`) and its decoded fields
//! (`<n>.json`) for cross-checking against another decoder.

use std::collections::BTreeMap;
use std::process::ExitCode;

use reef_infer::isa;
use reef_infer::model::load_edgetpu_ops;

#[derive(Default)]
struct Stats {
    instrs: usize,
    words: usize,
    by_opcode: BTreeMap<u8, usize>,
    unknown: BTreeMap<u8, usize>,
    /// (layout, reserved field) -> instructions where it is nonzero
    unexplained: BTreeMap<(String, String), usize>,
    failures: Vec<String>,
}

fn check_chunk(
    bytes: &[u8],
    label: &str,
    st: &mut Stats,
    disasm: bool,
) -> Option<serde_json::Value> {
    let instrs = match isa::split(bytes) {
        Ok(v) => v,
        Err(e) => {
            st.failures.push(format!("{label}: split: {e}"));
            return None;
        }
    };
    let mut out = Vec::new();
    let mut reencoded = Vec::new();
    for ins in &instrs {
        st.instrs += 1;
        st.words += ins.words.len();
        *st.by_opcode.entry(ins.opcode()).or_default() += 1;
        match ins.decode() {
            Ok(d) => {
                for (f, _) in d.unexplained() {
                    *st.unexplained
                        .entry((d.layout.name.into(), f.name.clone()))
                        .or_default() += 1;
                }
                let fields: serde_json::Map<_, _> = d
                    .named()
                    .map(|(n, v)| (n.to_string(), serde_json::json!(v)))
                    .collect();
                let rsv_bits: u32 = d.unexplained().map(|(_, v)| (v as u64).count_ones()).sum();
                let rsv: serde_json::Map<_, _> = d
                    .unexplained()
                    .map(|(f, v)| (f.name.clone(), serde_json::json!([f.lo, f.width, v])))
                    .collect();
                out.push(serde_json::json!({
                    "word_index": ins.word_index, "opcode": ins.opcode(), "layout": d.layout.name,
                    "fields": fields, "rsv_bits": rsv_bits, "rsv": rsv,
                }));
                match d.encode() {
                    Ok(w) => reencoded.extend(w),
                    Err(e) => st
                        .failures
                        .push(format!("{label}@{}: encode: {e}", ins.word_index)),
                }
            }
            Err(_) => {
                *st.unknown.entry(ins.opcode()).or_default() += 1;
                out.push(serde_json::json!({"word_index": ins.word_index, "opcode": ins.opcode(), "layout": null}));
                reencoded.extend(&ins.words);
            }
        }
    }
    if isa::assemble(reencoded.chunks(1)) != bytes {
        st.failures
            .push(format!("{label}: re-encoded bitstream differs"));
    }
    // Structure: start ... end, with start.length_bits = 128 * words in between.
    let first = instrs.first().and_then(|i| i.decode().ok());
    let last = instrs.last().map(|i| i.opcode());
    match (first, last) {
        (Some(s), Some(0x3f)) if s.layout.name == "start" => {
            let between = bytes.len() / 16 - 2;
            if s.get("length_bits") != Some(128 * between as i64) {
                st.failures.push(format!(
                    "{label}: start.length_bits={:?}, expected {}",
                    s.get("length_bits"),
                    128 * between
                ));
            }
        }
        _ => st
            .failures
            .push(format!("{label}: does not run from start to end")),
    }
    if disasm {
        println!("== {label}");
        print!("{}", isa::disassemble(bytes).unwrap());
    }
    Some(serde_json::Value::Array(out))
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let disasm = args.iter().any(|a| a == "--disasm");
    args.retain(|a| a != "--disasm");
    let dump = args.iter().position(|a| a == "--dump").map(|i| {
        let d = args.remove(i + 1);
        args.remove(i);
        std::fs::create_dir_all(&d).expect("create dump dir");
        d
    });
    let mut st = Stats::default();
    let (mut chunks, mut skipped) = (0, Vec::new());
    for path in &args {
        let ops = match load_edgetpu_ops(path) {
            Ok(m) => m,
            Err(e) => {
                skipped.push(format!("{path}: {e}"));
                continue;
            }
        };
        let execs = ops.iter().enumerate().flat_map(|(oi, m)| {
            m.executables
                .iter()
                .enumerate()
                .map(move |(ei, ex)| (oi, ei, ex))
        });
        for (oi, ei, ex) in execs {
            for (ci, ch) in ex.instruction_chunks.iter().enumerate() {
                let label = format!("{path} op{oi} exec{ei} ({:?}) chunk{ci}", ex.exec_type);
                if let Some(json) = check_chunk(&ch.bitstream, &label, &mut st, disasm) {
                    if let Some(d) = &dump {
                        std::fs::write(format!("{d}/{chunks}.bin"), &ch.bitstream).unwrap();
                        let doc = serde_json::json!({
                            "label": label, "path": path, "op": oi, "exec": ei,
                            "exec_type": format!("{:?}", ex.exec_type), "chunk": ci, "instrs": json,
                        });
                        std::fs::write(format!("{d}/{chunks}.json"), doc.to_string()).unwrap();
                    }
                }
                chunks += 1;
            }
        }
    }
    println!(
        "models: {}  skipped: {}  bitstreams: {chunks}",
        args.len(),
        skipped.len()
    );
    println!("instructions: {}  words: {}", st.instrs, st.words);
    println!("opcode counts:");
    for (op, n) in &st.by_opcode {
        let name = isa::layout(*op).map_or("??", |l| l.name);
        println!("  {op:#04x} {name:14} {n}");
    }
    let unknown_words: usize = st
        .unknown
        .iter()
        .map(|(op, n)| isa::instr_len(*op) * n)
        .sum();
    println!(
        "undecoded opcodes: {:?} ({unknown_words} words)",
        st.unknown
    );
    println!("nonzero reserved fields (layout.field: instructions):");
    for ((l, f), n) in &st.unexplained {
        println!("  {l}.{f}: {n}");
    }
    for s in &skipped {
        println!("skipped {s}");
    }
    for f in &st.failures {
        println!("FAIL {f}");
    }
    println!(
        "{}",
        if st.failures.is_empty() {
            "OK"
        } else {
            "FAILED"
        }
    );
    if st.failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
