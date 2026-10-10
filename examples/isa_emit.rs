//! Prints a compiled bitstream as `codegen::Asm::emit` calls, the starting
//! point for transcribing a compiler program into a generator.
//!
//!     cargo run --example isa_emit -- model_edgetpu.tflite [executable]
//!
//! `start` / `end` (emitted by `Asm`) and `seq` (numbered by `Asm`) are
//! omitted; values of signed fields print as negative numbers.

use reef_infer::isa;
use reef_infer::model::load_tflite_model;

fn op_name(op: u8) -> String {
    match op {
        0x01 => "OP_TENSOR".into(),
        0x10 => "OP_RING_PRODUCER".into(),
        0x11 => "OP_RING_CONSUMER".into(),
        0x13 => "OP_NARROW_TO_WIDE".into(),
        0x14 => "OP_WIDE_TO_NARROW".into(),
        0x17 => "OP_MESH_N".into(),
        0x1a => "OP_SYNC".into(),
        0x20 => "OP_SCALAR".into(),
        0x21 => "OP_HALT".into(),
        0x24 => "OP_SCSYNC".into(),
        0x25 => "OP_POP".into(),
        0x26 => "OP_INFEED".into(),
        0x27 => "OP_OUTFEED".into(),
        o => format!("0x{o:02x}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m = load_tflite_model(&args[0]).expect("load model");
    let e: usize = args.get(1).map_or(0, |a| a.parse().unwrap());
    let chunk = &m.executables[e].instruction_chunks[0];
    let mut word = 0;
    for ins in isa::split(&chunk.bitstream).unwrap() {
        let d = ins.decode().unwrap();
        let op = ((ins.words[0] >> 6) & 0x3f) as u8;
        let n = ins.words.len();
        if matches!(d.layout.name, "start" | "end") {
            word += n;
            continue;
        }
        let fields: Vec<String> = d
            .named()
            .filter(|&(f, v)| v != 0 && !matches!(f, "opcode" | "seq" | "enable_scalar" | "branch"))
            .map(|(f, v)| {
                if v.unsigned_abs() > 9 {
                    format!(
                        "(\"{f}\", {}0x{:x})",
                        if v < 0 { "-" } else { "" },
                        v.unsigned_abs()
                    )
                } else {
                    format!("(\"{f}\", {v})")
                }
            })
            .chain(
                d.unexplained()
                    .map(|(f, v)| format!("(\"{}\", 0x{v:x})", f.name)),
            )
            .collect();
        println!("// {word}: {}", d.layout.name);
        println!("a.emit({}, &[{}])?;", op_name(op), fields.join(", "));
        word += n;
    }
}
