//! Generated programs: framing, relocations and (when the ISA corpus from
//! `tools/isa_corpus/gen.py` is present) byte-for-byte equality with
//! `edgetpu_compiler` 16.0. The corpus directory defaults to
//! `target/isa_corpus`; override with `REEF_ISA_CORPUS`. Hardware checks
//! live in `tests/codegen_hw.rs` and `examples/codegen_hw.rs`.

use std::path::PathBuf;

use reef_infer::codegen::Quant;
use reef_infer::codegen::elementwise::{self, Requant};
use reef_infer::codegen::fc::Fc;
use reef_infer::isa;
use reef_infer::model::{ExecInfo, LayerInfo, load_tflite_model};

fn corpus(rel: &str) -> Option<String> {
    let dir = std::env::var("REEF_ISA_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/isa_corpus"));
    let p = dir.join(rel).join("model_edgetpu.tflite");
    p.exists().then(|| p.to_string_lossy().into_owned())
}

fn quant(l: &LayerInfo) -> Quant {
    Quant {
        scale: l.dequantization_factor,
        zero_point: l.zero_point,
    }
}

fn fc(k: usize, n: usize, input: Quant, output: Quant) -> Fc {
    Fc {
        k,
        n,
        weights: (0..n * k).map(|i| (i % 251) as i8).collect(),
        w_scale: vec![0.01; n],
        bias: vec![0; n],
        input,
        output,
        relu: false,
    }
}

/// Every instruction decodes, `start` / `end` frame the stream, and each
/// relocation points at a `scalar.imm_scalar` field.
fn check_framing(e: &ExecInfo) {
    let chunk = &e.instruction_chunks[0];
    let instrs = isa::split(&chunk.bitstream).unwrap();
    let first = instrs[0].decode().unwrap();
    let last = instrs.last().unwrap().decode().unwrap();
    assert_eq!(first.layout.name, "start");
    assert_eq!(last.layout.name, "end");
    let words = chunk.bitstream.len() / 16;
    assert_eq!(
        first.get("length_bits").unwrap() as usize,
        (words - 2) * 128
    );
    let mut starts = Vec::new();
    let mut w = 0;
    for ins in &instrs {
        let d = ins.decode().unwrap();
        assert_eq!(d.encode().unwrap(), ins.words, "{d}");
        starts.push((w, d.layout.name));
        w += ins.words.len();
    }
    for r in &chunk.field_offsets {
        let word = r.offset_bit as usize / 128;
        assert_eq!(
            r.offset_bit as usize % 128,
            70,
            "relocation not at imm_scalar"
        );
        assert!(
            starts.contains(&(word, "scalar")),
            "relocation at word {word} is not a scalar op"
        );
    }
}

#[test]
fn generated_programs_are_well_formed() {
    let q = Quant {
        scale: 0.05,
        zero_point: 128,
    };
    let p = elementwise::build(&Requant::from_quant(q, q, true), q, q, true)
        .unwrap()
        .into_model();
    check_framing(&p.executables[0]);
    for (k, n) in [(8, 16), (24, 48), (64, 64)] {
        let m = fc(k, n, q, q).build().unwrap();
        check_framing(&m.executables[0]);
        check_framing(&m.executables[1]);
        assert_eq!(m.executables[1].parameters.len(), n * (12 + k));
    }
    assert!(fc(12, 64, q, q).build().is_err());
    assert!(fc(64, 80, q, q).build().is_err());
}

#[test]
fn fc_parameter_layout() {
    let q = Quant {
        scale: 0.05,
        zero_point: 128,
    };
    let mut f = fc(8, 16, q, q);
    f.weights[3 * 8 + 5] = -7; // W[n = 3][k = 5]
    f.bias[2] = -42;
    let p = f.parameters();
    let n = 16;
    assert_eq!(f32::from_le_bytes(p[0..4].try_into().unwrap()), f.mult(0));
    assert_eq!(
        i32::from_le_bytes(p[4 * n + 8..4 * n + 12].try_into().unwrap()),
        -42
    );
    assert_eq!(
        i32::from_le_bytes(p[8 * n..8 * n + 4].try_into().unwrap()),
        128
    );
    // [K/4][N][4]: k = 5 is group 1, lane 1
    assert_eq!(p[12 * n + (n + 3) * 4 + 1], (-7i32 + 128) as u8);
}

#[test]
fn relu_matches_compiler() {
    let Some(path) = corpus("act/act_in_range8p0_oprelu_shape16x16x32") else {
        eprintln!("ISA corpus not found; skipping");
        return;
    };
    let compiled = load_tflite_model(&path).unwrap();
    let e = &compiled.executables[0];
    let (qi, qo) = (quant(&e.input_layers[0]), quant(&e.output_layers[0]));
    let ours = elementwise::build(&Requant::from_quant(qi, qo, true), qi, qo, true).unwrap();
    assert_eq!(ours.chunk.bitstream, e.instruction_chunks[0].bitstream);
    let offsets = |c: &reef_infer::model::ChunkInfo| -> Vec<i32> {
        c.field_offsets.iter().map(|f| f.offset_bit).collect()
    };
    assert_eq!(offsets(&ours.chunk), offsets(&e.instruction_chunks[0]));
    assert_eq!(
        format!("{:?}", ours.output.output_layout),
        format!("{:?}", e.output_layers[0].output_layout)
    );
}

#[test]
fn fc_matches_compiler() {
    let Some(path) = corpus("fc/fc_K64_N64_actnone_biasTrue_in_range4p0_iouint8") else {
        eprintln!("ISA corpus not found; skipping");
        return;
    };
    let compiled = load_tflite_model(&path).unwrap();
    let e = &compiled.executables[0];
    // The instructions depend only on the shape and the activation quantization.
    let ours = fc(
        64,
        64,
        quant(&e.input_layers[0]),
        quant(&e.output_layers[0]),
    )
    .build()
    .unwrap();
    for i in 0..2 {
        assert_eq!(
            ours.executables[i].exec_type,
            compiled.executables[i].exec_type
        );
        assert_eq!(
            ours.executables[i].instruction_chunks[0].bitstream,
            compiled.executables[i].instruction_chunks[0].bitstream,
            "executable {i}"
        );
    }
}
