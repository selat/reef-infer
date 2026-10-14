//! Generated programs: framing, relocations and (when the ISA corpus from
//! `tools/isa_corpus/gen.py` is present) byte-for-byte equality with
//! `edgetpu_compiler` 16.0. The corpus directory defaults to
//! `target/isa_corpus`; override with `REEF_ISA_CORPUS`. Hardware checks
//! live in `tests/codegen_hw.rs` and `examples/codegen_hw.rs`.

use std::path::PathBuf;

use reef_infer::codegen::Quant;
use reef_infer::codegen::conv::Conv;
use reef_infer::codegen::elementwise::{self, Requant};
use reef_infer::codegen::fc::Fc;
use reef_infer::codegen::pool::MaxPool;
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

/// Corpus FC models covering each tiling case: one tile, K over 2 / 4 /
/// 13 / 16 input tiles (a partial last tile for 784 and 1000), N over 2 / 4
/// / 16 compute tiles, N padded to 16 or rounded to 8 B of output, and the
/// MNIST MLP's two layers. Instructions and output layout (they depend only
/// on the shape, the activation quantization and RELU).
#[test]
fn fc_matches_compiler() {
    for (name, k, n, relu) in [
        (
            "K64_N64_actnone_biasTrue_in_range4p0_iouint8",
            64,
            64,
            false,
        ),
        ("K8_N64_actnone_biasTrue_in_range4p0_ioint8", 8, 64, false),
        (
            "K128_N64_actnone_biasTrue_in_range4p0_ioint8",
            128,
            64,
            false,
        ),
        (
            "K256_N64_actnone_biasTrue_in_range4p0_ioint8",
            256,
            64,
            false,
        ),
        (
            "K1000_N64_actnone_biasTrue_in_range4p0_ioint8",
            1000,
            64,
            false,
        ),
        (
            "K64_N128_actnone_biasTrue_in_range4p0_ioint8",
            64,
            128,
            false,
        ),
        (
            "K64_N1024_actnone_biasTrue_in_range4p0_ioint8",
            64,
            1024,
            false,
        ),
        ("K64_N4_actnone_biasTrue_in_range4p0_ioint8", 64, 4, false),
        ("K64_N10_actnone_biasTrue_in_range4p0_ioint8", 64, 10, false),
        (
            "K784_N128_actrelu_biasTrue_in_range4p0_ioint8",
            784,
            128,
            true,
        ),
        (
            "K128_N10_actnone_biasTrue_in_range4p0_ioint8",
            128,
            10,
            false,
        ),
    ] {
        let Some(path) = corpus(&format!("fc/fc_{name}")) else {
            eprintln!("ISA corpus model fc_{name} not found; skipping");
            continue;
        };
        let compiled = load_tflite_model(&path).unwrap();
        let e = &compiled.executables[0];
        let mut f = fc(k, n, quant(&e.input_layers[0]), quant(&e.output_layers[0]));
        f.relu = relu;
        let ours = f.build().unwrap();
        for i in 0..2 {
            assert_eq!(
                ours.executables[i].exec_type,
                compiled.executables[i].exec_type
            );
            assert_eq!(
                ours.executables[i].instruction_chunks[0].bitstream,
                compiled.executables[i].instruction_chunks[0].bitstream,
                "{name}: executable {i}"
            );
        }
        let (a, b) = (&ours.executables[0].output_layers[0], &e.output_layers[0]);
        assert_eq!(a.size_bytes, b.size_bytes, "{name}");
        assert_eq!(
            format!("{:?}", a.output_layout),
            format!("{:?}", b.output_layout),
            "{name}"
        );
    }
}

/// Every corpus CONV_2D model `codegen::conv` supports: instructions,
/// relocations and output layout, and the parameter blob when the weights
/// are exported (`tools/isa_corpus/conv_ref.py`).
#[test]
fn conv_matches_compiler() {
    let models = [
        (
            "k1_padsame_stride1",
            "Cin16_Cout32",
            "actnone_in_range4p0",
            1,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout16",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout8",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout32",
            "actnone_in_range0p5",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout32",
            "actnone_in_range16p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout32",
            "actrelu_in_range4p0",
            3,
            true,
        ),
        (
            "k3_padsame_stride1",
            "Cin16_Cout32",
            "actrelu6_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin32_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin64_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin128_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin1_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin3_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin4_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
        (
            "k1_padsame_stride2",
            "Cin16_Cout32",
            "actnone_in_range4p0",
            1,
            false,
        ),
        (
            "k3_padsame_stride2",
            "Cin16_Cout32",
            "actnone_in_range4p0",
            3,
            false,
        ),
    ];
    let mut checked = 0;
    for (kind, ch, act, k, relu) in models {
        let Some(path) = corpus(&format!("conv/conv_{ch}_H16_W16_{act}_{kind}")) else {
            continue;
        };
        let compiled = load_tflite_model(&path).unwrap();
        let e = &compiled.executables[0];
        let (cin, cout) = (
            e.input_layers[0].z_dim as usize,
            e.output_layers[0].z_dim as usize,
        );
        // weights from tools/isa_corpus/conv_ref.py, if exported
        let r: Option<serde_json::Value> =
            std::fs::read_to_string(PathBuf::from(&path).with_file_name("conv_ref.json"))
                .ok()
                .map(|s| serde_json::from_str(&s).unwrap());
        let nums = |key: &str| -> Option<Vec<f64>> {
            r.as_ref().map(|r| {
                r[key]
                    .as_array()
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
            stride: if kind.ends_with("stride2") { 2 } else { 1 },
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
            relu,
        };
        let ours = conv.build().unwrap();
        for i in 0..2 {
            let (a, b) = (&ours.executables[i], &compiled.executables[i]);
            assert_eq!(a.exec_type, b.exec_type);
            assert_eq!(
                a.instruction_chunks[0].bitstream, b.instruction_chunks[0].bitstream,
                "{path}: executable {i}"
            );
            let offs = |e: &ExecInfo| -> Vec<i32> {
                e.instruction_chunks[0]
                    .field_offsets
                    .iter()
                    .map(|f| f.offset_bit)
                    .collect()
            };
            assert_eq!(offs(a), offs(b), "{path}: relocations, executable {i}");
        }
        if r.is_some() {
            assert_eq!(
                ours.executables[1].parameters, compiled.executables[1].parameters,
                "{path}: parameters"
            );
        }
        assert_eq!(
            format!("{:?}", ours.executables[0].output_layers[0].output_layout),
            format!("{:?}", e.output_layers[0].output_layout),
            "{path}: output layout"
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("ISA corpus not found; skipping");
    }
}

/// The MNIST MLP from `tools/mnist/train.py` (skipped when absent): the
/// front end maps it to two FC layers, both build, and the host reference
/// (the TPU's numerics) picks the same class as TFLite's CPU interpreter.
#[test]
fn mnist_graph_matches_tflite() {
    use reef_infer::graph::{Graph, Layer, u8_to_i8};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/mnist");
    let Ok(images) = std::fs::read(dir.join("test_images.bin")) else {
        eprintln!("target/mnist not found (tools/mnist/train.py); skipping");
        return;
    };
    let g = Graph::from_tflite(dir.join("mlp.tflite").to_str().unwrap()).unwrap();
    let shapes: Vec<_> = g
        .layers
        .iter()
        .map(|l| match l {
            Layer::Fc(f) => (f.k, f.n, f.relu),
        })
        .collect();
    assert_eq!(shapes, [(784, 128, true), (128, 10, false)]);
    assert_eq!(g.build().unwrap().len(), 2);
    // input scale 1/255, zero point -128: the uint8 input is the pixel
    assert_eq!((g.input.scale, g.input.zero_point), (1.0 / 255.0, 0));
    let tflite = std::fs::read(dir.join("tflite_logits.bin")).unwrap();
    let argmax = |v: &[i8]| {
        (0..v.len())
            .max_by_key(|&i| (v[i], usize::MAX - i))
            .unwrap()
    };
    for i in 0..500 {
        let y = u8_to_i8(&g.reference(&images[784 * i..784 * (i + 1)]));
        let t: Vec<i8> = tflite[10 * i..10 * (i + 1)]
            .iter()
            .map(|&b| b as i8)
            .collect();
        assert_eq!(argmax(&y), argmax(&t), "image {i}");
        assert!(
            y.iter().zip(&t).all(|(a, b)| (a - b).abs() <= 1),
            "image {i}"
        );
    }
}

/// Every corpus 2x2 max pool `codegen::pool` supports, including the MNIST
/// CNN's (32x32x32, 16x16x64): instructions and output layout.
#[test]
fn pool_matches_compiler() {
    for (h, w, c) in [
        (16, 16, 32),
        (16, 16, 64),
        (16, 16, 128),
        (32, 16, 32),
        (32, 32, 32),
    ] {
        let name = format!("pool/pool_C{c}_H{h}_W{w}_opmax_padvalid_size2_stride2");
        let Some(path) = corpus(&name) else {
            eprintln!("ISA corpus model {name} not found; skipping");
            continue;
        };
        let compiled = load_tflite_model(&path).unwrap();
        let e = &compiled.executables[0];
        let ours = MaxPool {
            height: h,
            width: w,
            c,
            quant: quant(&e.input_layers[0]),
        }
        .build()
        .unwrap()
        .into_model();
        let o = &ours.executables[0];
        assert_eq!(
            o.instruction_chunks[0].bitstream, e.instruction_chunks[0].bitstream,
            "{name}"
        );
        assert_eq!(
            format!("{:?}", o.output_layers[0].output_layout),
            format!("{:?}", e.output_layers[0].output_layout),
            "{name}"
        );
    }
}
