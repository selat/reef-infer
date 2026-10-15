//! Generated programs: framing, relocations and (when the ISA corpus from
//! `tools/isa_corpus/gen.py` is present) byte-for-byte equality with
//! `edgetpu_compiler` 16.0. The corpus directory defaults to
//! `target/isa_corpus`; override with `REEF_ISA_CORPUS`. Hardware checks
//! live in `tests/codegen_hw.rs` and `examples/codegen_hw.rs`.

use std::path::PathBuf;

use reef_infer::codegen::Quant;
use reef_infer::codegen::avg_pool::GlobalAvgPool;
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
        (
            "K2048_N64_actnone_biasTrue_in_range4p0_ioint8",
            2048,
            64,
            false,
        ),
        (
            "K3136_N10_actnone_biasTrue_in_range4p0_ioint8",
            3136,
            10,
            false,
        ),
        (
            "K4096_N64_actnone_biasTrue_in_range4p0_ioint8",
            4096,
            64,
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
        // the MNIST CNN's second conv: 1 KiB of output per tile
        (
            "k3_padsame_stride1",
            "Cin32_Cout64",
            "actrelu_in_range4p0",
            3,
            true,
        ),
        // ResNet-8: stride-2 downsampling (3x3 and 1x1 shortcut), 64 -> 64
        (
            "k3_padsame_stride2",
            "Cin32_Cout64",
            "actrelu_in_range4p0",
            3,
            true,
        ),
        (
            "k1_padsame_stride2",
            "Cin32_Cout64",
            "actnone_in_range4p0",
            1,
            false,
        ),
        (
            "k3_padsame_stride1",
            "Cin64_Cout64",
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
        .steps
        .iter()
        .map(|s| match &s.layer {
            Layer::Fc(f) => (f.k, f.n, f.relu),
            l => panic!("unexpected {l:?}"),
        })
        .collect();
    assert_eq!(shapes, [(784, 128, true), (128, 10, false)]);
    assert_eq!(g.build().unwrap().calls(), 2);
    // input scale 1/255, zero point -128: the uint8 input is the pixel
    assert_eq!((g.input.scale, g.input.zero_point), (1.0 / 255.0, 0));
    let tflite = std::fs::read(dir.join("mlp_tflite_logits.bin")).unwrap();
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

/// The MNIST CNN from `tools/mnist/train.py --model cnn` (skipped when
/// absent): conv / pool / conv / pool / FC with the host padding glue, and
/// the host reference against TFLite's CPU interpreter.
#[test]
fn mnist_cnn_graph_matches_tflite() {
    use reef_infer::graph::{Graph, Layer, u8_to_i8};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/mnist");
    let (Ok(images), Ok(tflite)) = (
        std::fs::read(dir.join("test_images.bin")),
        std::fs::read(dir.join("cnn_tflite_logits.bin")),
    ) else {
        eprintln!("target/mnist CNN not found (tools/mnist/train.py --model cnn); skipping");
        return;
    };
    let g = Graph::from_tflite(dir.join("cnn.tflite").to_str().unwrap()).unwrap();
    let kinds: Vec<_> = g
        .steps
        .iter()
        .map(|s| match &s.layer {
            Layer::Conv(c) => format!("conv{}x{}:{}", c.height, c.width, c.cin),
            Layer::MaxPool(p) => format!("pool{}x{}", p.height, p.width),
            Layer::Fc(f) => format!("fc{}", f.k),
            l => panic!("unexpected {l:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "conv32x32:16",
            "pool32x32",
            "conv16x16:32",
            "pool16x16",
            "fc3136"
        ]
    );
    assert_eq!(g.build().unwrap().calls(), 5);
    for i in 0..100 {
        let y = u8_to_i8(&g.reference(&images[784 * i..784 * (i + 1)]));
        let t: Vec<i8> = tflite[10 * i..10 * (i + 1)]
            .iter()
            .map(|&b| b as i8)
            .collect();
        assert!(
            y.iter()
                .zip(&t)
                .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1),
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

/// The corpus global average pool (ResNet-8's head, 8x8x64) in the
/// compiler's form (centre = zero point, mult 1 / 64): instructions and
/// output layout.
#[test]
fn avg_pool_matches_compiler() {
    let name = "pool/pool_C64_H8_W8_opavg_padvalid_size8_stride8";
    let Some(path) = corpus(name) else {
        eprintln!("ISA corpus model {name} not found; skipping");
        return;
    };
    let compiled = load_tflite_model(&path).unwrap();
    let e = &compiled.executables[0];
    let q = quant(&e.input_layers[0]);
    let ours = GlobalAvgPool {
        height: 8,
        width: 8,
        c: 64,
        center: q.zero_point,
        mult: 1.0 / 64.0,
        quant: q,
    }
    .build()
    .unwrap()
    .into_model();
    let o = &ours.executables[0];
    assert_eq!(
        o.instruction_chunks[0].bitstream,
        e.instruction_chunks[0].bitstream
    );
    assert!(o.parameters.is_empty() && e.parameters.is_empty());
    assert_eq!(
        format!("{:?}", o.output_layers[0].output_layout),
        format!("{:?}", e.output_layers[0].output_layout)
    );
}

/// `GlobalAvgPool::tflite` rounds every channel sum as TFLite's integer
/// AVERAGE_POOL_2D does (sum / 64, half away from zero), ties included,
/// where the compiler's mult 1 / 64 rounds them to even.
#[test]
fn avg_pool_rounds_as_tflite() {
    let q = Quant {
        scale: 0.03,
        zero_point: 127,
    };
    for int8 in [true, false] {
        let p = GlobalAvgPool::tflite(8, 8, 64, q, int8).unwrap();
        let center = if int8 { 128 } else { 0 };
        let tflite = |x: &[u8]| -> Vec<u8> {
            (0..64)
                .map(|z| {
                    let s: i32 = (0..64).map(|i| x[64 * i + z] as i32 - center).sum();
                    let r = if s > 0 { (s + 32) / 64 } else { (s - 32) / 64 };
                    (r + center) as u8
                })
                .collect()
        };
        let mut seed = 7u64;
        let mut inputs: Vec<Vec<u8>> = (0..256u32)
            .map(|v| {
                // channel z: 32 pixels at v, 32 at v + z (ties at odd z / 2)
                (0..4096)
                    .map(|i: u32| (v + (i / 64 % 2) * (i % 64)).min(255) as u8)
                    .collect()
            })
            .collect();
        inputs.extend((0..64).map(|_| {
            (0..4096)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 32) as u8
                })
                .collect()
        }));
        let compiler_form = GlobalAvgPool {
            center,
            mult: 1.0 / 64.0,
            ..p.clone()
        };
        let mut compiler_differs = 0;
        for x in &inputs {
            assert_eq!(p.reference(x), tflite(x), "int8 {int8}");
            compiler_differs += (compiler_form.reference(x) != tflite(x)) as usize;
        }
        assert!(compiler_differs > 0, "the ties should tell the two apart");
    }
}

/// Every corpus element-wise ADD `codegen::add` supports (shapes 16x16x32 /
/// 16x16x64 / 32x32x16 / 32x32x64, scale ratios 1/4 .. 4, RELU):
/// instructions (weights, multiplier, clamps included) and output layout.
#[test]
fn add_matches_compiler() {
    use reef_infer::codegen::add::Add;
    for (shape, act, b_range) in [
        ("16x16x32", "none", "4p0"),
        ("16x16x32", "none", "1p0"),
        ("16x16x32", "none", "2p0"),
        ("16x16x32", "none", "16p0"),
        ("32x32x64", "none", "4p0"),
        ("16x16x32", "relu", "4p0"),
        ("16x16x64", "relu", "4p0"),
        ("32x32x16", "relu", "4p0"),
    ] {
        let name =
            format!("eltwise/eltwise_a_range4p0_act{act}_b_range{b_range}_opadd_shape{shape}");
        let Some(path) = corpus(&name) else {
            eprintln!("ISA corpus model {name} not found; skipping");
            continue;
        };
        let compiled = load_tflite_model(&path).unwrap();
        let e = &compiled.executables[0];
        let dims: Vec<usize> = shape.split('x').map(|d| d.parse().unwrap()).collect();
        let ours = Add {
            height: dims[0],
            width: dims[1],
            c: dims[2],
            a: quant(&e.input_layers[0]),
            b: quant(&e.input_layers[1]),
            output: quant(&e.output_layers[0]),
            relu: act == "relu",
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

/// ADD's integer weights: the best fraction of the scales with denominator
/// <= 32767, as the compiler picks them (corpus 1D / 3-D models).
#[test]
fn add_weights_follow_the_compiler() {
    use reef_infer::codegen::Quant;
    use reef_infer::codegen::add::Add;
    let q = |scale| Quant {
        scale,
        zero_point: 128,
    };
    for (sa, sb, want) in [
        (0.03136582, 0.031355884, (3157, 3156)),
        (0.03136884, 0.031368285, (32767, 32766)),
        (0.031372506, 0.031372417, (1, 1)),
        (0.03137238, 0.007843103, (4, 1)),
        (0.03137244, 0.1254892, (1, 4)),
        (0.03137235, 0.01568624, (2, 1)),
    ] {
        let add = Add {
            height: 16,
            width: 16,
            c: 32,
            a: q(sa),
            b: q(sb),
            output: q(0.06),
            relu: false,
        };
        assert_eq!(add.weights(), want, "{sa} / {sb}");
    }
}

/// ResNet-8 from `tools/cifar10/train.py` (skipped when absent): the DAG
/// front end (residual ADDs read tensors from earlier steps), and the host
/// reference against TFLite's CPU interpreter.
#[test]
fn resnet8_graph_matches_tflite() {
    use reef_infer::graph::{Graph, Layer, u8_to_i8};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cifar10");
    let (Ok(images), Ok(tflite)) = (
        std::fs::read(dir.join("test_images.bin")),
        std::fs::read(dir.join("resnet8_tflite_logits.bin")),
    ) else {
        eprintln!("target/cifar10 not found (tools/cifar10/train.py); skipping");
        return;
    };
    let g = Graph::from_tflite(dir.join("resnet8.tflite").to_str().unwrap()).unwrap();
    let count = |f: fn(&Layer) -> bool| g.steps.iter().filter(|s| f(&s.layer)).count();
    assert_eq!(count(|l| matches!(l, Layer::Conv(_))), 9);
    assert_eq!(count(|l| matches!(l, Layer::Add(_))), 3);
    assert_eq!(count(|l| matches!(l, Layer::Fc(_))), 1); // dense
    assert_eq!(count(|l| matches!(l, Layer::AvgPool(_))), 1);
    // each ADD reads a tensor produced before the step right before it
    // (the shortcut)
    let made_at = |t: i32| g.steps.iter().position(|s| s.output == t);
    for (k, s) in g.steps.iter().enumerate() {
        if matches!(s.layer, Layer::Add(_)) {
            let earliest = s.inputs.iter().filter_map(|i| made_at(i.tensor)).min();
            assert!(earliest.is_some_and(|p| p + 1 < k), "step {k}");
        }
    }
    // the whole graph (9 convs, 3 adds, stage 3 at 8x8 natively, the pool
    // and the FC on its output) runs on chip in one call
    let p = g.build().unwrap();
    assert_eq!(p.steps.len(), 14);
    assert_eq!(g.chains().unwrap(), vec![0..14]);
    assert_eq!(p.calls(), 1);
    let native = |k: usize| g.steps[k].on_chip.is_some();
    assert!((0..14).filter(|&k| native(k)).eq([9, 11]));
    // every layer's parameters at its own wide range, back to back
    let bases = g.wide_bases().unwrap();
    let end = bases[13] + g.steps[13].layer.wide_units();
    assert_eq!(end, 0x7a8);
    let px = g.input_len();
    for i in 0..100 {
        let y = u8_to_i8(&g.reference(&images[px * i..px * (i + 1)]));
        let t: Vec<i8> = tflite[10 * i..10 * (i + 1)]
            .iter()
            .map(|&b| b as i8)
            .collect();
        assert!(
            y.iter()
                .zip(&t)
                .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1),
            "image {i}"
        );
    }
}

/// `chain::splice` joins programs into one: a single start / end frame,
/// tile sequence numbers strictly increasing across the parts, and every
/// relocation still on a `mov` immediate.
#[test]
fn splice_keeps_programs_well_formed() {
    use reef_infer::codegen::chain::splice;
    use reef_infer::codegen::conv::Io;
    let conv = |cin: usize, cout: usize| Conv {
        cin,
        cout,
        k: 3,
        stride: 1,
        width: 16,
        height: 16,
        weights: vec![1; cout * 9 * cin],
        w_scale: vec![0.001; cout],
        bias: vec![0; cout],
        input: Quant {
            scale: 0.03,
            zero_point: 128,
        },
        output: Quant {
            scale: 0.2,
            zero_point: 128,
        },
        relu: false,
    };
    let (a, b) = (conv(16, 32), conv(32, 16));
    let pa = a
        .chain_program(
            0,
            Io {
                output_to: Some((0x1000, b.input_patch())),
                ..Io::default()
            },
        )
        .unwrap();
    let pb = b
        .chain_program(
            a.wide_units(),
            Io {
                input_at: Some((0x1000, b.input_patch())),
                ..Io::default()
            },
        )
        .unwrap();
    let joined = splice(&[pa, pb]).unwrap();
    let ins = isa::split(&joined.bitstream).unwrap();
    let ops: Vec<u8> = ins.iter().map(|i| i.opcode()).collect();
    assert_eq!(ops.iter().filter(|&&o| o == 0x3e).count(), 1); // start
    assert_eq!(ops.iter().filter(|&&o| o == 0x3f).count(), 1); // end
    let seqs: Vec<i64> = ins
        .iter()
        .filter_map(|i| i.decode().unwrap().get("seq"))
        .collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "sequence numbers");
    let words = isa::words(&joined.bitstream).unwrap();
    for r in &joined.field_offsets {
        let w = r.offset_bit as usize / 128;
        let d = isa::layout(isa::opcode(words[w]))
            .unwrap()
            .decode(&words[w..w + 1])
            .unwrap();
        assert_eq!(
            d.get("s_op"),
            Some(0x2f),
            "relocation at bit {}",
            r.offset_bit
        );
    }
    // one input relocation pair (first part) and one output pair (last)
    use reef_infer::executable_generated::platforms::darwinn::Description as D;
    let count = |d: D| joined.field_offsets.iter().filter(|r| r.desc == d).count();
    assert_eq!(count(D::BASE_ADDRESS_INPUT_ACTIVATION), 2);
    assert_eq!(count(D::BASE_ADDRESS_OUTPUT_ACTIVATION), 2);
}

/// `chain::plan` places the patches above every layer's own narrow
/// memory, reuses a patch's memory once it is dead, and `chain::convs`
/// refuses chains whose output does not fill the next patch (16x16 stride
/// 2 -> an 8x8 image the generators do not lay out).
#[test]
fn conv_chain_planning() {
    use reef_infer::codegen::chain::{Node, Op, convs, plan};
    let conv = |k: usize, stride: usize, hw: usize, cin: usize, cout: usize| Conv {
        cin,
        cout,
        k,
        stride,
        width: hw,
        height: hw,
        weights: vec![1; cout * k * k * cin],
        w_scale: vec![0.001; cout],
        bias: vec![0; cout],
        input: Quant {
            scale: 0.03,
            zero_point: 128,
        },
        output: Quant {
            scale: 0.2,
            zero_point: 128,
        },
        relu: false,
    };
    let (a, b, c, d) = (
        conv(3, 1, 32, 16, 32),
        conv(3, 1, 32, 32, 32),
        conv(3, 1, 32, 32, 32),
        conv(3, 2, 32, 32, 32),
    );
    let layers = [&a, &b, &c, &d];
    let nodes: Vec<Node> = (0..4)
        .map(|k| Node {
            op: Op::Conv(layers[k], 0),
            inputs: vec![k],
        })
        .collect();
    let p = plan(&nodes).unwrap();
    let at: Vec<(i64, _)> = (1..4).map(|t| p.at[t].unwrap()).collect();
    for l in layers {
        assert!(at.iter().all(|&(x, _)| x >= l.narrow_end()));
    }
    // 32x32 Cin 32: the layer's own buffers end past 0x1000
    assert!(b.narrow_end() > 0x1000);
    // each tensor is the next layer's patch; 1 and 2 are live together,
    // 3 takes 1's place
    for t in 0..3 {
        assert_eq!(at[t].1, layers[t + 1].input_patch());
    }
    assert!(at[1].0 >= at[0].0 + at[0].1.tp());
    assert_eq!(at[2].0, at[0].0);
    assert!(p.at[0].is_none() && p.at[4].is_none());
    assert!(convs(&[(&a, 0), (&b, 0x400), (&d, 0x800)]).is_ok());
    // 32x32 stride 1 -> 16x16: 8x8 pixels per tile do not fill a 4x4 patch
    let d16 = conv(3, 1, 16, 32, 32);
    assert!(convs(&[(&b, 0), (&d16, 0x400)]).is_err());
    // 16x16 stride 2 -> 8x8: not laid out on the 4x4 grid
    let (e, f) = (conv(3, 2, 16, 32, 32), conv(3, 1, 16, 32, 32));
    assert!(convs(&[(&e, 0), (&f, 0x400)]).is_err());
}
