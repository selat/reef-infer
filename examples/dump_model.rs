/// Dump the DarwiNN executable structure and TFLite tensor shapes of a .tflite model.
///
/// Usage: cargo run --example dump_model -- path/to/model.tflite ...
use reef_infer::model::load_tflite_model;
use reef_infer::schema_v3_generated::tflite::{TensorType, root_as_model};

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("Usage: cargo run --example dump_model -- <model.tflite> ...");
        std::process::exit(1);
    }
    for path in &paths {
        dump_model(path);
    }
}

fn dump_model(path: &str) {
    println!("\n{}", "=".repeat(72));
    println!("Model: {path}");
    println!("{}", "=".repeat(72));

    let model = match load_tflite_model(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ERROR: {e}");
            return;
        }
    };

    println!("  Executables: {}", model.executables.len());

    for (i, exec) in model.executables.iter().enumerate() {
        println!("\n  ── Executable [{i}] {:?}", exec.exec_type);
        println!("     scratch_size:  {} bytes", exec.scratch_size_bytes);
        println!("     parameters:   {} bytes", exec.parameters.len());
        println!("     instr chunks: {}", exec.instruction_chunks.len());

        // List every field offset across all chunks.
        for (ci, chunk) in exec.instruction_chunks.iter().enumerate() {
            println!("     chunk [{ci}] field_offsets: {}", chunk.field_offsets.len());
            for fo in &chunk.field_offsets {
                println!(
                    "       bit={:7}  {:?}  {:?}  name={:?}",
                    fo.offset_bit, fo.desc, fo.position, fo.name
                );
            }
        }

        println!("     input_layers:  {}", exec.input_layers.len());
        for (j, l) in exec.input_layers.iter().enumerate() {
            println!(
                "       [{j}] {:?}  {}B  zp={}  scale={}",
                l.name, l.size_bytes, l.zero_point, l.dequantization_factor
            );
        }

        println!("     output_layers: {}", exec.output_layers.len());
        for (j, l) in exec.output_layers.iter().enumerate() {
            println!(
                "       [{j}] {:?}  {}B  zp={}  scale={}",
                l.name, l.size_bytes, l.zero_point, l.dequantization_factor
            );
        }
    }

    // ── TFLite subgraph tensors ───────────────────────────────────────────
    let raw = std::fs::read(path).unwrap();
    if let Ok(tfl) = root_as_model(&raw) {
        if let Some(subgraphs) = tfl.subgraphs() {
            let sg = subgraphs.get(0);
            println!("\n  TFLite tensors ({} total):", sg.tensors().map(|t| t.len()).unwrap_or(0));
            if let Some(tensors) = sg.tensors() {
                for i in 0..tensors.len() {
                    let t = tensors.get(i);
                    let name  = t.name().unwrap_or("?");
                    let shape: Vec<i32> = t.shape()
                        .map(|s| (0..s.len()).map(|j| s.get(j)).collect())
                        .unwrap_or_default();
                    let dtype = match t.type_() {
                        TensorType::FLOAT32  => "f32",
                        TensorType::FLOAT16  => "f16",
                        TensorType::INT32    => "i32",
                        TensorType::UINT8    => "u8",
                        TensorType::INT64    => "i64",
                        TensorType::INT64    => "i64",
                        _                    => "?",
                    };
                    let q = t.quantization();
                    let (zp, scale) = q.and_then(|q| {
                        let zp = q.zero_point()?.get(0);
                        let sc = q.scale()?.get(0);
                        Some((zp, sc))
                    }).unwrap_or((0, 0.0));
                    println!(
                        "    [{i:3}] {name:55}  {dtype:5}  {shape:?}  zp={zp}  scale={scale:.8}"
                    );
                }
            }

            // Print subgraph inputs / outputs by tensor index.
            let in_idx: Vec<i32>  = sg.inputs().map(|v| (0..v.len()).map(|j| v.get(j)).collect()).unwrap_or_default();
            let out_idx: Vec<i32> = sg.outputs().map(|v| (0..v.len()).map(|j| v.get(j)).collect()).unwrap_or_default();
            println!("  Subgraph inputs:  {in_idx:?}");
            println!("  Subgraph outputs: {out_idx:?}");
        }
    }
}
