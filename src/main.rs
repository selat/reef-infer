use std::sync::Arc;

use reef_infer::runner::Device;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ── CLI ───────────────────────────────────────────────────────────────
    // Usage:
    //   reef-infer <model.tflite>              — run once and print output
    //   reef-infer <model.tflite> --bench [N]  — warmup + N timed reps (default 50)
    let args: Vec<String> = std::env::args().collect();

    let tflite_path = args.get(1).map(|s| s.as_str());

    // Detect --bench flag and optional rep count.
    let bench_reps: Option<usize> = args
        .windows(2)
        .find_map(|w| {
            if w[0] == "--bench" {
                w[1].parse().ok()
            } else {
                None
            }
        })
        .or_else(|| {
            if args.contains(&"--bench".to_string()) {
                Some(50)
            } else {
                None
            }
        });

    // --input <path>: raw bytes to send as input activations.
    let input_path: Option<&str> = args.windows(2).find_map(|w| {
        if w[0] == "--input" {
            Some(w[1].as_str())
        } else {
            None
        }
    });

    // --output <path>: write raw output bytes to this file instead of printing.
    let output_path: Option<&str> = args.windows(2).find_map(|w| {
        if w[0] == "--output" {
            Some(w[1].as_str())
        } else {
            None
        }
    });

    let model = match tflite_path {
        Some(path) => match reef_infer::model::load_tflite_model(path) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                eprintln!("Failed to load model from {path:?}: {e}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!(
                "Usage: reef-infer <path/to/model_edgetpu.tflite> [--bench [N]] [--input <file>] [--output <file>]"
            );
            eprintln!("  Example: reef-infer model_edgetpu.tflite");
            eprintln!("  Example: reef-infer model_edgetpu.tflite --bench 100");
            eprintln!(
                "  Example: reef-infer model_edgetpu.tflite --input input.bin --output output.bin"
            );
            std::process::exit(1);
        }
    };

    // Load input data from file, or defer to per-device default (all-ones).
    let input_data_from_file: Option<Vec<u8>> = match input_path {
        Some(path) => match std::fs::read(path) {
            Ok(data) => {
                println!("[input] loaded {} bytes from {path}", data.len());
                Some(data)
            }
            Err(e) => {
                eprintln!("Failed to read input file {path}: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    // ── Open device (handles DFU if needed) ─────────────────────────────
    let mut engine = match Device::open().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to open device: {e}");
            std::process::exit(1);
        }
    };

    if let Some(reps) = bench_reps {
        const WARMUP: usize = 3;
        match engine.bench_model(model, WARMUP, reps).await {
            Ok(()) => {}
            Err(e) => eprintln!("bench_model failed: {e:?}"),
        }
    } else {
        let input_data = input_data_from_file.unwrap_or_else(|| {
            let sz = model
                .executables
                .iter()
                .flat_map(|e| &e.input_layers)
                .map(|l| l.size_bytes as usize)
                .next()
                .unwrap_or(8);
            vec![1u8; sz]
        });
        println!("     input: {} byte(s)", input_data.len());

        let t0 = std::time::Instant::now();
        match engine.run_model(Arc::clone(&model), input_data).await {
            Ok(output) => {
                let inference_ms = t0.elapsed().as_secs_f64() * 1e3;
                println!("[timing] inference_ms={inference_ms:.3}");
                if let Some(path) = output_path {
                    match std::fs::write(path, &output) {
                        Ok(()) => println!("     output ({} B) written to {path}", output.len()),
                        Err(e) => println!("     failed to write output to {path}: {e}"),
                    }
                } else if let Some(out_layer) = model
                    .executables
                    .iter()
                    .flat_map(|e| &e.output_layers)
                    .next()
                {
                    let raw = output[0] as i32;
                    let dq = (raw - out_layer.zero_point) as f32 * out_layer.dequantization_factor;
                    println!(
                        "     output raw={raw}  dequantized={dq:.4}  \
                         (scale={} zp={})",
                        out_layer.dequantization_factor, out_layer.zero_point
                    );
                } else {
                    println!(
                        "     output ({} B): {:02x?}",
                        output.len(),
                        &output[..output.len().min(16)]
                    );
                }
            }
            Err(e) => eprintln!("run_model failed: {e:?}"),
        }
    }
}
