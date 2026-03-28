#![cfg(feature = "usb")]

use std::sync::Arc;

use reef_infer::model::load_tflite_model;
use reef_infer::runner::Device;

/// Test that the TPU produces results matching a known Nx2 matmul within
/// quantization tolerance. Uses `matmul_edgetpu.tflite`
/// which computes y = W @ x where W is 8x2 and x is 2x1.
///
/// Weight matrix (set in gen_models.py):
///   W = [[1,0], [0,1], [1,10], [1,-1], [3,7], [-5,10], [50,-30], [100,100]]
#[tokio::test]
async fn matmul_precision() {
    let mut dev = Device::open().await.expect("failed to open device");

    let model = Arc::new(
        load_tflite_model("tests/models/matmul_edgetpu.tflite")
            .expect("failed to load keras quant model"),
    );
    let loaded = dev.load_params(Arc::clone(&model)).await.unwrap();

    let (in_scale, in_zp) = loaded.input_quant();
    let (out_scale, _out_zp) = loaded.output_quant();

    println!(
        "Quant params: input(scale={}, zp={}), output(scale={}, zp={})",
        in_scale, in_zp, out_scale, _out_zp
    );

    // Known weight matrix from gen_models.py build_matmul_tflite
    let weights: [[f32; 2]; 8] = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 10.0],
        [1.0, -1.0],
        [3.0, 7.0],
        [-5.0, 10.0],
        [50.0, -30.0],
        [100.0, 100.0],
    ];

    // Test vectors as float pairs [a, b]
    let test_inputs: &[[f32; 2]] = &[
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [10.0, -5.0],
        [-50.0, 30.0],
    ];

    for input_float in test_inputs {
        // Compute expected using the quantize round-trip of the input
        // (what the TPU actually sees after quantization)
        let actual_input: Vec<f32> = input_float
            .iter()
            .map(|&x| {
                let q = ((x / in_scale).round() as i32 + in_zp).clamp(0, 255);
                (q as f32 - in_zp as f32) * in_scale
            })
            .collect();

        let expected: Vec<f32> = weights
            .iter()
            .map(|row| row[0] * actual_input[0] + row[1] * actual_input[1])
            .collect();

        let output_float = dev.run_inference_f32(&loaded, input_float).await.unwrap();

        println!(
            "input={:?} actual={:?} expected={:?} tpu={:?}",
            input_float, actual_input, expected, output_float
        );

        // Verify precision: allow up to 2 quantization steps + 5% relative error
        let abs_tol = out_scale * 2.0;
        for (i, (&exp, &got)) in expected.iter().zip(output_float.iter()).enumerate() {
            let err = (exp - got).abs();
            let tol = abs_tol + exp.abs() * 0.05;
            assert!(
                err <= tol,
                "input={:?} output[{}]: expected {}, got {}, error {} > tolerance {}",
                input_float,
                i,
                exp,
                got,
                err,
                tol
            );
        }
    }
}
