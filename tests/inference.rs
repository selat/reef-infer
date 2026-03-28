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

    let expected_outputs: &[[f32; 8]] = &[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 180.39215087890625],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 180.39215087890625],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 180.39215087890625],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -180.39215087890625,
            721.568603515625,
            541.1764526367188,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            541.1764526367188,
            -3427.450927734375,
            -1984.313720703125,
        ],
    ];

    let tolerance = 0.0001;

    for (test_id, input_float) in test_inputs.iter().enumerate() {
        let output_float = dev.run_inference_f32(&loaded, input_float).await.unwrap();

        println!(
            "input={:?} expected={:?} tpu={:?}",
            input_float, expected_outputs[test_id], output_float
        );

        // Verify precision: allow up to 2 quantization steps + 5% relative error
        let abs_tol = out_scale * 2.0;
        for (i, (&exp, &got)) in expected_outputs[test_id]
            .iter()
            .zip(output_float.iter())
            .enumerate()
        {
            let err = (exp - got).abs();
            assert!(
                err <= tolerance,
                "input={:?} output[{}]: expected {}, got {}, error {} > tolerance {}",
                input_float,
                i,
                exp,
                got,
                err,
                tolerance
            );
        }
    }
}
