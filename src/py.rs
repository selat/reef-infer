use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::model::{TfliteModel, load_tflite_model};
use crate::runner::{self, LoadedParams};

// ── helpers ───────────────────────────────────────────────────────────────────

fn runtime_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// A compiled Edge TPU model loaded from a `.tflite` file.
#[pyclass]
pub struct Model {
    pub(crate) inner: Arc<TfliteModel>,
}

#[pymethods]
impl Model {
    /// Return the number of input bytes expected by this model.
    #[getter]
    fn input_size(&self) -> usize {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.input_layers)
            .map(|l| l.size_bytes as usize)
            .next()
            .unwrap_or(0)
    }

    /// Return the number of output bytes produced by this model.
    #[getter]
    fn output_size(&self) -> usize {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.output_layers)
            .map(|l| l.size_bytes as usize)
            .sum()
    }

    /// Zero-point of the first input layer (for dequantization).
    #[getter]
    fn input_zero_point(&self) -> i32 {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.input_layers)
            .next()
            .map(|l| l.zero_point)
            .unwrap_or(0)
    }

    /// Scale (dequantization factor) of the first input layer.
    #[getter]
    fn input_scale(&self) -> f32 {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.input_layers)
            .next()
            .map(|l| l.dequantization_factor)
            .unwrap_or(1.0)
    }

    /// Zero-point of the first output layer.
    #[getter]
    fn output_zero_point(&self) -> i32 {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.output_layers)
            .next()
            .map(|l| l.zero_point)
            .unwrap_or(0)
    }

    /// Scale (dequantization factor) of the first output layer.
    #[getter]
    fn output_scale(&self) -> f32 {
        self.inner
            .executables
            .iter()
            .flat_map(|e| &e.output_layers)
            .next()
            .map(|l| l.dequantization_factor)
            .unwrap_or(1.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(input={}B, output={}B, executables={})",
            self.input_size(),
            self.output_size(),
            self.inner.executables.len(),
        )
    }
}

// ── LoadedModel ───────────────────────────────────────────────────────────────

/// Parameters loaded onto the chip — reuse this across many `run_inference`
/// calls to avoid re-uploading weights.
#[pyclass]
pub struct LoadedModel {
    inner: LoadedParams,
}

// ── Device ────────────────────────────────────────────────────────────────────

/// An open Edge TPU device, ready for inference.
#[pyclass]
pub struct Device {
    engine: runner::Device,
    rt: Runtime,
}

#[pymethods]
impl Device {
    /// Upload model weights to the chip's SRAM.
    ///
    /// This is the slow step (~100 ms for large models).  The returned
    /// `LoadedModel` can be passed to `run_inference` many times without
    /// re-uploading.
    fn load_params(&mut self, model: &Model) -> PyResult<LoadedModel> {
        let params = self
            .rt
            .block_on(self.engine.load_params(Arc::clone(&model.inner)))
            .map_err(|e| runtime_err(format!("{e:?}")))?;
        Ok(LoadedModel { inner: params })
    }

    /// Run one inference.  Returns the raw output bytes.
    ///
    /// `input` must be a `bytes`-like object whose length matches
    /// `model.input_size`.
    fn run_inference(&mut self, loaded: &LoadedModel, input: &[u8]) -> PyResult<Vec<u8>> {
        self.rt
            .block_on(self.engine.run_inference(&loaded.inner, input.to_vec()))
            .map_err(|e| runtime_err(format!("{e:?}")))
    }

    /// Run one inference and return dequantized float32 values.
    ///
    /// Each output layer is dequantized independently using its own zero_point
    /// and scale, so the result is correct for multi-output models (e.g. SSD)
    /// where different outputs have different quantization parameters.
    fn run_inference_f32(&mut self, loaded: &LoadedModel, input: Vec<f32>) -> PyResult<Vec<f32>> {
        self.rt
            .block_on(self.engine.run_inference_f32(&loaded.inner, &input))
            .map_err(|e| runtime_err(format!("{e:?}")))
    }

    fn __repr__(&self) -> &str {
        "Device(Edge TPU)"
    }
}

// ── Module-level functions ────────────────────────────────────────────────────

/// Load a compiled Edge TPU model from a `.tflite` file.
#[pyfunction]
fn load_model(path: &str) -> PyResult<Model> {
    let inner = load_tflite_model(path).map_err(runtime_err)?;
    Ok(Model {
        inner: Arc::new(inner),
    })
}

/// Open the first available Edge TPU device and initialise the chip.
///
/// If the device is in DFU mode, firmware is uploaded automatically.
/// Raises `RuntimeError` if no device is found or initialisation fails.
#[pyfunction]
fn open_device() -> PyResult<Device> {
    let rt = Runtime::new().map_err(runtime_err)?;

    let engine = rt
        .block_on(runner::Device::open())
        .map_err(|e| runtime_err(format!("{e}")))?;

    Ok(Device { engine, rt })
}

// ── PyO3 module ───────────────────────────────────────────────────────────────

#[pymodule]
pub fn reef_infer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<LoadedModel>()?;
    m.add_class::<Device>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(open_device, m)?)?;
    Ok(())
}
