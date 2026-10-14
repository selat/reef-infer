//! Whole models without `edgetpu_compiler`: reads a CPU int8 `.tflite`,
//! maps its ops to `codegen` generators, and runs the layer programs one
//! after another, through the host.
//!
//! Supported graphs: a chain of FULLY_CONNECTED layers (fused NONE / RELU),
//! with RESHAPE (a flatten: free in NHWC) anywhere. Ops computing a
//! reshape's int32 shape (SHAPE, STRIDED_SLICE, PACK, ...) are skipped.
//!
//! Activations are uint8 on the TPU: TFLite's int8 tensors map to uint8 by
//! adding 128 to the values and the zero point. Weights stay int8 (the FC
//! parameter blob stores `W + 128` with weight zero point 128).

use std::sync::Arc;

use crate::codegen::Quant;
use crate::codegen::fc::Fc;
use crate::model::TfliteModel;
use crate::runner::{Device, DeviceError};
use crate::schema_v3_generated::tflite::{
    ActivationFunctionType, BuiltinOperator, Model, TensorType, root_as_model,
};

/// TFLite `TensorType::INT8` (newer than the bundled v3 schema).
const INT8: TensorType = TensorType(9);

/// One layer program.
#[derive(Debug, Clone)]
pub enum Layer {
    Fc(Fc),
}

impl Layer {
    pub fn build(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        match self {
            Layer::Fc(f) => f.build(),
        }
    }

    /// Host reference (bit-exact with the TPU), uint8 in and out.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        match self {
            Layer::Fc(f) => f.reference(x),
        }
    }

    pub fn output_len(&self) -> usize {
        match self {
            Layer::Fc(f) => f.n,
        }
    }
}

/// A model as a chain of layer programs.
#[derive(Debug, Clone)]
pub struct Graph {
    pub layers: Vec<Layer>,
    /// Input / output quantization in the uint8 domain.
    pub input: Quant,
    pub output: Quant,
    /// Whether the `.tflite` input / output tensors are int8 (else uint8).
    pub int8_io: bool,
}

/// int8 -> uint8 (the TPU's activation type).
pub fn i8_to_u8(x: &[i8]) -> Vec<u8> {
    x.iter().map(|&v| (v as i16 + 128) as u8).collect()
}

/// uint8 -> int8.
pub fn u8_to_i8(x: &[u8]) -> Vec<i8> {
    x.iter().map(|&v| (v as i16 - 128) as i8).collect()
}

type Error = Box<dyn std::error::Error>;

impl Graph {
    pub fn from_tflite(path: &str) -> Result<Graph, Error> {
        Graph::from_tflite_bytes(&std::fs::read(path)?)
    }

    pub fn from_tflite_bytes(buf: &[u8]) -> Result<Graph, Error> {
        let model = root_as_model(buf)?;
        let sg = model.subgraphs().ok_or("no subgraph")?.get(0);
        let tensors = sg.tensors().ok_or("no tensors")?;
        let codes = model.operator_codes().ok_or("no operator codes")?;
        let io = |v: Option<flatbuffers::Vector<'_, i32>>| -> Result<i32, Error> {
            let v = v.ok_or("missing tensor list")?;
            if v.len() != 1 {
                return Err(format!("{} graph inputs / outputs; need 1", v.len()).into());
            }
            Ok(v.get(0))
        };
        let (graph_in, graph_out) = (io(sg.inputs())?, io(sg.outputs())?);
        let tq = |i: i32| tensor_quant(&model, i);
        let int8_io = tensors.get(graph_in as usize).type_() == INT8;

        // Follow the activation from the graph input through the ops.
        let mut act = graph_in;
        let mut layers = Vec::new();
        for op in sg.operators().ok_or("no operators")? {
            let ins: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();
            let outs: Vec<i32> = op.outputs().map(|v| v.iter().collect()).unwrap_or_default();
            let shape_math = outs
                .iter()
                .all(|&o| tensors.get(o as usize).type_() == TensorType::INT32);
            if !ins.contains(&act) || shape_math {
                continue; // a reshape's shape (SHAPE of the activation, ...)
            }
            let code = codes.get(op.opcode_index() as usize).builtin_code();
            match code {
                BuiltinOperator::RESHAPE => {
                    if tq(ins[0])? != tq(outs[0])? {
                        return Err("RESHAPE changes quantization".into());
                    }
                }
                BuiltinOperator::FULLY_CONNECTED => {
                    let opts = op
                        .builtin_options_as_fully_connected_options()
                        .ok_or("FULLY_CONNECTED without options")?;
                    let relu = match opts.fused_activation_function() {
                        ActivationFunctionType::NONE => false,
                        ActivationFunctionType::RELU => true,
                        a => return Err(format!("FC activation {a:?} not supported").into()),
                    };
                    layers.push(Layer::Fc(fc_layer(&model, &ins, outs[0], relu)?));
                }
                c => return Err(format!("op {c:?} not supported").into()),
            }
            act = outs[0];
        }
        if act != graph_out {
            return Err("the data path does not reach the graph output".into());
        }
        if layers.is_empty() {
            return Err("no layers".into());
        }
        Ok(Graph {
            layers,
            input: tq(graph_in)?,
            output: tq(graph_out)?,
            int8_io,
        })
    }

    /// Input bytes per inference.
    pub fn input_len(&self) -> usize {
        match &self.layers[0] {
            Layer::Fc(f) => f.k,
        }
    }

    /// Host reference over the whole graph, uint8 in and out.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        self.layers.iter().fold(x.to_vec(), |x, l| l.reference(&x))
    }

    /// Builds every layer's program (PARAMETER_CACHING + EXECUTION_ONLY).
    pub fn build(&self) -> Result<Vec<Arc<TfliteModel>>, Error> {
        self.layers
            .iter()
            .map(|l| Ok(Arc::new(l.build()?)))
            .collect()
    }

    /// Runs `programs` (from [`Graph::build`]) on the TPU, uint8 in and
    /// out. Every layer reloads its parameters: the layers' parameter
    /// blobs share tile wide memory from address 0.
    pub async fn run(
        &self,
        dev: &mut Device,
        programs: &[Arc<TfliteModel>],
        x: &[u8],
    ) -> Result<Vec<u8>, DeviceError> {
        let mut x = x.to_vec();
        for (l, p) in self.layers.iter().zip(programs) {
            let mut y = dev.run_model(Arc::clone(p), x).await?;
            y.truncate(l.output_len());
            x = y;
        }
        Ok(x)
    }
}

/// uint8-domain quantization of tensor `i` (int8 zero points + 128).
fn tensor_quant(model: &Model<'_>, i: i32) -> Result<Quant, Error> {
    let t = model
        .subgraphs()
        .unwrap()
        .get(0)
        .tensors()
        .unwrap()
        .get(i as usize);
    let q = t.quantization().ok_or("tensor without quantization")?;
    let (s, zp) = (
        q.scale().ok_or("no scale")?,
        q.zero_point().ok_or("no zero point")?,
    );
    if s.len() != 1 {
        return Err(format!("tensor {i}: per-channel activation quantization").into());
    }
    let shift = match t.type_() {
        INT8 => 128,
        TensorType::UINT8 => 0,
        ty => return Err(format!("tensor {i}: type {ty:?} not supported").into()),
    };
    Ok(Quant {
        scale: s.get(0),
        zero_point: zp.get(0) as i32 + shift,
    })
}

/// A FULLY_CONNECTED op: inputs (activation, weights [N][K] int8, bias i32).
fn fc_layer(model: &Model<'_>, ins: &[i32], out: i32, relu: bool) -> Result<Fc, Error> {
    let sg = model.subgraphs().unwrap().get(0);
    let tensors = sg.tensors().unwrap();
    let buffers = model.buffers().ok_or("no buffers")?;
    let data = |i: i32| -> Result<&[u8], Error> {
        let t = tensors.get(i as usize);
        let b = buffers.get(t.buffer() as usize);
        Ok(b.data().ok_or("constant tensor without data")?.bytes())
    };
    let w = tensors.get(ins[1] as usize);
    if w.type_() != INT8 {
        return Err("FC weights must be int8".into());
    }
    let shape: Vec<i32> = w.shape().ok_or("weights without shape")?.iter().collect();
    let [n, k] = shape[..] else {
        return Err("FC weights must be 2-D".into());
    };
    let (n, k) = (n as usize, k as usize);
    let wq = w.quantization().ok_or("weights without quantization")?;
    let w_scale: Vec<f32> = wq.scale().ok_or("weights without scale")?.iter().collect();
    let w_scale = match w_scale.len() {
        1 => vec![w_scale[0]; n],
        l if l == n => w_scale,
        l => return Err(format!("{l} weight scales for {n} channels").into()),
    };
    if wq.zero_point().is_some_and(|z| z.iter().any(|z| z != 0)) {
        return Err("asymmetric weights not supported".into());
    }
    let weights = data(ins[1])?.iter().map(|&b| b as i8).collect();
    let bias = match ins.get(2) {
        Some(&b) if b >= 0 => data(b)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => vec![0; n],
    };
    let fc = Fc {
        k,
        n,
        weights,
        w_scale,
        bias,
        input: tensor_quant(model, ins[0])?,
        output: tensor_quant(model, out)?,
        relu,
    };
    fc.check()?;
    Ok(fc)
}
