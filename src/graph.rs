//! Whole models without `edgetpu_compiler`: reads a CPU int8 `.tflite`,
//! maps its ops to `codegen` generators, and runs the layer programs one
//! after another, through the host.
//!
//! Supported ops:
//! - FULLY_CONNECTED (fused NONE / RELU);
//! - CONV_2D with same padding (fused NONE / RELU; dilation is not in the
//!   bundled schema and assumed 1);
//! - MAX_POOL_2D 2x2, stride 2;
//! - RESHAPE (free in NHWC). Ops computing a reshape's int32 shape (SHAPE,
//!   STRIDED_SLICE, PACK, ...) are skipped.
//!
//! Activations are uint8 on the TPU: TFLite's int8 tensors map to uint8 by
//! adding 128 to the values and the zero point. Weights stay int8.
//!
//! Host glue between layers: the generators cover image sizes 16 / 32 /
//! 64, so the host pads an activation at the bottom / right with its zero
//! point up to the program's size and crops the program's output back.
//! For a same-padding conv with stride 1 the zero-point rows are exactly
//! TensorFlow's padding, and a 2x2 pool never reads them for the outputs
//! kept, so the result is exact. Channels a conv generator does not take
//! (Cin 1 at 32x32) are padded to 16 with zero weights.

use std::sync::Arc;

use crate::codegen::Quant;
use crate::codegen::conv::Conv;
use crate::codegen::fc::Fc;
use crate::codegen::pool::MaxPool;
use crate::model::TfliteModel;
use crate::runner::{Device, DeviceError};
use crate::schema_v3_generated::tflite::{
    ActivationFunctionType, BuiltinOperator, Model, Operator, Padding, TensorType, root_as_model,
};

/// TFLite `TensorType::INT8` (newer than the bundled v3 schema).
const INT8: TensorType = TensorType(9);

/// An activation shape, NHWC without the batch: (y, x, z).
pub type Shape = (usize, usize, usize);

fn len((y, x, z): Shape) -> usize {
    y * x * z
}

/// One layer program.
#[derive(Debug, Clone)]
pub enum Layer {
    Fc(Fc),
    Conv(Conv),
    MaxPool(MaxPool),
}

impl Layer {
    pub fn build(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        match self {
            Layer::Fc(f) => f.build(),
            Layer::Conv(c) => c.build(),
            Layer::MaxPool(p) => Ok(p.build()?.into_model()),
        }
    }

    /// Host reference (bit-exact with the TPU), uint8 in and out, at the
    /// program's shapes.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        match self {
            Layer::Fc(f) => f.reference(x),
            Layer::Conv(c) => c.reference(x),
            Layer::MaxPool(p) => p.reference(x),
        }
    }

    /// The program's input and output shapes.
    pub fn shapes(&self) -> (Shape, Shape) {
        match self {
            Layer::Fc(f) => ((1, 1, f.k), (1, 1, f.n)),
            Layer::Conv(c) => {
                let s = c.stride;
                (
                    (c.height, c.width, c.cin),
                    (c.height.div_ceil(s), c.width.div_ceil(s), c.cout),
                )
            }
            Layer::MaxPool(p) => ((p.height, p.width, p.c), p.out_shape()),
        }
    }
}

/// A layer program with the TFLite shapes around it.
#[derive(Debug, Clone)]
pub struct Step {
    pub layer: Layer,
    /// The TFLite input / output shapes (the program's may be larger).
    pub input: Shape,
    pub output: Shape,
    /// Zero point of the input (the padding value).
    pub pad: u8,
}

impl Step {
    /// The TFLite input -> the program's input: zero-point padding at the
    /// bottom, right and in the extra channels.
    pub fn fit(&self, x: &[u8]) -> Vec<u8> {
        let ((py, px, pz), (y, xx, z)) = (self.layer.shapes().0, self.input);
        if (py, px, pz) == (y, xx, z) {
            return x.to_vec();
        }
        let mut out = vec![self.pad; py * px * pz];
        for r in 0..y {
            for c in 0..xx {
                let (src, dst) = ((r * xx + c) * z, (r * px + c) * pz);
                out[dst..dst + z].copy_from_slice(&x[src..src + z]);
            }
        }
        out
    }

    /// The program's output -> the TFLite output (crop).
    pub fn crop(&self, y: &[u8]) -> Vec<u8> {
        let ((_, px, pz), (oy, ox, oz)) = (self.layer.shapes().1, self.output);
        if len(self.layer.shapes().1) == len(self.output) {
            return y[..len(self.output)].to_vec();
        }
        let mut out = Vec::with_capacity(oy * ox * oz);
        for r in 0..oy {
            for c in 0..ox {
                let src = (r * px + c) * pz;
                out.extend_from_slice(&y[src..src + oz]);
            }
        }
        out
    }

    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        self.crop(&self.layer.reference(&self.fit(x)))
    }
}

/// A model as a chain of layer programs.
#[derive(Debug, Clone)]
pub struct Graph {
    pub steps: Vec<Step>,
    /// Input / output quantization in the uint8 domain.
    pub input: Quant,
    pub output: Quant,
    /// The TFLite input shape.
    pub input_shape: Shape,
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

/// Smallest program size from `sizes` holding `n`.
fn fit_size(n: usize, sizes: &[usize]) -> Option<usize> {
    sizes.iter().copied().find(|&s| s >= n)
}

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
        let shape = |i: i32| tensor_shape(&model, i);
        let int8_io = tensors.get(graph_in as usize).type_() == INT8;

        // Follow the activation from the graph input through the ops.
        let mut act = graph_in;
        let mut steps = Vec::new();
        for op in sg.operators().ok_or("no operators")? {
            let ins: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();
            let outs: Vec<i32> = op.outputs().map(|v| v.iter().collect()).unwrap_or_default();
            let shape_math = outs
                .iter()
                .all(|&o| tensors.get(o as usize).type_() == TensorType::INT32);
            if !ins.contains(&act) || shape_math {
                continue; // a reshape's shape (SHAPE of the activation, ...)
            }
            let (input, output) = (shape(ins[0])?, shape(outs[0])?);
            let pad = tq(ins[0])?.zero_point as u8;
            let code = codes.get(op.opcode_index() as usize).builtin_code();
            let layer = match code {
                BuiltinOperator::RESHAPE => {
                    if tq(ins[0])? != tq(outs[0])? {
                        return Err("RESHAPE changes quantization".into());
                    }
                    None
                }
                BuiltinOperator::FULLY_CONNECTED => {
                    let opts = op
                        .builtin_options_as_fully_connected_options()
                        .ok_or("FULLY_CONNECTED without options")?;
                    let relu = relu(opts.fused_activation_function())?;
                    Some(Layer::Fc(fc_layer(&model, &ins, outs[0], relu)?))
                }
                BuiltinOperator::CONV_2D => Some(Layer::Conv(conv_layer(&model, &op, input)?)),
                BuiltinOperator::MAX_POOL_2D => {
                    let o = op
                        .builtin_options_as_pool_2_doptions()
                        .ok_or("MAX_POOL_2D without options")?;
                    let (y, x, c) = input;
                    let even = y % 2 == 0 && x % 2 == 0;
                    if (
                        o.filter_width(),
                        o.filter_height(),
                        o.stride_w(),
                        o.stride_h(),
                    ) != (2, 2, 2, 2)
                        || (o.padding() == Padding::SAME && !even)
                        || relu(o.fused_activation_function())?
                        || tq(ins[0])? != tq(outs[0])?
                    {
                        return Err(
                            "MAX_POOL_2D: need 2x2, stride 2, valid, same quantization".into()
                        );
                    }
                    let p = MaxPool {
                        height: fit_size(y, &[16, 32]).ok_or("pool input too tall")?,
                        width: fit_size(x, &[16, 32]).ok_or("pool input too wide")?,
                        c,
                        quant: tq(ins[0])?,
                    };
                    p.check()?;
                    Some(Layer::MaxPool(p))
                }
                c => return Err(format!("op {c:?} not supported").into()),
            };
            if let Some(layer) = layer {
                let (pin, _) = layer.shapes();
                if len(pin) < len(input) {
                    return Err(format!("{layer:?}: input {input:?} does not fit").into());
                }
                steps.push(Step {
                    layer,
                    input,
                    output,
                    pad,
                });
            }
            act = outs[0];
        }
        if act != graph_out {
            return Err("the data path does not reach the graph output".into());
        }
        if steps.is_empty() {
            return Err("no layers".into());
        }
        Ok(Graph {
            steps,
            input: tq(graph_in)?,
            output: tq(graph_out)?,
            input_shape: shape(graph_in)?,
            int8_io,
        })
    }

    /// Input bytes per inference.
    pub fn input_len(&self) -> usize {
        len(self.input_shape)
    }

    /// Host reference over the whole graph, uint8 in and out.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        self.steps.iter().fold(x.to_vec(), |x, s| s.reference(&x))
    }

    /// Builds every layer's program.
    pub fn build(&self) -> Result<Vec<Arc<TfliteModel>>, Error> {
        self.steps
            .iter()
            .map(|s| Ok(Arc::new(s.layer.build()?)))
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
        for (s, p) in self.steps.iter().zip(programs) {
            let y = dev.run_model(Arc::clone(p), s.fit(&x)).await?;
            x = s.crop(&y);
        }
        Ok(x)
    }
}

fn relu(a: ActivationFunctionType) -> Result<bool, Error> {
    match a {
        ActivationFunctionType::NONE => Ok(false),
        ActivationFunctionType::RELU => Ok(true),
        a => Err(format!("activation {a:?} not supported").into()),
    }
}

/// A tensor's shape without the batch, as (y, x, z) (1-D: (1, 1, n)).
fn tensor_shape(model: &Model<'_>, i: i32) -> Result<Shape, Error> {
    let t = model
        .subgraphs()
        .unwrap()
        .get(0)
        .tensors()
        .unwrap()
        .get(i as usize);
    let s: Vec<usize> = t
        .shape()
        .ok_or("tensor without shape")?
        .iter()
        .map(|d| d as usize)
        .collect();
    match s[..] {
        [1, y, x, z] => Ok((y, x, z)),
        [1, y, x] => Ok((y, x, 1)),
        [1, n] | [n] => Ok((1, 1, n)),
        _ => Err(format!("tensor {i}: shape {s:?} not supported").into()),
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

/// Weights, per-channel weight scales and bias of an op.
type WeightsBias = (Vec<i8>, Vec<f32>, Vec<i32>);

/// Constant int8 weights with per-channel scales (`n` channels) and the
/// int32 bias of an op.
fn weights_bias(model: &Model<'_>, w: i32, b: Option<i32>, n: usize) -> Result<WeightsBias, Error> {
    let tensors = model.subgraphs().unwrap().get(0).tensors().unwrap();
    let buffers = model.buffers().ok_or("no buffers")?;
    let data = |i: i32| -> Result<&[u8], Error> {
        let t = tensors.get(i as usize);
        let b = buffers.get(t.buffer() as usize);
        Ok(b.data().ok_or("constant tensor without data")?.bytes())
    };
    let wt = tensors.get(w as usize);
    if wt.type_() != INT8 {
        return Err("weights must be int8".into());
    }
    let wq = wt.quantization().ok_or("weights without quantization")?;
    let w_scale: Vec<f32> = wq.scale().ok_or("weights without scale")?.iter().collect();
    let w_scale = match w_scale.len() {
        1 => vec![w_scale[0]; n],
        l if l == n => w_scale,
        l => return Err(format!("{l} weight scales for {n} channels").into()),
    };
    if wq.zero_point().is_some_and(|z| z.iter().any(|z| z != 0)) {
        return Err("asymmetric weights not supported".into());
    }
    let weights = data(w)?.iter().map(|&b| b as i8).collect();
    let bias = match b {
        Some(b) if b >= 0 => data(b)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => vec![0; n],
    };
    Ok((weights, w_scale, bias))
}

/// A FULLY_CONNECTED op: inputs (activation, weights [N][K] int8, bias i32).
fn fc_layer(model: &Model<'_>, ins: &[i32], out: i32, relu: bool) -> Result<Fc, Error> {
    let w = model
        .subgraphs()
        .unwrap()
        .get(0)
        .tensors()
        .unwrap()
        .get(ins[1] as usize);
    let shape: Vec<i32> = w.shape().ok_or("weights without shape")?.iter().collect();
    let [n, k] = shape[..] else {
        return Err("FC weights must be 2-D".into());
    };
    let (n, k) = (n as usize, k as usize);
    let (weights, w_scale, bias) = weights_bias(model, ins[1], ins.get(2).copied(), n)?;
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

/// A CONV_2D op: inputs (activation, weights [Cout][k][k][Cin] int8, bias).
/// The image is padded up to 16 / 32 / 64 (exact for stride 1), and Cin to
/// 16 with zero weights when the generator does not take it at that size.
fn conv_layer(model: &Model<'_>, op: &Operator<'_>, (y, x, cin): Shape) -> Result<Conv, Error> {
    let o = op
        .builtin_options_as_conv_2_doptions()
        .ok_or("CONV_2D without options")?;
    let ins: Vec<i32> = op.inputs().unwrap().iter().collect();
    let out = op.outputs().unwrap().get(0);
    let w = model
        .subgraphs()
        .unwrap()
        .get(0)
        .tensors()
        .unwrap()
        .get(ins[1] as usize);
    let shape: Vec<usize> = w
        .shape()
        .ok_or("weights without shape")?
        .iter()
        .map(|d| d as usize)
        .collect();
    let [cout, kh, kw, wcin] = shape[..] else {
        return Err("conv weights must be 4-D".into());
    };
    let stride = o.stride_w() as usize;
    if kh != kw || wcin != cin || o.stride_h() != o.stride_w() || o.padding() != Padding::SAME {
        return Err("CONV_2D: need a square kernel, equal strides and same padding".into());
    }
    let (height, width) = (
        fit_size(y, &[16, 32, 64]).ok_or("conv input too tall")?,
        fit_size(x, &[16, 32, 64]).ok_or("conv input too wide")?,
    );
    if (height, width) != (y, x) && stride != 1 {
        return Err("CONV_2D: padding the image is exact only for stride 1".into());
    }
    let (weights, w_scale, bias) = weights_bias(model, ins[1], ins.get(2).copied(), cout)?;
    let mut conv = Conv {
        cin,
        cout,
        k: kh,
        stride,
        width,
        height,
        weights,
        w_scale,
        bias,
        input: tensor_quant(model, ins[0])?,
        output: tensor_quant(model, out)?,
        relu: relu(o.fused_activation_function())?,
    };
    if conv.check().is_err() && cin < 16 {
        // feed Cin as 16 channels: zero weights for the extra ones
        let k2 = kh * kw;
        conv.weights = (0..cout * k2)
            .flat_map(|i| (0..16).map(move |c| (i, c)))
            .map(|(i, c)| {
                if c < cin {
                    conv.weights[i * cin + c]
                } else {
                    0
                }
            })
            .collect();
        conv.cin = 16;
    }
    conv.check()?;
    Ok(conv)
}
