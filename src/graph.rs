//! Whole models without `edgetpu_compiler`: reads a CPU int8 `.tflite`,
//! maps its ops to `codegen` generators, and runs the layer programs in
//! graph order through the host, keeping every activation by tensor id (so
//! a residual ADD can read a tensor produced several steps earlier).
//!
//! Supported ops:
//! - FULLY_CONNECTED (fused NONE / RELU);
//! - CONV_2D with same padding (fused NONE / RELU; dilation is not in the
//!   bundled schema and assumed 1);
//! - MAX_POOL_2D 2x2, stride 2;
//! - ADD of two equal-shape tensors (fused NONE / RELU);
//! - AVERAGE_POOL_2D over the whole image (global): 8x8x64 natively
//!   (`codegen::avg_pool`, a mesh reduction with no parameters), other
//!   shapes lowered to an FC that sums each channel's pixels; both round
//!   exactly as TFLite does;
//! - RESHAPE (free in NHWC). Ops computing a reshape's int32 shape (SHAPE,
//!   STRIDED_SLICE, PACK, ...) are skipped.
//!
//! Activations are uint8 on the TPU: TFLite's int8 tensors map to uint8 by
//! adding 128 to the values and the zero point. Weights stay int8.
//!
//! Parameters stay cached on the chip: [`Graph::build`] gives every layer
//! with parameters its own range of wide memory (consecutive from address
//! 0, [`Layer::wide_units`] each), so [`Graph::load`] caches all of them
//! once and [`Graph::run`] only sends instructions and activations.
//!
//! Layers that can run back to back on chip are chained into one call
//! ([`Graph::chains`], `codegen::chain`): convs and residual adds whose
//! intermediate tensors (shortcuts included) stay in narrow memory, so
//! only the chain's input and output cross the host.
//!
//! Host glue between layers: the generators cover image sizes 16 / 32 /
//! 64, so the host pads an activation at the bottom / right with its zero
//! point up to the program's size and crops the program's output back.
//! For a same-padding conv with stride 1 the zero-point rows are exactly
//! TensorFlow's padding, and a 2x2 pool never reads them for the outputs
//! kept, so the result is exact. Channels a conv generator does not take
//! (Cin 1 at 32x32) are padded to 16 with zero weights.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::codegen::add::Add;
use crate::codegen::avg_pool::{GlobalAvgPool, exact_multiplier};
use crate::codegen::chain::{self, Node, Op};
use crate::codegen::conv::Conv;
use crate::codegen::fc::Fc;
use crate::codegen::pool::MaxPool;
use crate::codegen::{PARAM_WIDE_LIMIT, Quant};
use crate::executable_generated::platforms::darwinn::ExecutableType;
use crate::model::{ExecInfo, TfliteModel};
use crate::runner::{Device, DeviceError, LoadedParams};
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
    AvgPool(GlobalAvgPool),
    Add(Add),
}

impl Layer {
    pub fn build(&self) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        self.build_at(0)
    }

    /// The program with its parameters (if any) cached from wide address
    /// `base` (64 B units).
    pub fn build_at(&self, base: i64) -> Result<TfliteModel, Box<dyn std::error::Error>> {
        match self {
            Layer::Fc(f) => f.build_at(base),
            Layer::Conv(c) => c.build_at(base),
            Layer::MaxPool(p) => Ok(p.build()?.into_model()),
            Layer::AvgPool(p) => Ok(p.build()?.into_model()),
            Layer::Add(a) => Ok(a.build()?.into_model()),
        }
    }

    /// Wide memory (64 B units per tile) its cached parameters take.
    pub fn wide_units(&self) -> i64 {
        match self {
            Layer::Fc(f) => f.wide_units(),
            Layer::Conv(c) => c.wide_units(),
            Layer::MaxPool(_) | Layer::AvgPool(_) | Layer::Add(_) => 0,
        }
    }

    /// Host reference (bit-exact with the TPU), uint8 in and out, at the
    /// program's shapes.
    pub fn reference(&self, x: &[Vec<u8>]) -> Vec<u8> {
        match self {
            Layer::Fc(f) => f.reference(&x[0]),
            Layer::Conv(c) => c.reference(&x[0]),
            Layer::MaxPool(p) => p.reference(&x[0]),
            Layer::AvgPool(p) => p.reference(&x[0]),
            Layer::Add(a) => a.reference(&x[0], &x[1]),
        }
    }

    /// The program's input shapes and output shape.
    pub fn shapes(&self) -> (Vec<Shape>, Shape) {
        match self {
            Layer::Fc(f) => (vec![(1, 1, f.k)], (1, 1, f.n)),
            Layer::Conv(c) => {
                let s = c.stride;
                (
                    vec![(c.height, c.width, c.cin)],
                    (c.height.div_ceil(s), c.width.div_ceil(s), c.cout),
                )
            }
            Layer::MaxPool(p) => (vec![(p.height, p.width, p.c)], p.out_shape()),
            Layer::AvgPool(p) => (vec![(p.height, p.width, p.c)], (1, 1, p.c)),
            Layer::Add(a) => {
                let sh = (a.height, a.width, a.c);
                (vec![sh, sh], sh)
            }
        }
    }
}

/// An activation a step reads: TFLite tensor id, its shape, and its zero
/// point (the padding value).
#[derive(Debug, Clone)]
pub struct Input {
    pub tensor: i32,
    pub shape: Shape,
    pub pad: u8,
}

/// A layer program with the TFLite tensors around it.
#[derive(Debug, Clone)]
pub struct Step {
    pub layer: Layer,
    pub inputs: Vec<Input>,
    /// The output tensor id and its TFLite shape (the program's may be
    /// larger).
    pub output: i32,
    pub output_shape: Shape,
}

/// The TFLite tensor `x` (shape `from`) -> a program input of shape `to`:
/// zero-point padding at the bottom, right and in the extra channels
/// (unchanged when only the shape's split differs, as a flatten).
fn fit(x: &[u8], from: Shape, to: Shape, pad: u8) -> Vec<u8> {
    let ((py, px, pz), (y, xx, z)) = (to, from);
    if len(to) == len(from) && (to.0 == 1 && to.1 == 1 || to == from) {
        return x.to_vec();
    }
    let mut out = vec![pad; py * px * pz];
    for r in 0..y {
        for c in 0..xx {
            let (src, dst) = ((r * xx + c) * z, (r * px + c) * pz);
            out[dst..dst + z].copy_from_slice(&x[src..src + z]);
        }
    }
    out
}

impl Step {
    /// The step's TFLite inputs -> the program's inputs.
    pub fn fit(&self, x: &[&[u8]]) -> Vec<Vec<u8>> {
        let (shapes, _) = self.layer.shapes();
        self.inputs
            .iter()
            .zip(x)
            .zip(shapes)
            .map(|((i, x), to)| fit(x, i.shape, to, i.pad))
            .collect()
    }

    /// The program's output -> the TFLite output (crop).
    pub fn crop(&self, y: &[u8]) -> Vec<u8> {
        let ((_, px, pz), (oy, ox, oz)) = (self.layer.shapes().1, self.output_shape);
        if len(self.layer.shapes().1) == len(self.output_shape) {
            return y[..len(self.output_shape)].to_vec();
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

    pub fn reference(&self, x: &[&[u8]]) -> Vec<u8> {
        self.crop(&self.layer.reference(&self.fit(x)))
    }
}

/// A model as layer programs in graph order.
#[derive(Debug, Clone)]
pub struct Graph {
    pub steps: Vec<Step>,
    /// Input / output quantization in the uint8 domain.
    pub input: Quant,
    pub output: Quant,
    /// The TFLite input shape.
    pub input_shape: Shape,
    /// Graph input / output tensor ids.
    pub input_tensor: i32,
    pub output_tensor: i32,
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

        // Activations available so far (graph input, then op outputs);
        // RESHAPE aliases its output to its input.
        let mut alias: HashMap<i32, i32> = HashMap::new();
        let mut have = vec![graph_in];
        let resolve = |alias: &HashMap<i32, i32>, t: i32| *alias.get(&t).unwrap_or(&t);
        let mut steps = Vec::new();
        for op in sg.operators().ok_or("no operators")? {
            let ins: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();
            let outs: Vec<i32> = op.outputs().map(|v| v.iter().collect()).unwrap_or_default();
            let shape_math = outs
                .iter()
                .all(|&o| tensors.get(o as usize).type_() == TensorType::INT32);
            let data: Vec<i32> = ins
                .iter()
                .copied()
                .filter(|&i| i >= 0 && have.contains(&resolve(&alias, i)))
                .collect();
            if data.is_empty() || shape_math {
                continue; // a reshape's shape (SHAPE of the activation, ...)
            }
            let input = |i: i32| -> Result<Input, Error> {
                Ok(Input {
                    tensor: resolve(&alias, i),
                    shape: shape(i)?,
                    pad: tq(i)?.zero_point as u8,
                })
            };
            let (in0, output) = (shape(ins[0])?, shape(outs[0])?);
            let code = codes.get(op.opcode_index() as usize).builtin_code();
            let layer = match code {
                BuiltinOperator::RESHAPE => {
                    if tq(ins[0])? != tq(outs[0])? {
                        return Err("RESHAPE changes quantization".into());
                    }
                    alias.insert(outs[0], resolve(&alias, ins[0]));
                    continue;
                }
                BuiltinOperator::FULLY_CONNECTED => {
                    let opts = op
                        .builtin_options_as_fully_connected_options()
                        .ok_or("FULLY_CONNECTED without options")?;
                    let relu = relu(opts.fused_activation_function())?;
                    Layer::Fc(fc_layer(&model, &ins, outs[0], relu)?)
                }
                BuiltinOperator::CONV_2D => Layer::Conv(conv_layer(&model, &op, in0)?),
                BuiltinOperator::MAX_POOL_2D => {
                    let o = op
                        .builtin_options_as_pool_2_doptions()
                        .ok_or("MAX_POOL_2D without options")?;
                    let (y, x, c) = in0;
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
                    Layer::MaxPool(p)
                }
                BuiltinOperator::AVERAGE_POOL_2D => {
                    let o = op
                        .builtin_options_as_pool_2_doptions()
                        .ok_or("AVERAGE_POOL_2D without options")?;
                    let (y, x, c) = in0;
                    if (o.filter_height() as usize, o.filter_width() as usize) != (y, x)
                        || relu(o.fused_activation_function())?
                        || tq(ins[0])? != tq(outs[0])?
                    {
                        return Err("AVERAGE_POOL_2D: only global pools, same quantization".into());
                    }
                    let int8 = tensors.get(ins[0] as usize).type_() == INT8;
                    match GlobalAvgPool::tflite(y, x, c, tq(ins[0])?, int8) {
                        Ok(p) => Layer::AvgPool(p),
                        Err(_) if int8 => Layer::Fc(average_fc(y * x, c)?),
                        Err(e) => return Err(e.into()),
                    }
                }
                BuiltinOperator::ADD => {
                    let o = op
                        .builtin_options_as_add_options()
                        .ok_or("ADD without options")?;
                    let (y, x, c) = in0;
                    if shape(ins[1])? != in0 {
                        return Err("ADD: broadcasting not supported".into());
                    }
                    let a = Add {
                        height: fit_size(y, &[16, 32]).ok_or("add input too tall")?,
                        width: fit_size(x, &[16, 32]).ok_or("add input too wide")?,
                        c,
                        a: tq(ins[0])?,
                        b: tq(ins[1])?,
                        output: tq(outs[0])?,
                        relu: relu(o.fused_activation_function())?,
                    };
                    a.check()?;
                    Layer::Add(a)
                }
                c => return Err(format!("op {c:?} not supported").into()),
            };
            let inputs: Vec<Input> = match &layer {
                Layer::Add(_) => vec![input(ins[0])?, input(ins[1])?],
                _ => vec![input(ins[0])?],
            };
            for (i, s) in inputs.iter().zip(layer.shapes().0) {
                if len(s) < len(i.shape) {
                    return Err(format!("{layer:?}: input {:?} does not fit", i.shape).into());
                }
            }
            steps.push(Step {
                layer,
                inputs,
                output: outs[0],
                output_shape: output,
            });
            have.push(outs[0]);
        }
        let output_tensor = resolve(&alias, graph_out);
        if !have.contains(&output_tensor) {
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
            input_tensor: graph_in,
            output_tensor,
            int8_io,
        })
    }

    /// Input bytes per inference.
    pub fn input_len(&self) -> usize {
        len(self.input_shape)
    }

    /// Host reference over the whole graph, uint8 in and out.
    pub fn reference(&self, x: &[u8]) -> Vec<u8> {
        let mut acts: HashMap<i32, Vec<u8>> = HashMap::from([(self.input_tensor, x.to_vec())]);
        for s in &self.steps {
            let ins: Vec<&[u8]> = s.inputs.iter().map(|i| &acts[&i.tensor][..]).collect();
            let y = s.reference(&ins);
            acts.insert(s.output, y);
        }
        acts.remove(&self.output_tensor).unwrap()
    }

    /// Wide address (64 B units) of each step's cached parameters: the
    /// layers one after the other from 0.
    pub fn wide_bases(&self) -> Result<Vec<i64>, Error> {
        let mut next = 0;
        let bases = self
            .steps
            .iter()
            .map(|s| {
                let base = next;
                next += s.layer.wide_units();
                base
            })
            .collect();
        if next > PARAM_WIDE_LIMIT {
            return Err(format!(
                "parameters need {next:#x} wide units per tile, {PARAM_WIDE_LIMIT:#x} available"
            )
            .into());
        }
        Ok(bases)
    }

    /// Builds every layer's program, each with its parameters at its own
    /// wide address ([`Graph::wide_bases`]), and the chains run in place
    /// of their layers ([`Graph::chains`]).
    pub fn build(&self) -> Result<Programs, Error> {
        let steps = self
            .steps
            .iter()
            .zip(self.wide_bases()?)
            .map(|(s, base)| Ok(Arc::new(s.layer.build_at(base)?)))
            .collect::<Result<_, Error>>()?;
        let chains = self
            .chains()?
            .into_iter()
            .filter(|r| r.len() > 1)
            .map(|r| Ok((r.clone(), Arc::new(self.chain(r)?))))
            .collect::<Result<_, Error>>()?;
        Ok(Programs { steps, chains })
    }

    /// The steps as runs that each go in one call: greedily from the
    /// first step, the longest run [`Graph::chain`] takes, else the step
    /// alone.
    pub fn chains(&self) -> Result<Vec<Range<usize>>, Error> {
        let bases = self.wide_bases()?;
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.steps.len() {
            let end = (i + 2..=self.steps.len())
                .rev()
                .find(|&j| self.chain_nodes(i..j, &bases).is_some())
                .unwrap_or(i + 1);
            out.push(i..end);
            i = end;
        }
        Ok(out)
    }

    /// `steps` as chain nodes, if they can run on chip in one call: the
    /// first reads one tensor from outside, every other input is made
    /// inside, nothing but the last output is read outside, no host glue
    /// pads or crops a tensor that stays inside, and the chain plans.
    fn chain_nodes<'a>(&'a self, steps: Range<usize>, bases: &[i64]) -> Option<Vec<Node<'a>>> {
        let made: HashMap<i32, usize> = steps
            .clone()
            .map(|k| (self.steps[k].output, k + 1 - steps.start))
            .collect();
        let last = self.steps[steps.end - 1].output;
        let escapes = |t: i32| {
            t == self.output_tensor
                || self.steps[..steps.start]
                    .iter()
                    .chain(&self.steps[steps.end..])
                    .any(|s| s.inputs.iter().any(|i| i.tensor == t))
        };
        if made.keys().any(|&t| t != last && escapes(t)) {
            return None;
        }
        let mut nodes = Vec::new();
        for k in steps.clone() {
            let s = &self.steps[k];
            let (ins, out) = s.layer.shapes();
            let internal = k + 1 < steps.end;
            if internal && out != s.output_shape {
                return None;
            }
            let mut inputs = Vec::new();
            for (i, shape) in s.inputs.iter().zip(&ins) {
                match made.get(&i.tensor) {
                    Some(&t) if i.shape == *shape => inputs.push(t),
                    None if k == steps.start => inputs.push(0),
                    _ => return None,
                }
            }
            let op = match &s.layer {
                Layer::Conv(c) => Op::Conv(c, bases[k]),
                Layer::Add(a) => Op::Add(a),
                _ => return None,
            };
            nodes.push(Node { op, inputs });
        }
        chain::program(&nodes).is_ok().then_some(nodes)
    }

    /// One EXECUTION_ONLY program running `steps` on chip (a run from
    /// [`Graph::chains`]), the layers' parameters cached by their own
    /// programs. Input: the first step's program input; output: the last
    /// one's program output.
    pub fn chain(&self, steps: Range<usize>) -> Result<TfliteModel, Error> {
        let bases = self.wide_bases()?;
        let nodes = self
            .chain_nodes(steps.clone(), &bases)
            .ok_or(format!("steps {steps:?} do not chain"))?;
        let chunk = chain::program(&nodes)?;
        let exec = |k: usize| -> Result<ExecInfo, Error> {
            Ok(self.steps[k]
                .layer
                .build_at(bases[k])?
                .executables
                .into_iter()
                .find(|e| e.exec_type != ExecutableType::PARAMETER_CACHING)
                .ok_or("no execution program")?)
        };
        let (first, last) = (exec(steps.start)?, exec(steps.end - 1)?);
        Ok(TfliteModel {
            executables: vec![ExecInfo {
                exec_type: ExecutableType::EXECUTION_ONLY,
                parameters: Vec::new(),
                instruction_chunks: vec![chunk],
                input_layers: first.input_layers,
                output_layers: last.output_layers,
                scratch_size_bytes: 0,
            }],
        })
    }

    /// Caches every layer's parameters on the TPU and loads the chains
    /// (programs from [`Graph::build`]). The parameters stay valid until
    /// another model's are loaded.
    pub async fn load(&self, dev: &mut Device, programs: &Programs) -> Result<Loaded, DeviceError> {
        let mut steps = Vec::with_capacity(programs.steps.len());
        for p in &programs.steps {
            steps.push(dev.load_params(Arc::clone(p)).await?);
        }
        let mut chains = Vec::with_capacity(programs.chains.len());
        for (r, p) in &programs.chains {
            chains.push((r.clone(), dev.load_params(Arc::clone(p)).await?));
        }
        Ok(Loaded { steps, chains })
    }

    /// Runs the graph loaded by [`Graph::load`] on the TPU, a chain in one
    /// call and every other layer in its own, uint8 in and out.
    pub async fn run(
        &self,
        dev: &mut Device,
        loaded: &Loaded,
        x: &[u8],
    ) -> Result<Vec<u8>, DeviceError> {
        let mut acts: HashMap<i32, Vec<u8>> = HashMap::from([(self.input_tensor, x.to_vec())]);
        let mut k = 0;
        while k < self.steps.len() {
            let (end, p) = match loaded.chains.iter().find(|(r, _)| r.start == k) {
                Some((r, p)) => (r.end, p),
                None => (k + 1, &loaded.steps[k]),
            };
            let (first, last) = (&self.steps[k], &self.steps[end - 1]);
            let ins: Vec<&[u8]> = first.inputs.iter().map(|i| &acts[&i.tensor][..]).collect();
            let y = dev.run_inference(p, first.fit(&ins).concat()).await?;
            acts.insert(last.output, last.crop(&y));
            k = end;
        }
        Ok(acts.remove(&self.output_tensor).unwrap())
    }
}

/// A graph's programs ([`Graph::build`]): every layer's own (they also
/// cache its parameters), and the chains that replace runs of them.
pub struct Programs {
    pub steps: Vec<Arc<TfliteModel>>,
    pub chains: Vec<(Range<usize>, Arc<TfliteModel>)>,
}

impl Programs {
    /// Calls per inference.
    pub fn calls(&self) -> usize {
        self.steps.len() - self.chains.iter().map(|(r, _)| r.len() - 1).sum::<usize>()
    }
}

/// A graph on the TPU ([`Graph::load`]).
pub struct Loaded {
    pub steps: Vec<LoadedParams>,
    pub chains: Vec<(Range<usize>, LoadedParams)>,
}

/// A global average pool over `n` pixels of `c` channels as an FC on the
/// flattened input (K = n c, weight 127 from each pixel to its channel),
/// rounding exactly as TFLite's int8 AVERAGE_POOL_2D: the raw int8 sum /
/// n, half away from zero. The FC subtracts zero point 128 (summing the
/// raw int8 values), adds it back, and multiplies by the smallest f32 at
/// or above 1 / (127 n) for which the TPU's round-half-even of every
/// possible sum matches (ties sit just past .5 in magnitude).
fn average_fc(n: usize, c: usize) -> Result<Fc, Error> {
    let q = Quant {
        scale: 1.0,
        zero_point: 128,
    };
    let ni = n as i64;
    let m = exact_multiplier(ni, 127, -128 * ni..=127 * ni, -128.0, 127.0)
        .ok_or(format!("average pool over {n} pixels: no exact multiplier"))?;
    let k = n * c;
    let fc = Fc {
        k,
        n: c,
        weights: (0..c * k)
            .map(|i| if (i % k) % c == i / k { 127 } else { 0 })
            .collect(),
        w_scale: vec![m; c],
        bias: vec![0; c],
        input: q,
        output: q,
        relu: false,
    };
    fc.check()?;
    Ok(fc)
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
