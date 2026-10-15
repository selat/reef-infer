//! PyTorch front end: a model quantized with PT2E (`torch.export`,
//! `tools/torch/edgetpu_quant.py`) and exported as a JSON graph, mapped to
//! the same layer programs as the TFLite front end ([`Graph::from_torch`]).
//!
//! The JSON graph (`"format": "reef-graph"`, version 1) numbers its
//! tensors (0 is the graph input) and lists ops in execution order, each
//! with its output tensor's NHWC shape and int8 quantization; weights are
//! int8 per output channel, biases int32 at `s_in s_w` (format in
//! `tools/torch/edgetpu_quant.py`). The int8 tensors map to the TPU's
//! uint8 as in the TFLite front end (+128).
//!
//! PyTorch's semantics where they differ from TFLite's: a conv's
//! `padding = p` pads p before and what the output size needs after
//! (`Conv::pad_before`); a global average pool keeps its input's
//! quantization and rounds the mean half to even.

use std::collections::HashMap;

use serde_json::Value;

use crate::codegen::Quant;
use crate::codegen::avg_pool::GlobalAvgPool;
use crate::codegen::conv::Conv;
use crate::codegen::fc::Fc;
use crate::graph::{Graph, Input, Layer, Shape, Step, add_layer, fit_conv, max_pool_layer};

type Error = Box<dyn std::error::Error>;

fn field<'a>(v: &'a Value, name: &str) -> Result<&'a Value, Error> {
    v.get(name)
        .ok_or_else(|| format!("missing \"{name}\"").into())
}

fn int(v: &Value, name: &str) -> Result<i64, Error> {
    field(v, name)?
        .as_i64()
        .ok_or_else(|| format!("\"{name}\" is not an integer").into())
}

fn num(v: &Value, name: &str) -> Result<f64, Error> {
    field(v, name)?
        .as_f64()
        .ok_or_else(|| format!("\"{name}\" is not a number").into())
}

fn ints(v: &Value, name: &str) -> Result<Vec<i64>, Error> {
    field(v, name)?
        .as_array()
        .ok_or(format!("\"{name}\" is not an array"))?
        .iter()
        .map(|x| {
            x.as_i64()
                .ok_or_else(|| format!("\"{name}\": not an integer").into())
        })
        .collect()
}

fn floats(v: &Value, name: &str) -> Result<Vec<f32>, Error> {
    field(v, name)?
        .as_array()
        .ok_or(format!("\"{name}\" is not an array"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| format!("\"{name}\": not a number").into())
        })
        .collect()
}

fn shape(v: &Value) -> Result<Shape, Error> {
    match ints(v, "shape")?[..] {
        [y, x, z] => Ok((y as usize, x as usize, z as usize)),
        _ => Err("\"shape\" must be [H, W, C]".into()),
    }
}

/// A tensor's int8 quantization in the TPU's uint8 domain.
fn quant(v: &Value) -> Result<Quant, Error> {
    Ok(Quant {
        scale: num(v, "scale")? as f32,
        zero_point: int(v, "zero_point")? as i32 + 128,
    })
}

fn weights(v: &Value) -> Result<Vec<i8>, Error> {
    ints(v, "weights")?
        .into_iter()
        .map(|w| i8::try_from(w).map_err(|_| "weight outside int8".into()))
        .collect()
}

fn bias(v: &Value) -> Result<Vec<i32>, Error> {
    ints(v, "bias")?
        .into_iter()
        .map(|b| i32::try_from(b).map_err(|_| "bias outside int32".into()))
        .collect()
}

impl Graph {
    /// A graph exported by `tools/torch/edgetpu_quant.py`.
    pub fn from_torch(path: &str) -> Result<Graph, Error> {
        Graph::from_torch_json(&std::fs::read_to_string(path)?)
    }

    pub fn from_torch_json(json: &str) -> Result<Graph, Error> {
        let g: Value = serde_json::from_str(json)?;
        if g.get("format").and_then(Value::as_str) != Some("reef-graph") || int(&g, "version")? != 1
        {
            return Err("not a reef-graph version 1".into());
        }
        let gin = field(&g, "input")?;
        let input_tensor = int(gin, "tensor")? as i32;
        // per tensor: shape and quantization; reshapes alias their input
        let mut tensors: HashMap<i32, (Shape, Quant)> =
            HashMap::from([(input_tensor, (shape(gin)?, quant(gin)?))]);
        let mut alias: HashMap<i32, i32> = HashMap::new();
        let resolve = |alias: &HashMap<i32, i32>, t: i32| *alias.get(&t).unwrap_or(&t);
        let mut steps = Vec::new();
        for (i, op) in field(&g, "ops")?
            .as_array()
            .ok_or("\"ops\" is not an array")?
            .iter()
            .enumerate()
        {
            let kind = field(op, "op")?.as_str().ok_or("\"op\" is not a string")?;
            let ctx = |e: Error| -> Error { format!("op {i} ({kind}): {e}").into() };
            let ins = ints(op, "inputs")?;
            let out = int(op, "output")? as i32;
            let (out_shape, out_q) = (shape(op)?, quant(op)?);
            let arg = |k: usize| -> Result<(i32, Shape, Quant), Error> {
                let t = *ins.get(k).ok_or("missing input")? as i32;
                let (s, q) = *tensors
                    .get(&t)
                    .ok_or(format!("tensor {t} not produced yet"))?;
                Ok((t, s, q))
            };
            let input = |k: usize| -> Result<Input, Error> {
                let (t, s, q) = arg(k)?;
                Ok(Input {
                    tensor: resolve(&alias, t),
                    shape: s,
                    pad: q.zero_point as u8,
                })
            };
            let relu = || -> Result<bool, Error> {
                field(op, "relu")?
                    .as_bool()
                    .ok_or_else(|| "\"relu\" is not a bool".into())
            };
            let layer = match kind {
                "reshape" => {
                    let (t, s, q) = arg(0)?;
                    if q != out_q || s.0 * s.1 * s.2 != out_shape.0 * out_shape.1 * out_shape.2 {
                        return Err(ctx("changes quantization or size".into()));
                    }
                    alias.insert(out, resolve(&alias, t));
                    tensors.insert(out, (out_shape, out_q));
                    continue;
                }
                "conv" => {
                    let (_, (y, x, cin), q) = arg(0)?;
                    let k = int(op, "kernel")? as usize;
                    let stride = int(op, "stride")? as usize;
                    let cout = out_shape.2;
                    if (out_shape.0, out_shape.1) != (y.div_ceil(stride), x.div_ceil(stride)) {
                        return Err(ctx(format!(
                            "output {out_shape:?}: need ceil(size / stride) pixels"
                        )
                        .into()));
                    }
                    let conv = Conv {
                        cin,
                        cout,
                        k,
                        stride,
                        width: x,
                        height: y,
                        weights: weights(op)?,
                        w_scale: floats(op, "weight_scales")?,
                        bias: bias(op)?,
                        input: q,
                        output: out_q,
                        relu: relu()?,
                        pad_before: Some(int(op, "pad")? as usize),
                    };
                    Layer::Conv(fit_conv(conv).map_err(ctx)?)
                }
                "fc" => {
                    let (_, s, q) = arg(0)?;
                    let fc = Fc {
                        k: s.0 * s.1 * s.2,
                        n: out_shape.2,
                        weights: weights(op)?,
                        w_scale: floats(op, "weight_scales")?,
                        bias: bias(op)?,
                        input: q,
                        output: out_q,
                        relu: relu()?,
                    };
                    fc.check().map_err(|e| ctx(e.into()))?;
                    Layer::Fc(fc)
                }
                "add" => {
                    let ((_, sa, qa), (_, sb, qb)) = (arg(0)?, arg(1)?);
                    if sa != sb {
                        return Err(ctx("broadcasting not supported".into()));
                    }
                    Layer::Add(add_layer(sa, qa, qb, out_q, relu()?).map_err(ctx)?)
                }
                "global_avg_pool" => {
                    let (t, (y, x, c), q) = arg(0)?;
                    if q != out_q {
                        return Err(ctx("needs the same quantization in and out".into()));
                    }
                    if (y, x) == (1, 1) {
                        // the mean of one pixel: the input itself
                        alias.insert(out, resolve(&alias, t));
                        tensors.insert(out, (out_shape, out_q));
                        continue;
                    }
                    // round_half_even(mean) in the input's quantization
                    let p = GlobalAvgPool {
                        height: y,
                        width: x,
                        c,
                        center: q.zero_point,
                        mult: 1.0 / (y * x) as f32,
                        quant: q,
                    };
                    p.check().map_err(|e| ctx(e.into()))?;
                    Layer::AvgPool(p)
                }
                "max_pool" => {
                    let (_, s, q) = arg(0)?;
                    if (int(op, "kernel")?, int(op, "stride")?) != (2, 2) || q != out_q {
                        return Err(ctx("need 2x2, stride 2, the same quantization".into()));
                    }
                    Layer::MaxPool(max_pool_layer(s, q).map_err(ctx)?)
                }
                k => return Err(format!("op {i}: \"{k}\" not supported").into()),
            };
            let inputs = match &layer {
                Layer::Add(_) => vec![input(0)?, input(1)?],
                _ => vec![input(0)?],
            };
            steps.push(Step::new(layer, inputs, out, out_shape).map_err(ctx)?);
            tensors.insert(out, (out_shape, out_q));
        }
        let output = int(&g, "output")? as i32;
        let (_, output_q) = *tensors
            .get(&output)
            .ok_or("the output tensor is not produced")?;
        if steps.is_empty() {
            return Err("no layers".into());
        }
        Ok(Graph {
            steps,
            input: quant(gin)?,
            output: output_q,
            input_shape: shape(gin)?,
            input_tensor,
            output_tensor: resolve(&alias, output),
            int8_io: true,
        })
    }
}
