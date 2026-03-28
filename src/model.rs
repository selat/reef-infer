//! Loads a compiled Edge TPU `.tflite` model and exposes the DarwiNN
//! executables needed to drive the device.
//!
//! # File format
//!
//! An Edge TPU `.tflite` file is a standard TFLite FlatBuffer (`TFL3`)
//! containing a single operator whose `custom_code` is `"edgetpu-custom-op"`.
//! The operator's `custom_options` field is a **FlexBuffer** map
//! (see `libedgetpu/tflite/custom_op_data.h`) with the following keys:
//!
//! | key | type   | meaning                                       |
//! |-----|--------|-----------------------------------------------|
//! | "1" | int    | version (currently 0)                         |
//! | "4" | bytes  | serialized DarwiNN `Package` FlatBuffer (DWN1)|
//! | "5" | int    | execution preference (NNAPI only)             |
//! | "6" | [int]  | chip version enum per executable              |
//! | "7" | [str]  | additional serialized executables             |
//!
//! The Package embeds a `MultiExecutable` (in `serialized_multi_executable`)
//! which in turn holds one or more serialized `Executable` FlatBuffers (each
//! stored as a FlatBuffer-string in `serialized_executables`).

use crate::executable_generated::platforms::darwinn::{
    Description, Executable, ExecutableType, MultiExecutable, Position, root_as_package,
};
use crate::schema_v3_generated::tflite::root_as_model;

// ── FlexBuffer binary-string helper ────────────────────────────────────────
//
// The Edge TPU compiler stores the Package FlatBuffer as a FlexBuffer
// `String` value (binary data, not UTF-8).  The standard `flexbuffers`
// crate rejects non-UTF-8 strings in `get_str()`.  We work around this by
// supplying a custom `Buffer` implementation whose `buffer_str()` does NOT
// validate UTF-8 — it just wraps the raw bytes.  We only ever call
// `.as_bytes()` on the resulting "string", so the invariant is safe.

/// Wraps `&[u8]` for use with the `flexbuffers::Buffer` trait.
struct BinaryBuffer<'a>(&'a [u8]);

impl<'a> std::ops::Deref for BinaryBuffer<'a> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0
    }
}

/// A `BufferString` that holds raw bytes but presents them as `&str` without
/// UTF-8 validation.  Only `.as_bytes()` should be called on the result.
struct BinaryStr<'a>(&'a [u8]);

impl<'a> std::ops::Deref for BinaryStr<'a> {
    type Target = str;
    fn deref(&self) -> &str {
        // SAFETY: we only ever call .as_bytes() on this value; we never
        // interpret it as actual UTF-8 text.
        unsafe { std::str::from_utf8_unchecked(self.0) }
    }
}

impl serde::Serialize for BinaryStr<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(self.0)
    }
}

impl<'a> flexbuffers::Buffer for BinaryBuffer<'a> {
    type BufferString = BinaryStr<'a>;

    fn slice(&self, range: std::ops::Range<usize>) -> Option<Self> {
        self.0.get(range).map(BinaryBuffer)
    }
    fn empty() -> Self {
        BinaryBuffer(&[])
    }
    fn buffer_str(&self) -> Result<BinaryStr<'a>, std::str::Utf8Error> {
        Ok(BinaryStr(self.0)) // no UTF-8 check
    }
}

// ── Public data model ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct TfliteModel {
    pub executables: Vec<ExecInfo>,
}

#[derive(Debug)]
pub struct ExecInfo {
    pub exec_type: ExecutableType,
    /// Weight parameters for this executable (may be empty for EXECUTION_ONLY).
    pub parameters: Vec<u8>,
    pub instruction_chunks: Vec<ChunkInfo>,
    pub input_layers: Vec<LayerInfo>,
    pub output_layers: Vec<LayerInfo>,
    pub scratch_size_bytes: i32,
}

#[derive(Debug)]
pub struct ChunkInfo {
    pub bitstream: Vec<u8>,
    pub field_offsets: Vec<OffsetInfo>,
}

#[derive(Debug)]
pub struct OffsetInfo {
    /// Bit position within the instruction bitstream to patch.
    pub offset_bit: i32,
    /// Which address type this field references.
    pub desc: Description,
    /// Whether this field holds the lower or upper 32 bits of the address.
    pub position: Position,
    /// Layer name (non-empty for INPUT_ACT / OUTPUT_ACT fields).
    pub name: String,
}

#[derive(Debug)]
pub struct LayerInfo {
    pub name: String,
    pub size_bytes: i32,
    pub zero_point: i32,
    pub dequantization_factor: f32,
}

// ── DarwiNN Package parsing ────────────────────────────────────────────────

fn parse_layer_info(
    layer: crate::executable_generated::platforms::darwinn::Layer<'_>,
) -> LayerInfo {
    let (zp, dq) = layer
        .numerics()
        .map(|n| (n.zero_point(), n.dequantization_factor()))
        .unwrap_or((0, 0.0));
    LayerInfo {
        name: layer.name().unwrap_or("").to_string(),
        size_bytes: layer.size_bytes(),
        zero_point: zp,
        dequantization_factor: dq,
    }
}

fn parse_executable(exec: Executable<'_>) -> ExecInfo {
    let parameters = exec
        .parameters()
        .map(|v| v.bytes().to_vec())
        .unwrap_or_default();

    let mut instruction_chunks = Vec::new();
    if let Some(bitstreams) = exec.instruction_bitstreams() {
        for i in 0..bitstreams.len() {
            let bs = bitstreams.get(i);
            let bitstream = bs
                .bitstream()
                .map(|v| v.bytes().to_vec())
                .unwrap_or_default();

            let mut field_offsets = Vec::new();
            if let Some(fos) = bs.field_offsets() {
                for j in 0..fos.len() {
                    let fo = fos.get(j);
                    if let Some(meta) = fo.meta() {
                        field_offsets.push(OffsetInfo {
                            offset_bit: fo.offset_bit(),
                            desc: meta.desc(),
                            position: meta.position(),
                            name: meta.name().unwrap_or("").to_string(),
                        });
                    }
                }
            }

            instruction_chunks.push(ChunkInfo {
                bitstream,
                field_offsets,
            });
        }
    }

    let input_layers = exec
        .input_layers()
        .map(|v| (0..v.len()).map(|i| parse_layer_info(v.get(i))).collect())
        .unwrap_or_default();

    let output_layers = exec
        .output_layers()
        .map(|v| (0..v.len()).map(|i| parse_layer_info(v.get(i))).collect())
        .unwrap_or_default();

    ExecInfo {
        exec_type: exec.type_(),
        parameters,
        instruction_chunks,
        input_layers,
        output_layers,
        scratch_size_bytes: exec.scratch_size_bytes(),
    }
}

// ── Public entry point ─────────────────────────────────────────────────────

/// Reads a compiled Edge TPU `.tflite` file and returns the parsed model.
pub fn load_tflite_model(path: &str) -> Result<TfliteModel, Box<dyn std::error::Error>> {
    let tflite_buf = std::fs::read(path)?;

    let model = root_as_model(&tflite_buf)?;

    if model.subgraphs().is_none() || model.subgraphs().unwrap().len() != 1 {
        return Err("Expected exactly 1 subgraph in the TFLite model".into());
    }
    if model.operator_codes().is_none() || model.operator_codes().unwrap().len() != 1 {
        return Err("Expected exactly 1 operator code in the TFLite model".into());
    }

    let opcode = model.operator_codes().unwrap().get(0);
    if opcode.custom_code() != Some("edgetpu-custom-op") {
        return Err("Expected operator code with custom_code 'edgetpu-custom-op'".into());
    }
    let subgraph = model.subgraphs().unwrap().get(0);

    if subgraph.operators().is_none() || subgraph.operators().unwrap().len() != 1 {
        return Err("Expected exactly 1 operator in the subgraph".into());
    }
    let operator = subgraph.operators().unwrap().get(0);
    let custom_options: &[u8] = operator.custom_options().map(|v| v.bytes()).unwrap_or(&[]);

    // custom_options is a FlexBuffer map (libedgetpu/tflite/custom_op_data.h).
    // Key "4" holds the serialized DarwiNN Package FlatBuffer as a byte string.
    let flex_root = flexbuffers::Reader::get_root(BinaryBuffer(custom_options))
        .map_err(|e| format!("FlexBuffer parse error: {e}"))?;
    let flex_map = flex_root.as_map();
    let key4 = flex_map.idx("4");
    let key4_str = key4.as_str();
    let package_bytes: &[u8] = key4_str.as_bytes();
    if package_bytes.is_empty() {
        return Err("FlexBuffer key '4' (serialized Package) is absent or empty".into());
    }
    let package =
        root_as_package(package_bytes).map_err(|e| format!("Package parse error: {e}"))?;

    let sme_bytes: &[u8] = package
        .serialized_multi_executable()
        .ok_or("Package.serialized_multi_executable is absent")?
        .bytes();

    // serialized_executables contains raw binary FlatBuffers stored as strings,
    // which are not valid UTF-8 — skip verification.
    // SAFETY: bytes come from the Edge TPU compiler output via the Package FlatBuffer.
    let multi_exec = unsafe { flatbuffers::root_unchecked::<MultiExecutable>(sme_bytes) };

    let se_vec = multi_exec
        .serialized_executables()
        .ok_or("MultiExecutable.serialized_executables is absent")?;
    let mut executables = Vec::new();
    for i in 0..se_vec.len() {
        let exec_bytes: &[u8] = se_vec.get(i).as_bytes();
        let exec = flatbuffers::root::<Executable>(exec_bytes)
            .map_err(|e| format!("Executable[{i}] parse error: {e}"))?;
        executables.push(parse_executable(exec));
    }

    tracing::debug!(
        "loaded {}: {} executable(s)",
        path,
        executables.len()
    );
    for (i, e) in executables.iter().enumerate() {
        tracing::debug!(
            "  [{}] type={:?}  chunks={}  params={}B  in={} out={}",
            i,
            e.exec_type,
            e.instruction_chunks.len(),
            e.parameters.len(),
            e.input_layers.len(),
            e.output_layers.len(),
        );
        for l in &e.input_layers {
            tracing::debug!(
                "      input  {:?}  {}B  zp={}  dq={}",
                l.name, l.size_bytes, l.zero_point, l.dequantization_factor
            );
        }
        for l in &e.output_layers {
            tracing::debug!(
                "      output {:?}  {}B  zp={}  dq={}",
                l.name, l.size_bytes, l.zero_point, l.dequantization_factor
            );
        }
    }

    Ok(TfliteModel { executables })
}
