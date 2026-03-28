use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use nusb::Endpoint;
use nusb::transfer::{Bulk, In, Interrupt, TransferError};
use tracing::{debug, info, warn};

use crate::chip::chip_init;
use crate::executable_generated::platforms::darwinn::{Description, ExecutableType, Position};
use crate::model::{ChunkInfo, ExecInfo, TfliteModel};
use crate::usb::dfu::dfu_download;
use crate::usb::transfer::{
    DescriptorTag, EP_EVENTS, EP_INTERRUPTS, EP_OUTPUT_ACTIVATIONS, TpuEvent, bulk_recv,
    single_ep_send,
};
use crate::usb::{DFU_VID, open_app_device};

// ── Address patching ──────────────────────────────────────────────────────────

fn copy_u32_bits(buf: &mut [u8], offset_bit: usize, value: u32) {
    for i in 0..32 {
        let bit_pos = offset_bit + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        let bit = ((value >> i) & 1) as u8;
        buf[byte_idx] = (buf[byte_idx] & !(1 << bit_idx)) | (bit << bit_idx);
    }
}

fn patch_field(buf: &mut [u8], offset_bit: usize, position: u8, addr: u64) {
    let word = if position == 0 {
        addr as u32
    } else {
        (addr >> 32) as u32
    };
    copy_u32_bits(buf, offset_bit, word);
}

fn patch_chunk(
    chunk: &ChunkInfo,
    params_addr: u64,
    input_addr: u64,
    output_addrs: &HashMap<String, u64>,
) -> Vec<u8> {
    let mut instr = chunk.bitstream.clone();
    for fo in &chunk.field_offsets {
        let base_addr = match fo.desc {
            Description::BASE_ADDRESS_PARAMETER => params_addr,
            Description::BASE_ADDRESS_INPUT_ACTIVATION => input_addr,
            Description::BASE_ADDRESS_OUTPUT_ACTIVATION => {
                output_addrs.get(&fo.name).copied().unwrap_or(0)
            }
            Description::BASE_ADDRESS_SCRATCH => 0,
            _ => 0,
        };
        let position = (fo.position == Position::UPPER_32BIT) as u8;
        patch_field(&mut instr, fo.offset_bit as usize, position, base_addr);
    }
    instr
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Errors returned by [`Device`] methods.
#[derive(Debug)]
pub enum DeviceError {
    /// No Edge TPU device found on USB.
    NotFound,
    /// USB or transfer-level error.
    Usb(TransferError),
    /// DFU firmware download failed.
    Dfu(String),
    /// Model contains no runnable executable (EXECUTION_ONLY or STAND_ALONE).
    NoExecutable,
    /// Unexpected or malformed response from the TPU.
    Protocol,
    /// TPU did not respond within the timeout window.
    Timeout,
    /// Inference completed but the TPU produced no output buffer.
    NoOutput,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NotFound => write!(f, "no Edge TPU device found"),
            DeviceError::Usb(e) => write!(f, "USB error: {e:?}"),
            DeviceError::Dfu(e) => write!(f, "DFU error: {e}"),
            DeviceError::NoExecutable => write!(f, "model has no runnable executable"),
            DeviceError::Protocol => write!(f, "unexpected response from TPU"),
            DeviceError::Timeout => write!(f, "TPU did not respond in time"),
            DeviceError::NoOutput => write!(f, "inference produced no output"),
        }
    }
}

impl std::error::Error for DeviceError {}

impl From<TransferError> for DeviceError {
    fn from(e: TransferError) -> Self {
        DeviceError::Usb(e)
    }
}

/// Handle to an open Edge TPU device.  Stores the USB interface and
/// pre-opened endpoint queues for events and interrupts.
pub struct Device {
    iface: nusb::Interface,
    events: Endpoint<Bulk, In>,
    interrupts: Endpoint<Interrupt, In>,
    /// Prevent the nusb::Device from being dropped (interface borrows from it).
    _usb_device: nusb::Device,
}

impl Device {
    /// Open the first available Edge TPU device.
    ///
    /// If the device is in DFU mode, firmware is uploaded automatically and
    /// the device is re-enumerated in app mode.  The chip is initialised and
    /// endpoint queues are opened before returning.
    pub async fn open() -> Result<Self, DeviceError> {
        let devs = crate::usb::list_devices().await;
        let info = devs.into_iter().next().ok_or(DeviceError::NotFound)?;

        let is_dfu = info.vendor_id() == DFU_VID;

        let app_info = if is_dfu {
            let device = info.open().await.map_err(|_| DeviceError::NotFound)?;
            info!("DFU mode — uploading firmware...");
            dfu_download(&device)
                .await
                .map_err(|e| DeviceError::Dfu(e.to_string()))?;
            let _ = device.reset().await;
            drop(device);
            info!("waiting for app-mode re-enumeration...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            open_app_device().await.ok_or(DeviceError::NotFound)?
        } else {
            info
        };

        let usb_device = app_info.open().await.map_err(|_| DeviceError::NotFound)?;
        let iface = usb_device
            .claim_interface(0)
            .await
            .map_err(|_| DeviceError::NotFound)?;
        chip_init(&iface, false).await?;

        let events = iface
            .endpoint::<Bulk, In>(EP_EVENTS)
            .map_err(|_| DeviceError::NotFound)?;
        let interrupts = iface
            .endpoint::<Interrupt, In>(EP_INTERRUPTS)
            .map_err(|_| DeviceError::NotFound)?;

        info!("device ready");
        Ok(Self {
            iface,
            events,
            interrupts,
            _usb_device: usb_device,
        })
    }
}

/// State produced by [`Device::load_params`].  Keep this alive across inference
/// calls to avoid re-uploading weights for every inference.
pub struct LoadedParams {
    /// Weight bytes pinned at a stable host address (the TPU DMAs from here).
    pub params_buf: Vec<u8>,
    /// Keeps the model alive; `execution_idx` indexes into it.
    model: Arc<TfliteModel>,
    /// Index of the EXEC_ONLY / STAND_ALONE executable within `model.executables`.
    execution_idx: usize,
    /// Total output bytes for one inference.
    pub output_size: usize,
}

impl LoadedParams {
    fn execution(&self) -> &ExecInfo {
        &self.model.executables[self.execution_idx]
    }

    /// Input quantization parameters (scale, zero_point) from the first input layer.
    pub fn input_quant(&self) -> (f32, i32) {
        let l = &self.execution().input_layers[0];
        (l.dequantization_factor, l.zero_point)
    }

    /// Output quantization parameters (scale, zero_point) from the first output layer.
    pub fn output_quant(&self) -> (f32, i32) {
        let l = &self.execution().output_layers[0];
        (l.dequantization_factor, l.zero_point)
    }

    /// Number of meaningful output bytes (from the first output layer).
    pub fn output_len(&self) -> usize {
        self.execution().output_layers[0].size_bytes as usize
    }

    /// Dequantize raw output bytes using per-layer quant params.
    ///
    /// Each output layer may have a different zero_point and scale, so the raw
    /// bytes are split by layer size and dequantized independently.
    pub fn dequantize_output(&self, raw: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(raw.len());
        let mut offset = 0;
        for layer in &self.execution().output_layers {
            let len = layer.size_bytes as usize;
            let zp = layer.zero_point as f32;
            let scale = layer.dequantization_factor;
            let end = (offset + len).min(raw.len());
            for &q in &raw[offset..end] {
                out.push((q as f32 - zp) * scale);
            }
            offset += len;
            if offset >= raw.len() {
                break;
            }
        }
        out
    }
}

impl Device {
    /// Sends the PARAMETER_CACHING instructions, waits for the chip to DMA the
    /// weights into its SRAM, and returns a [`LoadedParams`] that can be reused
    /// across many [`run_inference`] calls without re-uploading the weights.
    ///
    /// For STAND_ALONE models (no PARAMETER_CACHING executable) this is a no-op
    /// that simply captures the execution executable.
    pub async fn load_params(
        &mut self,
        model: Arc<TfliteModel>,
    ) -> Result<LoadedParams, DeviceError> {
        let param_caching_idx = model
            .executables
            .iter()
            .position(|e| e.exec_type == ExecutableType::PARAMETER_CACHING);

        let execution_idx = model
            .executables
            .iter()
            .position(|e| {
                e.exec_type == ExecutableType::EXECUTION_ONLY
                    || e.exec_type == ExecutableType::STAND_ALONE
            })
            .ok_or_else(|| {
                warn!("no EXECUTION_ONLY or STAND_ALONE executable");
                DeviceError::NoExecutable
            })?;

        let param_caching = param_caching_idx.map(|i| &model.executables[i]);
        let execution = &model.executables[execution_idx];

        let output_size = execution
            .output_layers
            .iter()
            .map(|l| l.size_bytes as usize)
            .sum::<usize>()
            .max(8);

        let params_buf: Vec<u8> = param_caching
            .map(|pc| pc.parameters.clone())
            .unwrap_or_default();

        if let Some(pc) = param_caching {
            let params_addr = params_buf.as_ptr() as u64;

            let pc_instrs: Vec<Vec<u8>> = pc
                .instruction_chunks
                .iter()
                .map(|c| patch_chunk(c, params_addr, 0, &HashMap::new()))
                .collect();

            self.events.submit(self.events.allocate(512));
            self.interrupts.submit(self.interrupts.allocate(64));

            debug!(
                "sending PARAM_CACHING instructions ({} chunk(s))",
                pc_instrs.len()
            );
            for instr in pc_instrs.iter() {
                single_ep_send(&self.iface, DescriptorTag::Instructions, instr).await?;
            }

            let mut params_sent = false;
            'pc: for round in 0..20 {
                tokio::select! {
                    completion = self.events.next_complete() => {
                        completion.status?;
                        let data = completion.buffer.to_vec();
                        self.events.submit(self.events.allocate(512));
                        let ev = TpuEvent::from_bytes(&data).ok_or(DeviceError::Protocol)?;
                        debug!("load_params event[{round}]: tag={}", ev.tag);
                        match ev.tag {
                            2 => {
                                single_ep_send(&self.iface, DescriptorTag::Parameters, &params_buf).await?;
                                params_sent = true;
                            }
                            4..=7 if params_sent => {
                                debug!("PARAM_CACHING phase complete");
                                break 'pc;
                            }
                            4..=7 => {}
                            t => debug!("load_params unhandled event tag={t}"),
                        }
                    }
                    completion = self.interrupts.next_complete() => {
                        completion.status?;
                        let raw = u32::from_le_bytes(
                            completion.buffer.get(..4)
                                .and_then(|b| b.try_into().ok())
                                .unwrap_or([0u8; 4]),
                        );
                        self.interrupts.submit(self.interrupts.allocate(64));
                        debug!("load_params hw interrupt[{round}]: 0x{raw:08x}");
                    }
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {
                        warn!("load_params timeout");
                        return Err(DeviceError::Timeout);
                    }
                }
            }
        }

        debug!("load_params done: params={}B", params_buf.len());
        Ok(LoadedParams {
            params_buf,
            model,
            execution_idx,
            output_size,
        })
    }

    /// Runs one inference using previously loaded parameters.
    ///
    /// Sends freshly patched EXEC instructions (with the current input/output
    /// buffer addresses), then drives the `kInputActivations` /
    /// `kOutputActivations` event loop.  The `params_buf` address inside
    /// `loaded` is stable across calls so the chip always finds the right weights.
    pub async fn run_inference(
        &mut self,
        loaded: &LoadedParams,
        input_data: Vec<u8>,
    ) -> Result<Vec<u8>, DeviceError> {
        let exec = loaded.execution();

        // Allocate a contiguous staging buffer; give each output layer a distinct
        // address within it so the device DMAs to non-overlapping regions.
        let output_buf = vec![0u8; loaded.output_size];
        let output_base = output_buf.as_ptr() as u64;
        let mut output_addrs = HashMap::new();
        let mut byte_offset = 0usize;
        for layer in &exec.output_layers {
            output_addrs.insert(layer.name.clone(), output_base + byte_offset as u64);
            byte_offset += layer.size_bytes as usize;
        }

        // If the EXECUTION_ONLY executable carries its own parameters (e.g. SSD
        // anchor / NMS tables), patch their address and serve them on tag-2
        // requests.  Otherwise fall back to the cached weight buffer so that
        // models without inline execution params continue to work.
        let (exec_params, exec_params_addr) = if exec.parameters.is_empty() {
            (loaded.params_buf.as_slice(), loaded.params_buf.as_ptr() as u64)
        } else {
            (exec.parameters.as_slice(), exec.parameters.as_ptr() as u64)
        };

        let input_addr = input_data.as_ptr() as u64;

        let exec_instrs: Vec<Vec<u8>> = exec
            .instruction_chunks
            .iter()
            .map(|c| patch_chunk(c, exec_params_addr, input_addr, &output_addrs))
            .collect();

        self.events.submit(self.events.allocate(512));
        self.interrupts.submit(self.interrupts.allocate(64));

        debug!(
            "sending EXEC instructions ({} chunk(s))",
            exec_instrs.len()
        );
        for instr in &exec_instrs {
            single_ep_send(&self.iface, DescriptorTag::Instructions, instr).await?;
        }

        // Collect one tag-3 chunk per output layer before declaring completion.
        let num_outputs = exec.output_layers.len().max(1);
        let mut output_chunks: Vec<Vec<u8>> = Vec::with_capacity(num_outputs);

        'exec: for round in 0..30 {
            tokio::select! {
                completion = self.events.next_complete() => {
                    completion.status?;
                    let data = completion.buffer.to_vec();
                    self.events.submit(self.events.allocate(512));
                    let ev = TpuEvent::from_bytes(&data).ok_or(DeviceError::Protocol)?;
                    debug!("run_inference event[{round}]: tag={} len={}", ev.tag, ev.length);
                    match ev.tag {
                        1 => {
                            single_ep_send(&self.iface, DescriptorTag::InputActivations, &input_data).await?;
                        }
                        2 => {
                            let offset = ev.device_address.wrapping_sub(exec_params_addr) as usize;
                            let len = ev.length as usize;
                            match exec_params.get(offset..offset + len) {
                                Some(slice) => {
                                    single_ep_send(&self.iface, DescriptorTag::Parameters, slice).await?;
                                }
                                None => {
                                    warn!(
                                        "run_inference: param request out of range \
                                         (addr={:#x} len={} base={:#x} buf={}B)",
                                        ev.device_address, len, exec_params_addr, exec_params.len()
                                    );
                                    return Err(DeviceError::Protocol);
                                }
                            }
                        }
                        3 => {
                            let out = bulk_recv(&self.iface, EP_OUTPUT_ACTIVATIONS, ev.length as usize).await?;
                            output_chunks.push(out);
                            if output_chunks.len() == num_outputs {
                                break 'exec;
                            }
                        }
                        4..=7 => {}
                        t => debug!("run_inference unhandled event tag={t}"),
                    }
                }
                completion = self.interrupts.next_complete() => {
                    completion.status?;
                    let raw = u32::from_le_bytes(
                        completion.buffer.get(..4)
                            .and_then(|b| b.try_into().ok())
                            .unwrap_or([0u8; 4]),
                    );
                    self.interrupts.submit(self.interrupts.allocate(64));
                    debug!("run_inference hw interrupt[{round}]: 0x{raw:08x}");
                }
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    warn!("run_inference timeout (round {round})");
                    return Err(DeviceError::Timeout);
                }
            }
        }

        // Keep buffers alive until here so the TPU's DMA window remains valid.
        drop(output_buf);

        if output_chunks.is_empty() {
            warn!("run_inference: no output received");
            return Err(DeviceError::NoOutput);
        }

        Ok(output_chunks.into_iter().flatten().collect())
    }

    /// Runs one inference with automatic quantization/dequantization.
    ///
    /// Accepts float inputs, quantizes them to uint8 using the model's input
    /// scale/zero_point, runs the TPU inference, then dequantizes the raw
    /// uint8 output back to floats.
    pub async fn run_inference_f32(
        &mut self,
        loaded: &LoadedParams,
        input: &[f32],
    ) -> Result<Vec<f32>, DeviceError> {
        let (in_scale, in_zp) = loaded.input_quant();

        // Quantize: q = clamp(round(x / scale) + zero_point, 0, 255)
        let input_data: Vec<u8> = input
            .iter()
            .map(|&x| {
                let q = (x / in_scale).round() as i32 + in_zp;
                q.clamp(0, 255) as u8
            })
            .collect();

        let raw = self.run_inference(loaded, input_data).await?;
        let output = loaded.dequantize_output(&raw);

        Ok(output)
    }

    /// Convenience wrapper: loads params then runs one inference.
    pub async fn run_model(
        &mut self,
        model: Arc<TfliteModel>,
        input_data: Vec<u8>,
    ) -> Result<Vec<u8>, DeviceError> {
        let loaded = self.load_params(model).await?;
        self.run_inference(&loaded, input_data).await
    }

    /// Loads params once, then runs `warmup` silent inferences followed by
    /// `reps` timed inferences.  Prints min / avg / max latency.
    pub async fn bench_model(
        &mut self,
        model: Arc<TfliteModel>,
        warmup: usize,
        reps: usize,
    ) -> Result<(), DeviceError> {
        let input_size = model
            .executables
            .iter()
            .flat_map(|e| &e.input_layers)
            .map(|l| l.size_bytes as usize)
            .next()
            .unwrap_or(8);
        let input_data = vec![1u8; input_size];

        let loaded = self.load_params(model).await?;

        println!("[bench] warmup={warmup} reps={reps} input={input_size}B");

        for i in 0..warmup {
            print!("[bench] warmup {}/{}...\r", i + 1, warmup);
            self.run_inference(&loaded, input_data.clone()).await?;
        }
        println!("[bench] warmup done                    ");

        let mut times = Vec::with_capacity(reps);
        for i in 0..reps {
            let t0 = std::time::Instant::now();
            self.run_inference(&loaded, input_data.clone()).await?;
            let elapsed = t0.elapsed();
            println!(
                "[bench] rep {:>3}/{}: {:.2}ms",
                i + 1,
                reps,
                elapsed.as_secs_f64() * 1e3
            );
            times.push(elapsed);
        }

        let min = times.iter().min().unwrap();
        let max = times.iter().max().unwrap();
        let avg = times.iter().sum::<Duration>() / reps as u32;
        println!("[bench] ---");
        println!(
            "[bench] min {:.2}ms  avg {:.2}ms  max {:.2}ms",
            min.as_secs_f64() * 1e3,
            avg.as_secs_f64() * 1e3,
            max.as_secs_f64() * 1e3,
        );

        Ok(())
    }
} // impl Device
