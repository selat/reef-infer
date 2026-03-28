use nusb::transfer::{Bulk, In, Out, TransferError};
use tracing::error;

// ── Bulk transfer endpoints (multi-endpoint mode) ──────────────────────────
const EP_INSTRUCTIONS: u8 = 0x01; // OUT – instruction bitstream  (kInstructionsEndpoint)
pub const EP_OUTPUT_ACTIVATIONS: u8 = 0x81; // IN  – output activations
pub const EP_EVENTS: u8 = 0x82; // IN  – 16-byte bulk DMA-request events
pub const EP_INTERRUPTS: u8 = 0x83; // IN  – interrupt signals (phase completion)

/// Send raw bytes to a bulk OUT endpoint (multi-endpoint mode, no tag header).
pub async fn bulk_send(iface: &nusb::Interface, ep: u8, data: &[u8]) -> Result<(), TransferError> {
    let mut ep_out = iface.endpoint::<Bulk, Out>(ep).map_err(|e| {
        error!("Failed to open bulk OUT endpoint {ep:#04x}: {e:?}");
        TransferError::Unknown(0)
    })?;

    let mut payload = ep_out.allocate(data.len());
    payload.extend_from_slice(data);
    ep_out.submit(payload);
    let result = ep_out.next_complete().await;
    result.status?;

    Ok(())
}

/// Receive `len` bytes from bulk IN endpoint `ep`.
///
/// nusb requires the requested transfer length to be a nonzero multiple of the
/// endpoint's max packet size (512 for these bulk endpoints).  We round up and
/// trim the result back to `len`.
pub async fn bulk_recv(
    iface: &nusb::Interface,
    ep: u8,
    len: usize,
) -> Result<Vec<u8>, TransferError> {
    const MAX_PACKET: usize = 512;
    let req_len = len.div_ceil(MAX_PACKET) * MAX_PACKET;
    let mut ep_in = iface.endpoint::<Bulk, In>(ep).map_err(|e| {
        error!("Failed to open bulk IN endpoint {ep:#04x}: {e:?}");
        TransferError::Unknown(0)
    })?;
    let buf = ep_in.allocate(req_len);
    ep_in.submit(buf);
    let completion = ep_in.next_complete().await;
    completion.status?;
    let mut v = completion.buffer.into_vec();
    v.truncate(len);
    Ok(v)
}

/// Tag nibble for the 8-byte single-endpoint bulk-out header.
///
/// Matches `UsbMlCommands::DescriptorTag` in usb_ml_commands.h.
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum DescriptorTag {
    Instructions = 0,
    InputActivations = 1,
    Parameters = 2,
    OutputActivations = 3,
}

/// Sends `data` over EP 0x01 in single-endpoint mode.
///
/// Prepends the 8-byte header required by the firmware:
///   bytes 0..4 – payload length (u32 LE)
///   byte  4    – tag nibble (`DescriptorTag as u8 & 0xF`)
///   bytes 5..8 – zero
///
/// Mirrors `UsbMlCommands::WriteHeader` + `BulkOutTransfer` in
/// usb_ml_commands.cc / usb_driver.cc `ProcessIo`.
pub async fn single_ep_send(
    iface: &nusb::Interface,
    tag: DescriptorTag,
    data: &[u8],
) -> Result<(), TransferError> {
    let mut header = [0u8; 8];
    header[0..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
    header[4] = tag as u8 & 0xF;
    bulk_send(iface, EP_INSTRUCTIONS, &header).await?;
    bulk_send(iface, EP_INSTRUCTIONS, data).await
}

/// 16-byte completion event from the device (bulk IN endpoint 0x82).
///
/// Layout (usb_ml_commands.cc `AsyncReadEvent`):
///   bytes  0..8  – device virtual address (u64 LE)
///   bytes  8..12 – byte length (u32 LE)
///   byte   12    – tag nibble (0=instructions, 1=input, 2=params, 3=output)
#[derive(Debug)]
pub struct TpuEvent {
    pub device_address: u64,
    pub length: u32,
    pub tag: u8,
}

impl TpuEvent {
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 13 {
            return None;
        }
        Some(TpuEvent {
            device_address: u64::from_le_bytes(b[0..8].try_into().ok()?),
            length: u32::from_le_bytes(b[8..12].try_into().ok()?),
            tag: b[12] & 0xF,
        })
    }
}

/// Read one 16-byte completion event.
pub async fn read_event(iface: &nusb::Interface) -> Result<TpuEvent, TransferError> {
    let data = bulk_recv(iface, EP_EVENTS, 16).await?;
    TpuEvent::from_bytes(&data).ok_or(TransferError::Unknown(0))
}
