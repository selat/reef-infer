use std::time::Duration;

use nusb::transfer::{ControlIn, ControlOut, ControlType, Recipient, TransferError};

/// Reads a 64-bit CSR register (bRequest=0, length=8).
/// Used for registers accessed via `registers_->Read()` / `Poll()` in libedgetpu.
pub async fn read_register64(iface: &nusb::Interface, offset: u32) -> Result<u64, TransferError> {
    let result = iface
        .control_in(
            ControlIn {
                control_type: ControlType::Vendor,
                recipient: Recipient::Device,
                request: 0,
                value: (offset & 0xffff) as u16,
                index: (offset >> 16) as u16,
                length: 8,
            },
            Duration::from_millis(1000),
        )
        .await?;

    result
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| TransferError::Unknown(0))
}

/// Writes a 64-bit CSR register (bRequest=0, 8 bytes).
/// Used for registers accessed via `registers_->Write()` in libedgetpu.
pub async fn write_register64(
    iface: &nusb::Interface,
    offset: u32,
    value: u64,
) -> Result<(), TransferError> {
    iface
        .control_out(
            ControlOut {
                control_type: ControlType::Vendor,
                recipient: Recipient::Device,
                request: 0,
                value: (offset & 0xffff) as u16,
                index: (offset >> 16) as u16,
                data: &value.to_le_bytes(),
            },
            Duration::from_millis(1000),
        )
        .await
}

/// Polls `reg` until its value equals `expected`, up to `max_attempts` reads.
pub async fn poll_register(
    iface: &nusb::Interface,
    reg: u32,
    expected: u32,
    max_attempts: u32,
) -> Result<(), TransferError> {
    for _ in 0..max_attempts {
        if read_register32(iface, reg).await? == expected {
            return Ok(());
        }
    }
    Err(TransferError::Unknown(0))
}

/// Polls `reg` until `(value >> shift) & mask == expected_field`, up to `max_attempts` reads.
pub async fn poll_field(
    iface: &nusb::Interface,
    reg: u32,
    shift: u32,
    mask: u32,
    expected_field: u32,
    max_attempts: u32,
) -> Result<(), TransferError> {
    for _ in 0..max_attempts {
        let v = read_register32(iface, reg).await?;
        if (v >> shift) & mask == expected_field {
            return Ok(());
        }
    }
    Err(TransferError::Unknown(0))
}

/// Reads a 32-bit CSR register at `offset` via vendor control transfer.
/// bRequest=1, bmRequestType=0xC0, offset split across wValue/wIndex.
pub async fn read_register32(iface: &nusb::Interface, offset: u32) -> Result<u32, TransferError> {
    let result = iface
        .control_in(
            ControlIn {
                control_type: ControlType::Vendor,
                recipient: Recipient::Device,
                request: 1,
                value: (offset & 0xffff) as u16,
                index: (offset >> 16) as u16,
                length: 4,
            },
            Duration::from_millis(1000),
        )
        .await?;

    result
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| TransferError::Unknown(0))
}

pub async fn write_register32(
    iface: &nusb::Interface,
    offset: u32,
    value: u32,
) -> Result<(), TransferError> {
    let value_bytes = value.to_le_bytes();
    iface
        .control_out(
            ControlOut {
                control_type: ControlType::Vendor,
                recipient: Recipient::Device,
                request: 1,
                value: (offset & 0xffff) as u16,
                index: (offset >> 16) as u16,
                data: &value_bytes,
            },
            Duration::from_millis(1000),
        )
        .await
}
