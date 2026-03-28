use std::time::Duration;

use nusb::transfer::{ControlIn, ControlOut, ControlType, Recipient};

/// Single-endpoint firmware image, embedded at compile time.
const FIRMWARE: &[u8] = include_bytes!("apex_latest_single_ep.bin");

// DFU spec bRequest values (DFU 1.1 §6.1)
const DFU_DNLOAD: u8 = 1;
const DFU_GETSTATUS: u8 = 3;

// DFU bState values
const DFU_STATE_IDLE: u8 = 2; // dfuIDLE
const DFU_STATE_DNLOAD_IDLE: u8 = 5; // dfuDNLOAD-IDLE

// DFU functional descriptor type
const DFU_FUNCTIONAL_DESC_TYPE: u8 = 0x21;
// DFU interface class / subclass (DFU 1.1 §4.2.3)
const DFU_CLASS: u8 = 0xFE;
const DFU_SUBCLASS: u8 = 0x01;

struct DfuInfo {
    interface_number: u8,
    transfer_size: u16,
}

/// Walk the raw configuration descriptor bytes to find:
///   - the DFU interface number (class=0xFE, subclass=0x01)
///   - the wTransferSize from the DFU Functional Descriptor (type=0x21)
fn parse_dfu_info(config_bytes: &[u8]) -> Option<DfuInfo> {
    let mut i = 0;
    let mut dfu_iface: Option<u8> = None;
    let mut transfer_size: u16 = 4096; // safe default

    while i + 1 < config_bytes.len() {
        let len = config_bytes[i] as usize;
        let typ = config_bytes[i + 1];

        if len < 2 {
            break;
        }

        match typ {
            0x04 /* Interface */ if i + 8 < config_bytes.len() => {
                let class    = config_bytes[i + 5];
                let subclass = config_bytes[i + 6];
                if class == DFU_CLASS && subclass == DFU_SUBCLASS {
                    dfu_iface = Some(config_bytes[i + 2]);
                }
            }
            t if t == DFU_FUNCTIONAL_DESC_TYPE && i + 6 < config_bytes.len() => {
                transfer_size = u16::from_le_bytes([config_bytes[i + 5], config_bytes[i + 6]]);
            }
            _ => {}
        }

        i += len;
    }

    dfu_iface.map(|n| DfuInfo {
        interface_number: n,
        transfer_size,
    })
}

/// Downloads FIRMWARE to the device using the DFU protocol (DFU 1.1).
///
/// Sequence per spec + libedgetpu usb_dfu_commands.cc:
///   for each chunk: DFU_DNLOAD → DFU_GETSTATUS (expect dfuDNLOAD-IDLE)
///   final zero-length DFU_DNLOAD → DFU_GETSTATUS (expect dfuIDLE = manifest done)
pub async fn dfu_download(device: &nusb::Device) -> Result<(), Box<dyn std::error::Error>> {
    let config = device.active_configuration()?;
    let info = parse_dfu_info(config.as_bytes())
        .ok_or("DFU interface / functional descriptor not found")?;

    tracing::debug!(
        "DFU: interface={}, transfer_size={}",
        info.interface_number, info.transfer_size
    );

    let iface = device.claim_interface(info.interface_number).await?;
    let xfer = info.transfer_size as usize;
    let mut block: u16 = 0;
    let mut offset: usize = 0;

    loop {
        let chunk_size = (FIRMWARE.len() - offset).min(xfer);
        let chunk = &FIRMWARE[offset..offset + chunk_size];

        // DFU_DNLOAD: host → device, class, interface
        iface
            .control_out(
                ControlOut {
                    control_type: ControlType::Class,
                    recipient: Recipient::Interface,
                    request: DFU_DNLOAD,
                    value: block,
                    index: info.interface_number as u16,
                    data: chunk,
                },
                Duration::from_millis(5000),
            )
            .await?;

        // DFU_GETSTATUS: device → host, 6 bytes
        let st = iface
            .control_in(
                ControlIn {
                    control_type: ControlType::Class,
                    recipient: Recipient::Interface,
                    request: DFU_GETSTATUS,
                    value: 0,
                    index: info.interface_number as u16,
                    length: 6,
                },
                Duration::from_millis(5000),
            )
            .await?;

        if st.len() < 6 {
            return Err(format!("DFU_GETSTATUS: short response ({} bytes)", st.len()).into());
        }
        let dfu_err = st[0];
        let poll_ms = u32::from_le_bytes([st[1], st[2], st[3], 0]);
        let dfu_state = st[4];

        if dfu_err != 0 {
            return Err(format!("DFU error status {dfu_err} in state {dfu_state}").into());
        }

        if chunk_size == 0 {
            // Zero-length DFU_DNLOAD sent: device manifests and returns to dfuIDLE
            if dfu_state != DFU_STATE_IDLE {
                return Err(format!("DFU unexpected state after manifest: {dfu_state}").into());
            }
            tracing::debug!(
                "DFU: download complete ({} bytes, {} blocks)",
                FIRMWARE.len(),
                block
            );
            break;
        }

        if dfu_state != DFU_STATE_DNLOAD_IDLE {
            return Err(format!("DFU unexpected state {dfu_state} after block {block}").into());
        }

        if poll_ms > 0 {
            std::thread::sleep(Duration::from_millis(poll_ms as u64));
        }

        offset += chunk_size;
        block = block.wrapping_add(1);
    }

    Ok(())
}
