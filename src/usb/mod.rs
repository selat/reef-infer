pub mod dfu;
pub mod registers;
pub mod transfer;

use std::time::Duration;

pub const APP_VID: u16 = 0x18D1;
pub const APP_PID: u16 = 0x9302;
pub const DFU_VID: u16 = 0x1A6E;
pub const DFU_PID: u16 = 0x089A;

const KNOWN_USB_IDS: &[(u16, u16)] = &[(APP_VID, APP_PID), (DFU_VID, DFU_PID)];

pub async fn list_devices() -> Vec<nusb::DeviceInfo> {
    nusb::list_devices()
        .await
        .map(|iter| {
            iter.filter(|d| KNOWN_USB_IDS.contains(&(d.vendor_id(), d.product_id())))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn open_app_device() -> Option<nusb::DeviceInfo> {
    for attempt in 0..10 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(1));
        }
        let devs = list_devices().await;
        if let Some(d) = devs.into_iter().find(|d| d.vendor_id() == APP_VID) {
            return Some(d);
        }
    }
    None
}

pub fn dump_endpoints(device: &nusb::Device) {
    let config = match device.active_configuration() {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("active_configuration failed: {e}");
            return;
        }
    };
    for iface in config.interfaces() {
        for alt in iface.alt_settings() {
            tracing::debug!(
                "interface={} alt={} class={:#04x} subclass={:#04x}",
                alt.interface_number(),
                alt.alternate_setting(),
                alt.class(),
                alt.subclass()
            );
            for ep in alt.endpoints() {
                tracing::debug!(
                    "  ep addr={:#04x} dir={:?} type={:?} max_packet={}",
                    ep.address(),
                    ep.direction(),
                    ep.transfer_type(),
                    ep.max_packet_size()
                );
            }
        }
    }
}
