use nusb::transfer::TransferError;

use tracing::debug;

use crate::usb::registers::{
    poll_field, read_register32, read_register64, write_register32, write_register64,
};

// ── Beagle CSR offsets (from beagle_csr_offsets.h) ────────────────────────
// Top-level / SCU
const REG_OMC0_00: u32 = 0x1a000; // e-fuse data; bits [31:24] = programming revision
const REG_SCU_CTRL_0: u32 = 0x1a30c; // clock/PHY; bits [8:10] pcie_inact, [11:13] usb_inact
const REG_SCU_CTRL_2: u32 = 0x1a314; // bits [18:19] rg_gated_gcb (1=hw-gated, 2=force-on)
const REG_SCU_CTRL_3: u32 = 0x1a318; // bits [22:23] rg_force_sleep, [8:9] cur_pwr_state
const REG_GCBB_CREDIT0: u32 = 0x1907c; // CB bridge bulk credit; pulse 0xF→0 to clear

// Tile broadcast
const REG_TILE_CONFIG0: u32 = 0x48788; // 7-bit broadcast mask; 0x7F = all tiles
const REG_DEEP_SLEEP: u32 = 0x40020; // tile sleep/wake delays
const REG_IDLE: u32 = 0x4a000; // bit 31 = disable_idle

// Interrupt-related APEX registers (BeagleTopLevelInterruptManager, 32-bit r/m/w)
const REG_OMC0_D4: u32 = 0x1a0d4; // thermal warning enable   — set bit 31 (thm_warn_en)
const REG_OMC0_D8: u32 = 0x1a0d8; // thermal shutdown enable  — set bit 31 (sd_en)
const REG_SLV_ABM_EN: u32 = 0x1a500; // slave ABM enable
const REG_SLV_ERR_ISR_MASK: u32 = 0x1a558; // slave error ISR mask (write 0x3 to unmask)
const REG_MST_ABM_EN: u32 = 0x1a600; // master ABM enable
const REG_MST_ERR_ISR_MASK: u32 = 0x1a658; // master error ISR mask (write 0x3 to unmask)
const REG_RAMBIST_CTRL_1: u32 = 0x1a704; // MBIST — clear bits 22:20 (mask) & 18:16 (status W1C)
const REG_SCU_CTR_7: u32 = 0x1a33c; // boot-failure mask — clear bits 19:18; keep 17:16 = 0 (W1C)

// USB interrupt controller registers (64-bit writes; enable_all = 1 for num_interrupts=1)
const REG_FATAL_ERR_INT_CTRL: u32 = 0x4c060; // fatal_err_int_control
const REG_TOP_LEVEL_INT_0_CTRL: u32 = 0x4c070; // top_level_int_0_control
const REG_TOP_LEVEL_INT_1_CTRL: u32 = 0x4c080; // top_level_int_1_control
const REG_TOP_LEVEL_INT_2_CTRL: u32 = 0x4c090; // top_level_int_2_control
const REG_TOP_LEVEL_INT_3_CTRL: u32 = 0x4c0a0; // top_level_int_3_control

// USB interface
const REG_OUTFEED_CHUNK_LEN: u32 = 0x4c058; // outfeed chunk: 0x80 = 1 KB, 0x20 = 256 B
const REG_DESCR_EP: u32 = 0x4c148; // descriptor endpoint enable mask
const REG_EP_STATUS_CREDIT: u32 = 0x4c150; // register with transfer credits
const REG_MULTI_BO_EP: u32 = 0x4c160; // multi bulk-out mode (0=single, 1=multi)

// RunControl registers — scalar core (scalarCoreRunControl also used as poll target after reset)
const REG_SC_RUN_CTRL: u32 = 0x44018; // scalarCoreRunControl
const REG_AV_DATA_POP_RUN: u32 = 0x44158; // avDataPopRunControl
const REG_PARAM_POP_RUN: u32 = 0x44198; // parameterPopRunControl
const REG_INFEED_RUN: u32 = 0x441d8; // infeedRunControl
const REG_OUTFEED_RUN: u32 = 0x44218; // outfeedRunControl

// RunControl registers — tile datapaths (written under tileconfig0 broadcast)
const REG_TILE_OP_RUN: u32 = 0x420c0; // opRunControl
const REG_TILE_W2N_RUN: u32 = 0x42110; // wideToNarrowRunControl
const REG_TILE_N2W_RUN: u32 = 0x42150; // narrowToWideRunControl
const REG_TILE_RING_C0_RUN: u32 = 0x42190; // ringBusConsumer0RunControl
const REG_TILE_RING_C1_RUN: u32 = 0x421d0; // ringBusConsumer1RunControl
const REG_TILE_RING_PROD_RUN: u32 = 0x42210; // ringBusProducerRunControl
const REG_TILE_MESH_BUS0_RUN: u32 = 0x42250; // meshBus0RunControl
const REG_TILE_MESH_BUS1_RUN: u32 = 0x42298; // meshBus1RunControl
const REG_TILE_MESH_BUS2_RUN: u32 = 0x422e0; // meshBus2RunControl
const REG_TILE_MESH_BUS3_RUN: u32 = 0x42328; // meshBus3RunControl
// narrowToNarrowRunControl = kInvalidOffset for Beagle — not written.

#[derive(Debug)]
pub struct ChipInfo {
    /// E-fuse programming revision, read from omc0_00[31:24].
    pub efuse_revision: u8,
    /// Raw scu_ctrl_0 value (clock/PHY configuration).
    pub scu_ctrl_0: u32,
}

pub async fn read_chip_info(iface: &nusb::Interface) -> Result<ChipInfo, TransferError> {
    let omc0_00 = read_register32(iface, REG_OMC0_00).await?;
    let scu_ctrl_0 = read_register32(iface, REG_SCU_CTRL_0).await?;
    Ok(ChipInfo {
        efuse_revision: (omc0_00 >> 24) as u8,
        scu_ctrl_0,
    })
}

pub struct Credits {
    pub instructions: u32,
    pub input_activations: u32,
    pub parameters: u32,
}

pub async fn check_credits(iface: &nusb::Interface) -> Result<Credits, TransferError> {
    write_register32(iface, REG_OMC0_00, 0xffff).await?;

    let credits = read_register64(iface, REG_EP_STATUS_CREDIT).await?;

    let counter_in_bytes = 8;
    let credit_shift = 21;
    let credit_mask = (1 << credit_shift) - 1;
    let instructions = ((credits & credit_mask) * counter_in_bytes) as u32;
    let input_activations = (((credits >> credit_shift) & credit_mask) * counter_in_bytes) as u32;
    let parameters = (((credits >> (credit_shift * 2)) & credit_mask) * counter_in_bytes) as u32;
    Ok(Credits {
        instructions,
        input_activations,
        parameters,
    })
}

/// Full chip initialization sequence mirroring libedgetpu DoOpen for USB.
///
/// Steps (from beagle_top_level_handler.cc + usb_driver.cc):
///   Open → DisableHardwareClockGate → EnableReset → QuitReset →
///   EnableHardwareClockGate → InitializeChip
pub async fn chip_init(
    iface: &nusb::Interface,
    is_multi_endpoint: bool,
) -> Result<(), TransferError> {
    // ── Open() ─────────────────────────────────────────────────────────────
    let ctrl0 = read_register32(iface, REG_SCU_CTRL_0).await?;
    debug!("scu_ctrl_0 = 0x{ctrl0:08x}");
    let ctrl0_new = ctrl0 & !(0x7 << 8) & !(0x7 << 11);
    write_register32(iface, REG_SCU_CTRL_0, ctrl0_new).await?;
    debug!("scu_ctrl_0 -> 0x{ctrl0_new:08x} (cleared pcie/usb inact phy mode)");

    let ctrl2 = read_register32(iface, REG_SCU_CTRL_2).await?;
    let hw_clock_gated = (ctrl2 >> 18) & 0x3 == 0x1;
    debug!(
        "scu_ctrl_2 = 0x{ctrl2:08x}  rg_gated_gcb={} hw_clock_gated={hw_clock_gated}",
        (ctrl2 >> 18) & 0x3
    );

    // ── DisableHardwareClockGate() ─────────────────────────────────────────
    if hw_clock_gated {
        let ctrl2 = read_register32(iface, REG_SCU_CTRL_2).await?;
        let ctrl2_new = (ctrl2 & !(0x3 << 18)) | (0x2 << 18);
        write_register32(iface, REG_SCU_CTRL_2, ctrl2_new).await?;
        debug!("disable_hw_clk_gate: scu_ctrl_2 -> 0x{ctrl2_new:08x}");
    } else {
        debug!("disable_hw_clk_gate: skipped (not hw clock gated)");
    }

    // ── EnableReset() ──────────────────────────────────────────────────────
    let ctrl3 = read_register32(iface, REG_SCU_CTRL_3).await?;
    debug!(
        "enable_reset: scu_ctrl_3 = 0x{ctrl3:08x}  rg_force_sleep={} cur_pwr_state={}",
        (ctrl3 >> 22) & 0x3,
        (ctrl3 >> 8) & 0x3
    );
    if (ctrl3 >> 22) & 0x3 != 0x3 {
        let ctrl3_new = (ctrl3 & !(0x3 << 22)) | (0x3 << 22);
        write_register32(iface, REG_SCU_CTRL_3, ctrl3_new).await?;
        debug!(
            "enable_reset: scu_ctrl_3 -> 0x{ctrl3_new:08x}  waiting for cur_pwr_state=2..."
        );
        poll_field(iface, REG_SCU_CTRL_3, 8, 0x3, 0x2, 1000).await?;
        let ctrl3_after = read_register32(iface, REG_SCU_CTRL_3).await?;
        debug!("enable_reset: cur_pwr_state reached: scu_ctrl_3=0x{ctrl3_after:08x}");
        write_register32(iface, REG_GCBB_CREDIT0, 0xF).await?;
        write_register32(iface, REG_GCBB_CREDIT0, 0x0).await?;
        debug!("enable_reset: gcbb_credit0 pulsed");
    } else {
        debug!("enable_reset: skipped (already in forced sleep)");
    }

    // ── QuitReset() ────────────────────────────────────────────────────────
    let ctrl3 = read_register32(iface, REG_SCU_CTRL_3).await?;
    let ctrl3 = (ctrl3 & !(0x3 << 22)) | (0x2 << 22); // rg_force_sleep = 0b10
    let ctrl3 = ctrl3 & !(0x3 << 28); // gcb_clkdiv = 0 (500 MHz)
    let ctrl3 = ctrl3 & !(1 << 30); // axi = 250 MHz
    let ctrl3 = ctrl3 & !(1 << 31); // 8051 = 500 MHz
    write_register32(iface, REG_SCU_CTRL_3, ctrl3).await?;
    debug!("quit_reset: scu_ctrl_3 -> 0x{ctrl3:08x}  waiting for cur_pwr_state=0...");

    poll_field(iface, REG_SCU_CTRL_3, 8, 0x3, 0x0, 1000).await?;
    let ctrl3_after = read_register32(iface, REG_SCU_CTRL_3).await?;
    debug!("quit_reset: cur_pwr_state=0 reached: scu_ctrl_3=0x{ctrl3_after:08x}");

    // scalarCoreRunControl uses 64-bit Poll (registers_->Poll → Read → ReadRegister64)
    let sc_run = read_register64(iface, REG_SC_RUN_CTRL).await?;
    debug!("quit_reset: scalarCoreRunControl = 0x{sc_run:016x}  waiting for 0...");
    for _ in 0..100 {
        if read_register64(iface, REG_SC_RUN_CTRL).await? == 0 {
            break;
        }
    }
    debug!("quit_reset: scalarCoreRunControl = 0 ok");

    // idleRegister uses 64-bit Write (registers_->Write)
    write_register64(iface, REG_IDLE, 0x1).await?;
    debug!("quit_reset: idle register written");

    // tileconfig0 uses 64-bit Write + Poll
    write_register64(iface, REG_TILE_CONFIG0, 0x7F).await?;
    debug!("quit_reset: tileconfig0 written, waiting for ack...");
    for _ in 0..100 {
        if read_register64(iface, REG_TILE_CONFIG0).await? == 0x7F {
            break;
        }
    }
    debug!("quit_reset: tileconfig0 ack ok");

    // deepSleep uses 64-bit Write
    write_register64(iface, REG_DEEP_SLEEP, (30 << 8) | 2).await?;
    debug!("quit_reset: deepSleep written");

    // ── EnableHardwareClockGate() ──────────────────────────────────────────
    let ctrl2 = read_register32(iface, REG_SCU_CTRL_2).await?;
    let ctrl2_new = (ctrl2 & !(0x3 << 18)) | (0x1 << 18);
    write_register32(iface, REG_SCU_CTRL_2, ctrl2_new).await?;
    debug!("enable_hw_clk_gate: scu_ctrl_2 -> 0x{ctrl2_new:08x}");

    // ── InitializeChip() — all use 64-bit Write (registers_->Write) ────────
    write_register64(iface, REG_DESCR_EP, 0xFF).await?;
    debug!("init_chip: descr_ep = 0xFF (all descriptor events enabled)");
    if is_multi_endpoint {
        write_register64(iface, REG_MULTI_BO_EP, 1).await?;
        debug!("init_chip: multi_bo_ep = 1 (multi-endpoint mode)");
    } else {
        write_register64(iface, REG_MULTI_BO_EP, 0).await?;
        debug!("init_chip: multi_bo_ep = 0 (single-endpoint mode)");
    }
    write_register64(iface, REG_OUTFEED_CHUNK_LEN, 0x80).await?;
    debug!("init_chip: outfeed_chunk_length = 0x80");

    // ── DoRunControl(kMoveToRun) ───────────────────────────────────────────
    // Moves all scalar-core and tile datapaths from halt → run (value = 1).
    // Must happen after InitializeChip, before sending any instructions.
    // Offsets from beagle_csr_offsets.h; all _0/_1 variants are invalid for Beagle.
    const RUN: u64 = 1; // RunControl::kMoveToRun

    // Scalar core subsystems
    write_register64(iface, REG_SC_RUN_CTRL, RUN).await?;
    write_register64(iface, REG_AV_DATA_POP_RUN, RUN).await?;
    write_register64(iface, REG_PARAM_POP_RUN, RUN).await?;
    write_register64(iface, REG_INFEED_RUN, RUN).await?;
    write_register64(iface, REG_OUTFEED_RUN, RUN).await?;

    // Broadcast subsequent writes to all tiles (tileconfig0 = 0x7F).
    write_register64(iface, REG_TILE_CONFIG0, 0x7F).await?;
    for _ in 0..100 {
        if read_register64(iface, REG_TILE_CONFIG0).await? == 0x7F {
            break;
        }
    }

    // Tile datapaths (broadcast)
    write_register64(iface, REG_TILE_OP_RUN, RUN).await?;
    write_register64(iface, REG_TILE_W2N_RUN, RUN).await?;
    write_register64(iface, REG_TILE_N2W_RUN, RUN).await?;
    write_register64(iface, REG_TILE_RING_C0_RUN, RUN).await?;
    write_register64(iface, REG_TILE_RING_C1_RUN, RUN).await?;
    write_register64(iface, REG_TILE_RING_PROD_RUN, RUN).await?;
    write_register64(iface, REG_TILE_MESH_BUS0_RUN, RUN).await?;
    write_register64(iface, REG_TILE_MESH_BUS1_RUN, RUN).await?;
    write_register64(iface, REG_TILE_MESH_BUS2_RUN, RUN).await?;
    write_register64(iface, REG_TILE_MESH_BUS3_RUN, RUN).await?;
    // narrowToNarrowRunControl = kInvalidOffset for Beagle, skip.
    debug!("init_chip: DoRunControl(kMoveToRun) done");

    // ── RegisterAndEnableAllInterrupts() ──────────────────────────────────
    // Mirrors UsbDriver::RegisterAndEnableAllInterrupts() in usb_driver.cc.
    //
    // USB interrupt controllers use 64-bit writes (UsbRegisters::Write).
    // APEX-level registers use 32-bit read-modify-write (UsbRegisters::Write32).

    // fatal_error_interrupt_controller->EnableInterrupts(): enable_all = (1<<1)-1 = 1
    write_register64(iface, REG_FATAL_ERR_INT_CTRL, 1).await?;

    // top_level_interrupt_manager->EnableInterrupts():
    //   grouped controller writes 1 to each of the 4 top-level control regs
    write_register64(iface, REG_TOP_LEVEL_INT_0_CTRL, 1).await?;
    write_register64(iface, REG_TOP_LEVEL_INT_1_CTRL, 1).await?;
    write_register64(iface, REG_TOP_LEVEL_INT_2_CTRL, 1).await?;
    write_register64(iface, REG_TOP_LEVEL_INT_3_CTRL, 1).await?;

    //   DoEnableInterrupts() — BeagleTopLevelInterruptManager:

    // EnableThermalWarningInterrupt: set omc0_d4 bit 31 (thm_warn_en)
    let v = read_register32(iface, REG_OMC0_D4).await?;
    write_register32(iface, REG_OMC0_D4, v | (1 << 31)).await?;

    // EnableMbistInterrupt:
    //   rambist_ctrl_1: clear rg_mbist_int_mask (bits 22:20 = 0 → unmask)
    //                   clear rg_mbist_int_status (bits 18:16 = 0, W1C → don't toggle)
    let v = read_register32(iface, REG_RAMBIST_CTRL_1).await?;
    write_register32(iface, REG_RAMBIST_CTRL_1, v & !(0x7 << 20) & !(0x7 << 16)).await?;
    //   scu_ctr_7: clear rg_boot_failure_mask (bits 19:18 = 0 → unmask)
    //              keep pll_lock_failure (bit 16) and usb_sel_failure (bit 17) = 0 (W1C → don't clear)
    let v = read_register32(iface, REG_SCU_CTR_7).await?;
    write_register32(
        iface,
        REG_SCU_CTR_7,
        v & !(0x3 << 18) & !(1 << 17) & !(1 << 16),
    )
    .await?;

    // EnablePcieErrorInterrupt: enable slave/master ABM, unmask error ISRs
    write_register32(iface, REG_SLV_ABM_EN, 1).await?;
    write_register32(iface, REG_MST_ABM_EN, 1).await?;
    write_register32(iface, REG_SLV_ERR_ISR_MASK, 0x3).await?; // 0x3 = unmask
    write_register32(iface, REG_MST_ERR_ISR_MASK, 0x3).await?; // 0x3 = unmask

    // EnableThermalShutdownInterrupt: set omc0_d8 bit 31 (sd_en)
    let v = read_register32(iface, REG_OMC0_D8).await?;
    write_register32(iface, REG_OMC0_D8, v | (1 << 31)).await?;

    debug!("init_chip: RegisterAndEnableAllInterrupts done");

    Ok(())
}
