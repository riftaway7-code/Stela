//! What the boot screen says when no GPU started: which NVIDIA GPU was
//! refused and why, in words a user can act on from another computer.
use alloc::{format, string::String, vec::Vec};
use stela_drivers::gpu::{Adapter, Vendor};
use stela_loader::host_firmware::recipes::nvidia_chip;

/// The NVIDIA generations Stela's GPU driver runs (Turing, Ampere, Ada and
/// Blackwell: every chip `recipes::nvidia_chip` names), as people know them.
pub const SUPPORTED: &str = "an NVIDIA GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU";

/// Why an NVIDIA GPU did not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The USB holds no firmware for its chip.
    MissingFirmware,
    /// Anything else; the boot log names it.
    Other,
}

/// The screen line for the NVIDIA GPU `device` at PCI `address` that did
/// not start for `reason`.
pub fn refused(address: &str, device: u16, reason: Reason) -> String {
    match (nvidia_chip(device), reason) {
        (None, _) => format!(
            "the NVIDIA GPU at {address} (device {device:04x}) is not supported: Stela needs \
             {SUPPORTED}"
        ),
        (Some(chip), Reason::MissingFirmware) => format!(
            "the USB has no firmware for this PC's NVIDIA {chip} GPU. On another computer, run \
             `stela setup host-firmware fetch --gpu {chip} --usb USB` (USB: this USB's STELA \
             volume)"
        ),
        (Some(chip), Reason::Other) => format!(
            "the NVIDIA {chip} GPU at {address} did not start; the boot log on this USB says why"
        ),
    }
}

/// How the screen names a display adapter: its maker, PCI address and
/// device ID.
pub fn adapter(adapter: &Adapter) -> String {
    let maker = match adapter.vendor {
        Vendor::Nvidia => String::from("NVIDIA"),
        Vendor::Amd => String::from("AMD"),
        Vendor::Intel => String::from("Intel"),
        Vendor::Other(id) => format!("PCI vendor {id:04x}"),
    };
    format!(
        "the {maker} display adapter at {} (device {:04x})",
        adapter.address, adapter.device
    )
}

/// The line for a display adapter Stela's GPU driver does not run, and why.
pub fn unused(found: &Adapter) -> String {
    format!(
        "{} is not used: {}",
        adapter(found),
        found.unavailable_reason()
    )
}

/// The screen line for a PC without an NVIDIA GPU, naming the display
/// adapters it has instead.
pub fn none(adapters: &[Adapter]) -> String {
    let found: Vec<String> = adapters
        .iter()
        .filter(|found| found.vendor != Vendor::Nvidia)
        .map(adapter)
        .collect();
    match found.is_empty() {
        true => format!("this PC has no NVIDIA GPU: Stela needs {SUPPORTED}"),
        false => format!(
            "this PC has no NVIDIA GPU, only {}: Stela needs {SUPPORTED}",
            found.join(" and ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_gpu_is_named_with_its_fix() {
        let missing = refused("01:00.0", 0x2484, Reason::MissingFirmware);
        assert!(missing.contains("NVIDIA ga104 GPU"), "{missing}");
        assert!(
            missing.contains("`stela setup host-firmware fetch --gpu ga104 --usb USB`"),
            "{missing}"
        );
        let other = refused("03:00.0", 0x2b85, Reason::Other);
        assert_eq!(
            other,
            "the NVIDIA gb202 GPU at 03:00.0 did not start; the boot log on this USB says why"
        );
    }

    #[test]
    fn an_older_gpu_or_none_names_what_is_supported() {
        // A GeForce GTX 1080 (Pascal, gp104).
        let pascal = refused("01:00.0", 0x1b80, Reason::MissingFirmware);
        assert_eq!(
            pascal,
            "the NVIDIA GPU at 01:00.0 (device 1b80) is not supported: Stela needs an NVIDIA \
             GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU"
        );
        assert!(none(&[]).starts_with("this PC has no NVIDIA GPU: Stela needs"));
    }

    #[test]
    fn a_pc_with_only_other_display_adapters_names_them() {
        use stela_drivers::bus::pci::Address;
        let at = |bus, vendor, device| Adapter {
            address: Address {
                segment: 0,
                bus,
                device: 0,
                function: 0,
            },
            vendor,
            device,
            class: [3, 0, 0],
        };
        let amd = at(3, Vendor::Amd, 0x744c);
        let intel = at(0, Vendor::Intel, 0xa780);
        assert_eq!(
            none(&[amd, intel]),
            "this PC has no NVIDIA GPU, only the AMD display adapter at 0000:03:00.0 (device \
             744c) and the Intel display adapter at 0000:00:00.0 (device a780): Stela needs an \
             NVIDIA GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU"
        );
        assert_eq!(
            unused(&amd),
            "the AMD display adapter at 0000:03:00.0 (device 744c) is not used: AMDGPU is not \
             ported to Stela"
        );
        assert!(adapter(&at(1, Vendor::Other(0x1af4), 0x1050)).contains("PCI vendor 1af4"));
    }
}
