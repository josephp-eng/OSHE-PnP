# 💻 Software & Firmware Configuration

## Overview
The OSHE Pick-and-Place machine uses a modern 3D-printer software stack to manage high-precision motion control, vacuum actuation, and component indexing. 

As of **Version 2.0 (Spring 2026)**, the project has transitioned to **Klipper Firmware** managed via the **Mainsail** web interface. This configuration allows for real-time macro editing and superior hardware communication compared to traditional monolithic firmware.

---

## 🚀 The Klipper Stack (Recommended)
This is the current standard for the MTU-OSHE Pick-and-Place. It splits processing between a host computer (Raspberry Pi/PC) and the MCU (Octopro Control Board).

### 1. Host Interface: MainsailOS
* **Purpose:** Provides the web dashboard for machine operation.
* **Access:** Once installed, the machine is controlled via a local IP address in any web browser.

### 2. Control Logic: Custom Macros
The software includes specialized G-Code macros designed specifically for SMT assembly rather than 3D printing. These are located in `printer.cfg` and `macros.cfg`.
* **Spool Management:** Automated coordinate calculation for up to 5 component feeders.
* **Safety Logic:** Pre-configured "Z-Safe" heights to ensure the toolhead clears feeders and PCB clamps during rapid XY moves.
* **Vacuum Actuation:** Dedicated pin-mapping for the solenoid valve, allowing for `VACUUM_ON` and `VACUUM_OFF` commands.

### 3. Setup Instructions
1. **Flash MCU:** Use the provided Klipper firmware bin for your specific board (e.g., Creality V4.2.2 or SKR Mini).
2. **Configuration:** Upload the `printer.cfg` from this repository to your Mainsail config directory.
3. **Calibration:** Perform the **Machine Setup** and **Vacuum Calibration** steps detailed in the [Master Documentation](https://docs.google.com/document/d/1zdPMhOE8R7Szz68Ky9EoDyxVyp7iftD7q8Uco1WKPY8/edit?usp=sharing).

---

## ⚠️ Legacy: Marlin Firmware (Deprecated)
*Version 1.0 (Fall 2025) utilized Marlin. While instructions for this remain in the documentation for users with stock Ender 3 hardware, it is **not recommended** for V2.0 builds due to limitations in macro flexibility and real-time adjustment.*

## 🔍 Troubleshooting
* **TMC Driver Communication:** If all registers read zero, verify the 24V power supply to the board; Klipper will report a "Communication Failure" if the drivers are unpowered.
* **Vacuum Failure:** If the solenoid clicks but no suction occurs, check for air leaks in the 3D-printed toolhead manifold or verify the 24V-to-12V buck converter output.

---
*Refer to the Technical Information section (Page 55+) of the [Master Documentation](https://docs.google.com/document/d/1zdPMhOE8R7Szz68Ky9EoDyxVyp7iftD7q8Uco1WKPY8/edit?usp=sharing) for deep-dive code logic and coordinate offsets.*
