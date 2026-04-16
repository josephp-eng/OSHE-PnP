# OSHE Pick and Place (V2.0)

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-OSHW-blue)
![Platform](https://img.shields.io/badge/Platform-Klipper-orange)

<div align="center">

![Physical Model](https://github.com/josephp-eng/OSHE-PnP/blob/main/Media/setup.jpg?raw=true)

</div>

## 🎯 Summary

The **OSHE Pick-and-Place** is a low-cost, open-source automated PCB assembly system developed at Michigan Technological University. While industrial SMT solutions are often financially inaccessible for small labs and hobbyists, this project provides a modular, 3D-printable framework optimized for performance and ease of assembly. 

The system leverages the mechanical foundations of a Creality Ender 3, extending its capabilities into a dedicated manufacturing tool. By utilizing a vacuum-based toolhead and precision motion control, this project bridges the gap between manual prototyping and industrial production.

### 🛠️ Core Technical Achievements
* **Precision Motion Control:** Migrated to **Klipper Firmware** via Mainsail OS, allowing for higher-speed pathing and web-based machine management.
* **Advanced Toolhead Design:** A custom-engineered 3D-printed assembly featuring full 360° rotation and an integrated vacuum pick system.
* **Expanded Work Envelope:** Features a 370 x 480 mm footprint and support for PCBs up to 250 x 250 mm.
* **Safety-First Integration:** Adheres to **ISO-12100 standards**, featuring dedicated emergency stop (E-Stop) hardware and passive shielding.

### 📈 Current Project Status
The Spring 2026 revision (V2.0) has successfully transitioned the project into a functional integration phase. The machine currently supports SMD components (0804 and larger). Future development focuses on computer vision integration and automated reel-loading.
This project currently exists separate from the OpenPnP architecture. It was not designed around their suite of software.

---

## 📂 Project Documentation

> [!IMPORTANT]
> For the complete technical breakdown, including assembly guides and safety protocols, refer to the full master document below.

* **[📄 Full Technical Manual](https://docs.google.com/document/d/1zdPMhOE8R7Szz68Ky9EoDyxVyp7iftD7q8Uco1WKPY8/edit?usp=sharing)** - Detailed assembly, electrical, and safety documentation.
* **[📦 Bill of Materials (BOM)](./link-to-bom)** - Comprehensive list of hardware, electronics, and 3D-printed parts.
* **[⚙️ Configuration Files](Software/Klipper)** - `printer.cfg` and macro definitions for Klipper.
* **[📐 CAD & Model Files](CAD)** - Full assembly models and printable components.

---

## 📋 Repository Structure
* `/CAD`: STEP and STL files for the machine and custom components.
* `/Software`: Klipper configurations and legacy Marlin firmware.
* `/Archive`: Legacy V1 designs and research notes.
* `/Media`: Renders and isometric views of models.

## 💻 Getting Started
1. Review the [Our Documentation](https://docs.google.com/document/d/1zdPMhOE8R7Szz68Ky9EoDyxVyp7iftD7q8Uco1WKPY8/edit?usp=sharing) for the full build guide. Follow along with the 3D printing settings outlined in [CAD]([link](https://github.com/OSHE-Github/OSHE-PnP/tree/main/CAD))
2. Clone this repo to access the latest Klipper `printer.cfg`.
3. Join the conversation on the [OSHE Website]([link](https://oshe.io/)) and check out other OSHE projects at our [enterprise GitHub](https://github.com/OSHE-Github)

---
*Developed under the Michigan Technological University Open Source Hardware Enterprise (OSHE)*