# Pick and Place CAD Documentation

This directory contains all mechanical design files for the OSHE Pick and Place project.

Each subsystem is organized in the proceeding folders. Various STL and STEP files are available here for the current version of the pick and place machine.
Our system includes a nearly fully 3D printable toolhead, base structures, PCB clamping, pneumatic and electrical enclosure, and modified Push Pull Feeder from Mark Maker.

## Folder Structure
* **/V1-Legacy**: Baseline Ender 3 conversion (Fall 2025).
* **/V2-Active**: Current Klipper-integrated development (Spring 2026).
* `Ender-3-Full-Assembly.step`: Reference model for Ender 3 3D printer.

## Design Environment
* Autodesk Fusion 360, Solidworks used for modeling and integration
* All components exported to `.step` and `.stl` with Fusion 360 archives provided as `.f3d`
* **Master File:** `PnP_Full_Assembly_V2.step`

## Hardware Attribution
We utilize and build upon the work of the open-source community:
* **Push-Pull Feeder:** Designed by **Mark Maker**.
    * [Original Repository](https://github.com/markmaker/PushPullFeeder)
    * Our implementation includes optimization for use with 4040 v-slotted aluminum extrusion (commonly found on Ender 3), found in `/V2-Active/PushPullFeeder`.

## Fabrication Guide
* **Material:** PETG (Motor mounts, Push Pull Feeder, Structural), PLA (Toolhead, PCB Clamping, Enclosures)
*Our Pick and Place structures can be printed in either PETG or PLA, but PETG is preferred*
* **Minimal Settings:** 0.2mm layer height, 2+ perimeters, 12% Gyroid infill, tree supports recommended
*The Push Pull Feeder is recommended by Mark Maker to be printed with more specific settings. See attributed repository for more information.*
* **Hardware:** Our compoennts were printed and verified across the **Bambu Lab A1** and **Prusa Core 1** platforms with their respective slicers.
    * Standard 0.4mm nozzle recommended