# Pick and Place CAD Documentation

<p align="center">
  <img src="Media/CompleteAssemblyV2.png" alt="Complete Assembly V2" width="800">
</p>

This directory contains all mechanical design files for the OSHE Pick and Place project.

Each subsystem is organized in the proceeding folders. Various STL and STEP files are available here for the current version of the pick and place machine.
Our system includes a nearly fully 3D printable toolhead, base structures, PCB clamping, pneumatic and electrical enclosure, and modified Push Pull Feeder from Mark Maker.

## Folder Structure
* **/V1-Legacy**: Baseline Ender 3 conversion (Fall 2025).
* **/V2-Active**: Current Klipper-integrated development (Spring 2026).
* `Ender-3-Full-Assembly.step`: Reference model for Ender 3 3D printer.

## Design Environment
* Autodesk Fusion 360, Solidworks used for modeling and integration
* All components exported to `.step` and `.stl`

## V2 Directory
* **/AxisSupport**: Supporting pieces for rolling Y-axis along X-axes
    * `YAxisSupportA.step`: Mounts the Y-axis with rollers, featuring an end stop for alignment, for the side of the Y-motor (print one)
    * `YAxisSupportB.step`: Mounts the Y-axis with rollers openly (print one)
* **/Clamping**: PCB Clamping mechanism
    * `ClampingAssembly.step`: Reference assembly of clamp
    * `clampMount.step`, `clamp.step`, and `clampMountMirror.step`: Extrusion attachments for mount and clamping piece (print one of each)
    * `THandle.step`, `TSlotShaftCollar.step`, and `TSlotSpacer.step`: Printable clamping components to tension PCB. Print two of each, acquiring two compression springs.
* **/ElectricalEnclosure**: Housing for control board, raspberry pi, etc
    * `EncAssembly.step`: Reference assembly of constructed enclosure
    * Print one of each component in this folder. We utilized an Ender 3 fan and installed it on the back piece
* **/Framing**: The PnP structure and axis mounts
    * `BOT_leg.step` and `BOT_leg_MIRROR.step`: Print two of each to build idler, motor, mirrored idler, and mirrored motor structures, standard length
    * `MID_leg.step`: Print four of these for each of the four structure assemblies
    * `TOP_idler.step` and `TOP_idler_MIRROR.step`: Print one of each for the regular idler mount and mirrored idler structure
    * `TOP_motor.step` and `TOP_motor_MIRROR.step`: Print one of each for the regular motor mount and mirrored motor structure
    * *OPTIONALLY* `BOT_leg_LP.step` and `BOT_leg_LP_MIRROR.step`:  Alternative leg models for a lower profile, shorter machine
**/PneumaticEnclosure**: Housing for vacuum pump and solenoid
    * WIP
**/PushPullFeeder**: The Mark Maker, fully 3D printed tape and reel feeder (PETG)
    * `PushPullFeeder1.stl` and `PushPullFeeder2.stl`: Phase 1 and phase 2 of printing the feeder
    * `PushPullFeeder.scad`: The base code for the OpenSCAD model, venture if you dare
**/ToolHead**: Nearly fully 3D printable toolhead with Z rotation
    * Print one of everything here
    * Supports are absolutely necessary for many components
    * `Prongs.step`: Print four of each of these for the 3D printed nozzle support, recommended in PETG for flexure
    * `Tool_Head_Assembly.step`: Reference assembly of constructed toolhead
* `PnP_Full_Assembly_V2.step`: **Master File** 

## Hardware Attribution
We utilize and build upon the work of the open-source community:
* **Push-Pull Feeder:** Designed by **Mark Maker**.
    * [Original Repository](https://github.com/markmaker/PushPullFeeder)
    * Our implementation includes optimization for use with 4040 v-slotted aluminum extrusion (commonly found on Ender 3), found in `/V2-Active/PushPullFeeder`.
    * This feeder is pretty unreliable and in our testing we struggled to have it operate consistently with our current modifications.

## Fabrication Guide
* **Material:** PETG (Motor mounts, Push Pull Feeder, Structural), PLA (Toolhead, PCB Clamping, Enclosures)
*Our Pick and Place structures can be printed in either PETG or PLA, but PETG is preferred*
* **Minimal Settings:** 0.2mm layer height, 2+ perimeters, 12% Gyroid infill, tree supports recommended
*The Push Pull Feeder is recommended by Mark Maker to be printed with more specific settings. See attributed repository for more information.*
* **Hardware:** Our compoennts were printed and verified across the **Bambu Lab A1** and **Prusa Core 1** platforms with their respective slicers.
    * Standard 0.4mm nozzle recommended
