# Handoff: build the OSHE-PnP job generator ("slicer")

You are helping build a tool that turns KiCad PCB output into **job files** for OSHE-PnP, an
open-source pick-and-place machine. Read this brief, then `JOB_FORMAT.md` (same folder; also
at https://github.com/josephp-eng/OSHE-PnP/blob/main/Software/pnpvision/JOB_FORMAT.md once pushed).
**`JOB_FORMAT.md` is the contract.** If anything here disagrees with it, the spec wins.

## The machine, in one paragraph
Klipper firmware on a Raspberry Pi drives a CoreXY gantry with a vacuum nozzle and a rotation
stepper. A Python package, `pnpvision`, uses a down-looking camera to home, find the board by
its pads, and (later) an up-looking camera to check each part on the nozzle. A job runner in
pnpvision (not written yet) will read your job file and drive the machine live through
Moonraker. Your tool never talks to the machine.

## Your goal
A command-line tool: KiCad output in, one `oshe-pnp-job/1` JSON file per board side out.

**Inputs** (support at least the first; the second is better for pads):
1. KiCad placement file (CSV, columns `Ref,Val,Package,PosX,PosY,Rot,Side`) plus Gerbers
   (copper + outline) for pad geometry and the board outline.
2. Or the `.kicad_pcb` file directly: it has footprints, pads (shape, size, rotation,
   position) and the outline in one place, which avoids matching Gerber flashes to parts.
   KiCad's `kicad-cli` can also export placement files from it.

**Output:** files exactly as in `JOB_FORMAT.md`: board frame (outline lower-left origin,
+X right, +Y up, viewed from the placed side), one side per file, bottom side mirrored,
angles CCW-positive as seen from the placed side, unique refs, part centres, heights, and
2–4 well-chosen alignment pads.

## Known facts about the real test board
- Pos file `Test PCB-all-pos.csv`: 156 parts on both sides (92 top, 64 bottom); packages
  0201, 0402, 0603, 0805, 1206 resistors and SOIC-8; rotations include 30/60/120/150 deg.
- Its coordinates use KiCad's **page origin** (Y values are negative, about -47 to -89).
  Convert to the board frame.
- **References are not annotated**: every 0201 is literally `R0201`, etc. The tool should
  refuse duplicate refs with a clear error (tell the user to run Annotate in KiCad), with an
  optional flag to auto-number them for testing only.
- Whether KiCad mirrored X for bottom parts and how it signs bottom angles depends on the
  export options: verify with a footprint you can reason about (e.g. an asymmetric one at
  a known angle), and write a test for it.

## What NOT to do
- No machine coordinates, Z heights, feeder positions, nozzle offsets, tape orientation or
  motor direction in the job file: the runner owns all of that.
- Don't add vision logic; just choose good alignment pads (far apart, distinct, not collinear).

## Nice to have
- Height lookup per package (table in the tool, overridable from the BOM).
- Placement ordering: lowest `height_mm` first, then smallest package.
- `skip: true` for DNP parts (from the BOM or a KiCad attribute).
- Optional preview G-code for dry runs (see the spec's "preview" section; note Z is inverted
  on this machine: larger Z = nozzle lower).
- Unit tests, including a round-trip: a known footprint -> expected board-frame x, y, rot, for
  both sides.

## When you are unsure
Ask rather than guess: open questions go back to the OSHE-PnP side (the format can still
change, as a versioned `oshe-pnp-job/2`). A validator `python -m pnpvision check-job` is
planned on the pnpvision side; until it exists, follow `JOB_FORMAT.md` literally.
