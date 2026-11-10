# Per-machine vision setup checklist

Working notes for the build documentation: every step a builder has to run on **their own**
machine, in order, and when to redo it. Commands run on the Pi from `Software/pnpvision`
with `~/venv-pnp/bin/python -m pnpvision ...`. Status reflects the OSHE-PnP prototype.

| # | Step | Command / tool | Why it is per machine | Redo when | Status |
|---|---|---|---|---|---|
| 1 | Pi software | `python3 -m venv ~/venv-pnp`, `pip install --only-binary=:all: -r requirements.txt` | - | new Pi / OS | done |
| 2 | Lens calibration, each camera | `calib_capture.py` (ChArUco 7x5, 8 mm squares, 6 mm markers, DICT_4X4), copy the `.npz` into `calibration/`, then `check-lens` | every lens and sensor differs | lens refocused or swapped, camera resolution changed in crowsnest | done (down, up) |
| 3 | Undistorted Mainsail views + reticle | `undistort-down` / `undistort-up` services (`undistort_stream.py`, add `--motion` for mm ticks) | uses that machine's `.npz` and motion map | after steps 2 or 4 (restart the service) | done |
| 4 | Down-camera motion map | `calibrate-motion --cam down --step 3` on a dark ~3 mm dot, flat and **fixed**, at **PCB-top height** | camera scale, rotation and mirroring depend on the mount (ours: 0.5 deg rotated, ~59 px/mm) | camera moved or refocused | done |
| 5 | Repeatability check (diagnostic) | `repeatability --n 20` on a fixed dot; `--home-every 5` to test homing | belts, motors, frame | mechanical changes, accuracy problems | done (~0.01 mm; sensorless homing 0.1-2 mm) |
| 6 | Visual homing, teach once | `home --teach --polarity bright --thresh 230` with datum dot 0 under the crosshair; then use `home` instead of G28 | sensorless X/Y homing lands up to ~2 mm off; the taught XY **defines the machine frame** | datum or down camera moved (then redo every step below) | done |
| 6b | VISION_HOME button in Mainsail | install the `gcode_shell_command` Klipper extension (KIAUH), then the `[gcode_shell_command vision_home]` + `[gcode_macro VISION_HOME]` sections in printer.cfg (runs `scripts/vision_home.sh` in the background) | paths, user name | new Pi / repo moved | done (KIAUH v6: main menu E) Extensions > G-Code Shell Command; decline the example config) |
| 7 | Down-camera to nozzle offset | `calibrate-offset` (nozzle over a dot by eye, 3 runs); later refine by placing parts and measuring | camera and nozzle-holder mounting | camera or nozzle holder moved | done by eye on datum dot 0 (4 runs, spread +-0.04 mm): (8.056, 53.112) vs CAD (7.71, 53.04) |
| 8 | Up-camera motion map + centre | `calibrate-motion --cam up --find-center` | up-camera mount | up camera moved | pending (mount being reprinted) |
| 9 | Datum teach / reference | `datum --teach`, `datum --reference` | datum placement on the plate | datum moved | teach + measure done (dots fit rms 0.044 mm, rotation ~0 after reseat); `--reference` waits for the up camera |
| 10 | Up-camera light | 6 x WS2812 ring on a free endstop port (ours: EXP1 header, data EXP1_8 = PE15, GND EXP1_9, 5V EXP1_10; STOP/endstop ports did not pass LED data; 3.3 V data, add a 74AHCT125 level shifter if unreliable; later maybe a Pico as a second Klipper MCU), `[neopixel up_light]` + `UP_LIGHT_ON/OFF` in printer.cfg, `[lighting.up] type = "klipper_led"` in `config.toml` | free pins differ per build; endstop ports are filtered inputs and may not pass LED data | - | configured, not yet tested |

## Machine-specific settings to record
- `config.toml`: travel limits, `safe_travel_z` (larger Z = nozzle lower on this machine), crowsnest snapshot URLs, camera width/height (must match crowsnest **and** the `.npz`), light macros.
- Homing dot detector settings: a white dot on a sparkly or textured print needs a fixed threshold (`--polarity bright --thresh 230`); automatic thresholding failed there.
- Datum detector cut-offs in `config.toml` `[datum]`: `grid_threshold` (grid lines) and `dot_threshold` (white dots) differ; check both with a snapshot after reprinting or relighting the datum.
- Moving or reseating the datum moves the homing dot: re-teach visual homing (`home --teach`), then `datum --teach`; the camera-to-nozzle offset and motion map are unaffected.
- Sensorless homing (printer.cfg): with `driver_SGT 2`, homing at 20 mm/s or 0.8 A false-triggered (stopped early); 40 mm/s at 1.0 A works. Tune SGT before making homing gentler.

## Lessons worth keeping in the docs
- Calibration targets must not move: a loose paper dot slid ~0.05 mm during fast moves and showed up as a fake machine error. Tape or glue it, or use a bolted feature.
- The motion map absorbs camera rotation, mirroring and any X/Y scale difference, so cameras need no mechanical squaring.
- Undistorted images use square pixels (`square_new_K`); OpenCV's default made round dots look 24% oval.
- Addressable-LED data lines run in an unshielded harness next to steppers can corrupt colours (our white down light flickers purple). Fixes: GND wire twisted with data, 220-470 ohm series resistor, 100-1000 uF across LED power, 3.3->5 V level shifter. Matters for vision: purple light makes white features much darker in greyscale.
- The down camera is only ~11.5 mm from the work (undistorted f 681 px / 59 px per mm), so image scale changes ~9% per mm of height: the datum grid (about 0.5 mm off the calibration height) read 4% big, and a 1 deg tilt visibly makes it trapezoidal. Positions must come from centring features, not from reading them off-centre in the image.
- Measure features after centring them on the crosshair (the optical axis): there, a height difference does not shift the position.
