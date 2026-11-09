# pnpvision - vision foundation for Klipper/Moonraker pick & place

## Pi setup (Raspberry Pi OS / MainsailOS, run on the Pi, Python 3.11+)
    python3 -m venv ~/venv-pnp
    ~/venv-pnp/bin/pip install --only-binary=:all: -r requirements.txt
Always use `--only-binary=:all:`: without it pip may try to build OpenCV from source,
which runs a 1 GB Pi out of memory. Run long installs in `tmux`.

Cameras are read from crowsnest's snapshot URLs (`snapshot_url` in config.toml), so
crowsnest and the Mainsail webcam views keep running while pnpvision works. The
width/height in config.toml must match crowsnest.conf and the calibration `.npz`.
Lens calibrations come from `Archive/USB Camera Calibration Vibecode Project`
(`calib_capture.py`); copy a new `.npz` into `calibration/` after recalibrating.

Z on this machine: 0 = homed at the top, larger Z = nozzle lower
(nozzle touches the staging plate at Z ~ 50, the PCB top at Z ~ 38).

## Order of operations (from this folder, with `~/venv-pnp/bin/python` as `python`)
1. `python -m pnpvision check-lens`            (no hardware needed)
2. `python -m pnpvision snapshot --cam up`     (look at the images; tune exposure/lighting)
3. `python -m pnpvision calibrate-motion --cam down --step 1.0`   (px<->mm, axis directions)
   Use a dark ~3 mm dot on flat white paper at PCB-top height (scale changes with height).
   The fitted map also captures how the camera is rotated relative to the machine axes
   (0.5 deg on our build; any angle works), mirroring and any X/Y scale difference, so the
   camera needs no mechanical squaring. Re-run it after moving or remounting the camera.
   `python -m pnpvision repeatability --n 20`   (same dot: travel away and back, measure landing scatter;
   add `--home-every 5` to include X/Y re-homing; results saved in results/)
   **Visual homing** (X/Y homing is sensorless and lands up to ~2 mm off; one fixed dot fixes that):
   `python -m pnpvision home --teach --polarity bright --thresh 230`   (once: homed, datum dot 0 under the crosshair)
   `python -m pnpvision home`   (from then on, instead of plain G28: G28 X Y, centre on the dot,
   SET_KINEMATIC_POSITION to the taught XY; errors logged to results/visual_home_log.csv)
   The taught XY defines the machine frame for everything measured after it; teach before step 4.
4. `python -m pnpvision calibrate-motion --cam up --find-center`
5. `python -m pnpvision calibrate-offset`      (camera-to-nozzle offset; repeat 3-5x)
6. `python -m pnpvision datum --teach`       (jog down cam so the whole grid is centred and sharp)
7. `python -m pnpvision datum --reference`    (records the up camera's position in the datum frame)
8. `python -m pnpvision datum`                (any time later: re-measures, prints drift)
Tests (no hardware): `python tests/test_offline.py && python tests/test_datum.py`
