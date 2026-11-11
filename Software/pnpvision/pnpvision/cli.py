"""python -m pnpvision <command>   (run from the project folder)"""
import argparse
import json
import os
import time
from functools import partial

import cv2

from . import config as C
from . import machine_state as MS
from .camera import Camera, SnapshotCamera
from .datum import DatumGeometry, measure_datum, to_datum_xy, to_machine_xy
import numpy as np
from .detect import find_circle, find_circles
from .lens import LensModel
from .moonraker import Moonraker
from .motion import MotionMap
from .servo import calibrate_motion, center_on, repeatability_test, visual_home
from .lighting import make_light, grab_lit
from .upcam import UpCam
from .eventlog import EventLog


def open_cam(cfg, role):
    c = cfg["cameras"][role]
    lens = LensModel(C.rel(cfg, c["calibration"]))
    lc = cfg.get("lighting", {}).get(role, {})
    if lc.get("on_when_opened"):             # e.g. down light: switch on before measuring
        try:
            make_light(lc, make_mr(cfg)).on()
        except Exception as e:
            raise SystemExit(f"{role} camera: could not switch its light on through Moonraker "
                             f"({e}). Is Klipper running?")
        time.sleep(lc.get("settle_s", 0.3))
    if c.get("snapshot_url"):                 # normal case: share the camera with crowsnest
        cam = SnapshotCamera(c["snapshot_url"], c["width"], c["height"], name=role).start()
    else:                                     # direct V4L2 (crowsnest must not hold the device)
        cam = Camera(c["device"], c["width"], c["height"], c["fps"],
                     exposure=c.get("exposure", 0), name=role).start()
    return cam, lens


def dot_locator(a):
    """Circle finder configured from the --rmin/--rmax/--polarity/--thresh options."""
    return partial(find_circle, r_min_px=a.rmin, r_max_px=a.rmax,
                   polarity=a.polarity, threshold=a.thresh)


def add_dot_args(p):
    p.add_argument("--rmin", type=float, default=15, help="dot min radius, px")
    p.add_argument("--rmax", type=float, default=120, help="dot max radius, px")
    p.add_argument("--polarity", default="both", choices=["both", "dark", "bright"],
                   help="dark dot on light, bright dot on dark, or either")
    p.add_argument("--thresh", type=float,
                   help="fixed 0-255 black/white cut-off instead of automatic "
                        "(e.g. --polarity bright --thresh 230 for the datum's white dots)")


def make_mr(cfg):
    m = cfg["moonraker"]
    mc = cfg["machine"]
    return Moonraker(m["url"], m["feedrate_mm_min"], mc["x_range"], mc["y_range"], mc["z_range"],
                     safe_travel_z=mc.get("safe_travel_z"))


def cmd_check_lens(cfg, a):
    for role in (["up", "down"] if a.cam == "both" else [a.cam]):
        lens = LensModel(C.rel(cfg, cfg["cameras"][role]["calibration"]))
        print(role, json.dumps(lens.health(), indent=2))


def cmd_snapshot(cfg, a):
    cam, lens = open_cam(cfg, a.cam)
    try:
        if a.cam == "up" and a.lit:
            lc = cfg["lighting"]["up"]
            f = grab_lit(cam, make_light(lc, make_mr(cfg)), None, lc["settle_s"], lc["ambient_subtract"])
        else:
            f = cam.fresh()
        cv2.imwrite(f"{a.out}_raw.png", f)
        cv2.imwrite(f"{a.out}_undist.png", lens.undistort(f))
        print("wrote", f"{a.out}_raw.png", f"{a.out}_undist.png")
    finally:
        cam.stop()


def cmd_find_dot(cfg, a):
    """One frame, no moves: list the round blobs the detector sees, with the chosen one
    marked, so --rmin/--rmax/--polarity/--thresh can be tuned before a real command."""
    cam, lens = open_cam(cfg, a.cam)
    try:
        img = lens.undistort(cam.fresh(settle_s=cfg["moonraker"]["settle_s"]))
    finally:
        cam.stop()
    found = find_circles(img, a.rmin, a.rmax, polarity=a.polarity, threshold=a.thresh)
    principal = np.array([lens.newK[0, 2], lens.newK[1, 2]])
    mpath = C.rel(cfg, cfg["cameras"][a.cam]["motion_map"])
    motion = MotionMap.load(mpath) if os.path.exists(mpath) else None
    out = img.copy()
    cv2.drawMarker(out, tuple(int(v) for v in principal), (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
    print(f"{len(found)} round blob(s), best first (crosshair at {principal.round(1)}):")
    for i, c in enumerate(found[:8]):
        d = np.array(c.center_px) - principal
        mm = f"  = {motion.offset_mm(d).round(3)} mm" if motion else ""
        print(f"  {'*' if i == 0 else ' '} centre {np.round(c.center_px, 1)}  r {c.radius_px:.1f} px  "
              f"circ {c.circularity:.3f}  from crosshair {d.round(1)} px{mm}")
        col = (0, 0, 255) if i == 0 else (255, 128, 0)
        cv2.circle(out, tuple(int(v) for v in c.center_px), int(c.radius_px), col, 2 if i == 0 else 1)
    cv2.imwrite(a.out, out)
    print("wrote", a.out, "(red = the one commands would use)")


def cmd_focus_sweep(cfg, a):
    """Up camera: step the nozzle in Z (no X/Y moves) and score sharpness around the nozzle axis."""
    mr = make_mr(cfg)
    ms = MS.load(C.rel(cfg, "calibration/machine.json"))
    x, y, z_start = mr.position()
    if np.linalg.norm(np.subtract([x, y], ms["up_cam_xy_mm"])) > 3.0:
        raise SystemExit(f"move the nozzle over the up camera first (stored centre {ms['up_cam_xy_mm']})")
    if max(a.z0, a.z1) > 40.0:
        raise SystemExit("refusing Z > 40 over the up camera (plate ~Z50, lens just under it)")
    lc = cfg["lighting"]["up"]
    light = make_light(lc, mr)
    cam, lens = open_cam(cfg, "up")
    motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["up"]["motion_map"]))
    cx, cy = lens.newK[0, 2], lens.newK[1, 2]
    r = int(a.radius / motion.mm_per_px)
    rows = []
    try:
        n = int(round(abs(a.z1 - a.z0) / a.step)) + 1
        for z in np.linspace(a.z0, a.z1, n):
            mr.move_abs(z=float(z), feedrate=300)
            img = grab_lit(cam, light, lens, lc["settle_s"], keep_on=True)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = g[max(int(cy) - r, 0):int(cy) + r, max(int(cx) - r, 0):int(cx) + r]
            sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            rows.append((float(z), sharp, float(crop.mean()), float((crop >= 250).mean() * 100)))
            print(f"Z{z:7.2f}  sharpness {sharp:9.1f}  mean {rows[-1][2]:5.1f}  saturated {rows[-1][3]:4.1f}%")
            cv2.imwrite(f"focus_z{z:.2f}.png", img)
    finally:
        cam.stop()
        mr.move_abs(z=min(z_start, a.z0, a.z1), feedrate=300)   # end at the highest of the Zs used
    best = max(rows, key=lambda t: t[1])
    print(f"sharpest at Z{best[0]:.2f} (images focus_z*.png). Nozzle left at Z{min(z_start, a.z0, a.z1):.2f}")


def cmd_calibrate_motion(cfg, a):
    mr = make_mr(cfg)
    cam, lens = open_cam(cfg, a.cam)
    settle = cfg["moonraker"]["settle_s"]
    try:
        grab = lambda: lens.undistort(cam.fresh(settle_s=settle))
        locate = dot_locator(a)
        moving = "camera" if a.cam == "down" else "target"
        if a.cam == "up":
            # The nozzle tip is only sharp low over the up camera (below safe_travel_z), so
            # jog at the current height instead of lifting before every move.
            if 2 * a.step > mr.MAX_LOW_XY_MM:
                raise SystemExit(f"--step {a.step} too big for low jogs (max {mr.MAX_LOW_XY_MM / 2} mm)")
            move = lambda dx, dy: mr.move_rel(dx, dy, stay_low=True)
            z0 = mr.position()[2]
            print(f"up camera: calibrating with the nozzle tip at Z{z0:.3f} (jogs stay at this height)")
        else:
            move = mr.move_rel
        map_path = C.rel(cfg, cfg["cameras"][a.cam]["motion_map"])
        if a.center_only:
            if a.cam != "up":
                raise SystemExit("--center-only is for --cam up")
            mmap = MotionMap.load(map_path)
            print("using the existing motion map:", round(mmap.mm_per_px, 5), "mm/px")
        else:
            mmap = calibrate_motion(grab, locate, move, a.step, moving)
            print(json.dumps(mmap.describe(), indent=2))
            mmap.save(map_path)
        if a.cam == "up" and (a.find_center or a.center_only):
            ms_path = C.rel(cfg, "calibration/machine.json")
            old = MS.load(ms_path)
            target = (lens.newK[0, 2], lens.newK[1, 2])
            res = center_on(grab, locate, move, mmap, target, max_step_mm=mr.MAX_LOW_XY_MM * 0.8)
            print("centering:", res)
            if res["converged"]:
                x, y, z = mr.position()
                upd = {"up_cam_xy_mm": [x, y], "up_cam_focus_z": z}
                print("up camera optical axis at machine XY", (x, y), "nozzle tip in focus at Z", z)
                if any(old["up_cam_xy_mm"]):
                    # Up camera and down-camera frame are both fixed to the plate, so a change here
                    # means the nozzle moved relative to the down camera, by the same amount.
                    d = np.subtract([x, y], old["up_cam_xy_mm"])
                    off = np.add(old["down_cam_offset_mm"], d)
                    print(f"change since last time: {d.round(3)} mm -> down_cam_offset_mm would be "
                          f"{off.round(3)} (now {np.round(old['down_cam_offset_mm'], 3)})")
                    if a.update_offset:
                        upd["down_cam_offset_mm"] = off.tolist()
                        print("down_cam_offset_mm updated")
                    elif a.keep_offset:
                        print("saved the new centre only; down_cam_offset_mm unchanged (the feature "
                              "changed, not the nozzle)")
                    elif np.linalg.norm(d) > 0.02:
                        print("NOT saved. True only if the nozzle moved, not the up camera: check with "
                              "`datum` first, then re-run with --update-offset to save both.")
                        return
                MS.save(ms_path, **upd)
    finally:
        cam.stop()


def cmd_calibrate_offset(cfg, a):
    """Stamp method: nozzle XY over a dark dot -> record; centre down camera on the same dot -> record."""
    mr = make_mr(cfg)
    cam, lens = open_cam(cfg, "down")
    settle = cfg["moonraker"]["settle_s"]
    try:
        mmap = MotionMap.load(C.rel(cfg, cfg["cameras"]["down"]["motion_map"]))
        input("Jog the NOZZLE directly over the centre of a dark dot (marker dot on a white label "
              "at the SAME height the down-camera motion map was calibrated at). Press Enter...")
        x1, y1, _ = mr.position()
        input("Now lift the nozzle to a safe Z (on this machine SMALLER Z = HIGHER, e.g. Z5) and "
              "jog X/Y only, roughly, so the dot is in the down camera's view. Press Enter...")
        grab = lambda: lens.undistort(cam.fresh(settle_s=settle))
        locate = dot_locator(a)
        target = (lens.newK[0, 2], lens.newK[1, 2])
        res = center_on(grab, locate, mr.move_rel, mmap, target, tol_mm=0.01)
        print("centering:", res)
        if not res["converged"]:
            raise SystemExit("did not converge; offset not saved")
        x2, y2, _ = mr.position()
        off = [x1 - x2, y1 - y2]       # camera axis position relative to nozzle axis
        MS.save(C.rel(cfg, "calibration/machine.json"), down_cam_offset_mm=off)
        print("down_cam_offset_mm =", off, "(repeat 3-5x from different spots; spread = your error)")
    finally:
        cam.stop()


def cmd_repeatability(cfg, a):
    """Down camera over a fixed dot: travel away and back N times, measure where it lands."""
    mr = make_mr(cfg)
    if a.feed:
        mr.feedrate = a.feed
    cam, lens = open_cam(cfg, "down")
    settle = cfg["moonraker"]["settle_s"]
    try:
        mmap = MotionMap.load(C.rel(cfg, cfg["cameras"]["down"]["motion_map"]))
        grab = lambda: lens.undistort(cam.fresh(settle_s=settle))
        locate = dot_locator(a)
        target = (lens.newK[0, 2], lens.newK[1, 2])
        res = center_on(grab, locate, mr.move_rel, mmap, target, tol_mm=0.01)
        print("centering on the dot:", res)
        if not res["converged"]:
            raise SystemExit("could not centre on the dot; is it in view and at the calibrated height?")
        x, y, z = mr.position()
        print(f"\nAbout to run {a.n} cycles from X{x:.2f} Y{y:.2f}: travel {a.radius[0]:g}-{a.radius[1]:g} mm "
              f"away (8 directions) and back at F{mr.feedrate:.0f}"
              + (f", re-homing X/Y (G28 X Y) every {a.home_every} cycles" if a.home_every else "") + ".")
        print(f"Every X/Y move first lifts the nozzle to Z <= {mr.safe_travel_z} "
              f"(now Z{z:.2f}); make sure nothing within that radius is taller than that clearance.")
        input("Press Enter to start (Ctrl+C to cancel)...")
        out = repeatability_test(grab, locate, mmap, mr.move_abs, mr.position, n=a.n,
                                 radius_mm=tuple(a.radius), limits=mr.limits,
                                 home=lambda: mr.gcode("G28 X Y"), home_every=a.home_every, seed=a.seed)
        out["settings"] = {"start_xyz": [x, y, z], "feedrate": mr.feedrate, "radius_mm": a.radius,
                           "home_every": a.home_every, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(json.dumps(out["summary"], indent=2))
        path = C.rel(cfg, "results/repeatability_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print("saved", path)
    finally:
        cam.stop()


def cmd_home(cfg, a):
    """Visual homing on a fixed dot (datum dot 0). --teach once; then use instead of plain G28.
    --notify: also report the result in the Mainsail console (used by the VISION_HOME macro,
    which starts this in the background and so cannot show its output)."""
    if not a.notify:
        return _home(cfg, a)
    mr = make_mr(cfg)

    def respond(msg, error=False):
        msg = " ".join(str(msg).replace('"', "'").split())[:250]
        try:
            mr.gcode(f'RESPOND {"TYPE=error " if error else ""}MSG="{msg}"')
        except Exception:
            pass
    try:
        respond("VISION_HOME: " + _home(cfg, a))
    except BaseException as e:            # SystemExit too: those carry the useful message
        respond(f"VISION_HOME FAILED: {e}", error=True)
        raise


def _home(cfg, a):
    mr = make_mr(cfg)
    ms_path = C.rel(cfg, "calibration/machine.json")
    cam, lens = open_cam(cfg, "down")
    settle = cfg["moonraker"]["settle_s"]
    try:
        motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["down"]["motion_map"]))
        grab = lambda: lens.undistort(cam.fresh(settle_s=settle))
        target = (lens.newK[0, 2], lens.newK[1, 2])
        if a.teach:
            if not mr.is_homed():
                raise SystemExit("home the machine first (G28), then jog the homing dot under the crosshair")
            locate = dot_locator(a)
            res = center_on(grab, locate, mr.move_rel, motion, target, tol_mm=0.01, max_iter=8)
            print("centering:", res)
            if not res["converged"]:
                raise SystemExit("could not centre on the dot; check --polarity/--thresh with `snapshot`")
            x, y, _ = mr.position()
            MS.save(ms_path, visual_home={"xy": [x, y], "rmin": a.rmin, "rmax": a.rmax,
                                          "polarity": a.polarity, "thresh": a.thresh,
                                          "taught": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"taught homing dot at X{x:.4f} Y{y:.4f} (saved in {ms_path}).\n"
                  "This XY now DEFINES the machine frame: positions measured later (up camera, "
                  "feeders, boards) are only valid against it, so don't re-teach casually.")
            return f"taught homing dot at X{x:.4f} Y{y:.4f}"
        vh = MS.load(ms_path).get("visual_home")
        if not vh:
            raise SystemExit("run `home --teach` first")
        locate = partial(find_circle, r_min_px=vh["rmin"], r_max_px=vh["rmax"],
                         polarity=vh["polarity"], threshold=vh["thresh"])
        if a.no_g28:
            home = None
        elif mr.is_homed():
            home = lambda: mr.gcode("G28 X Y")
        else:
            home = lambda: mr.gcode("G28")         # first home after power-up: Z too

        def set_xy(x, y):
            z = mr.position()[2]
            mr.gcode(f"SET_KINEMATIC_POSITION X={x:.4f} Y={y:.4f} Z={z:.4f}")

        out = visual_home(grab, locate, motion, target, mr.move_abs, mr.position, set_xy,
                          vh["xy"], home=home)
        d = locate(grab())                          # check: no move, dot should sit on target
        check = "check: dot not found"
        if d is not None:
            chk = motion.offset_mm(np.array(d.center_px) - target)
            print(f"check after reset: dot {np.linalg.norm(chk):.4f} mm from the crosshair")
            check = f"check {np.linalg.norm(chk):.4f} mm"
        path = C.rel(cfg, "results/visual_home_log.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("time,g28,homing_error_x_mm,homing_error_y_mm\n")
            e = out["homing_error_mm"]
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{home is not None},{e[0]:.4f},{e[1]:.4f}\n")
        return (f"OK, frame was off X{e[0]:+.3f} Y{e[1]:+.3f} mm, reset to "
                f"X{vh['xy'][0]:.3f} Y{vh['xy'][1]:.3f}, {check}")
    finally:
        cam.stop()


def cmd_datum(cfg, a):
    """--teach: remember where the grid is.  --reference: record up-cam position in datum frame.
    (no flag): re-measure the datum and report drift of the up camera's machine position."""
    mr = make_mr(cfg)
    cam, lens = open_cam(cfg, "down")
    ms_path = C.rel(cfg, "calibration/machine.json")
    ms = MS.load(ms_path)
    settle = cfg["moonraker"]["settle_s"]
    try:
        if a.teach:
            input("Jog the down camera so the whole datum GRID is centred, sharp and well lit. "
                  "Press Enter...")
            x, y, z = mr.position()
            MS.save(ms_path, grid_view_xyz=[x, y, z])
            print("saved grid view", (x, y, z))
            return
        if not ms["grid_view_xyz"]:
            raise SystemExit("run `datum --teach` first")
        gx, gy, gz = ms["grid_view_xyz"]
        motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["down"]["motion_map"]))
        g = DatumGeometry.from_cfg(cfg["datum"])
        off = np.array(ms["down_cam_offset_mm"], float)
        principal = np.array([lens.newK[0, 2], lens.newK[1, 2]])

        def view(x, y):
            # XY only: the down camera isn't on Z, so lowering the nozzle never helps the
            # view and could put it into the bin/clamps (it sits ~53 mm from the camera).
            mr.move_abs(x=x, y=y)
            return lens.undistort(cam.fresh(settle_s=settle))

        def to_m(px, x, y):
            return np.array([x, y]) + off + motion.offset_mm(np.array(px) - principal)

        r_px = g.dot_diameter_mm / 2 / motion.mm_per_px
        pose = measure_datum(view, to_m, g, (gx, gy), off, cfg["datum"]["expected_rotation_deg"],
                             r_px, principal, grid_threshold=cfg["datum"].get("grid_threshold"),
                             dot_threshold=cfg["datum"].get("dot_threshold"))
        print(f"datum rotation {pose['rot_deg']:.3f} deg, scale {pose['s']:.5f}, "
              f"fit rms {pose['rms_mm']:.4f} mm, origin at machine XY {pose['t'].round(3)}")
        print("grid-only fit:", {k: round(v, 4) for k, v in pose["grid_only"].items()},
              "(grid read off ONE image: its scale depends on image scale at the datum's height)")
        print("dots-only fit:", {k: round(v, 4) for k, v in pose["dots_only"].items()},
              "(each dot centred: scale = machine mm vs design mm, independent of height)")
        dm = pose["dots_mm_measured"]
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            design = float(np.linalg.norm(np.subtract(g.dots_mm[j], g.dots_mm[i])))
            meas = float(np.linalg.norm(dm[j] - dm[i]))
            print(f"  dot {i}-{j}: measured {meas:.3f} mm, design {design:.3f} mm ({(meas / design - 1) * 100:+.2f}%)")
        up = np.array(ms["up_cam_xy_mm"], float)
        if a.reference:
            if not up.any():
                raise SystemExit("measure up_cam_xy_mm first (calibrate-motion --cam up --find-center)")
            MS.save(ms_path, up_cam_in_datum_mm=to_datum_xy(pose, up).tolist())
            print("recorded up camera at datum XY", to_datum_xy(pose, up).round(3))
        elif ms["up_cam_in_datum_mm"]:
            pred = to_machine_xy(pose, ms["up_cam_in_datum_mm"])
            print("up camera machine XY now", pred.round(3), "| stored", up.round(3),
                  "| drift (mm)", (pred - up).round(3))
    finally:
        cam.stop()


def cmd_light_test(cfg, a):
    """Flash the up light N times, save one lit and one dark frame, report brightness."""
    lc = cfg["lighting"]["up"]
    light = make_light(lc, make_mr(cfg))
    cam, _ = open_cam(cfg, "up")
    try:
        for n in range(a.n):
            light.off(); dark = cam.fresh(settle_s=lc["settle_s"])
            light.on(); lit = cam.fresh(settle_s=lc["settle_s"])
            print(f"flash {n + 1}: mean dark {dark.mean():.1f}  mean lit {lit.mean():.1f}")
        light.off()
        cv2.imwrite("light_dark.png", dark); cv2.imwrite("light_lit.png", lit)
        print("wrote light_dark.png light_lit.png  (lit should be clearly brighter and not saturated)")
    finally:
        light.off()
        cam.stop()


def cmd_inspect_up(cfg, a):
    """Measure whatever is on the nozzle over the up camera (jog there first)."""
    mr = make_mr(cfg)
    lc = cfg["lighting"]["up"]
    cam, lens = open_cam(cfg, "up")
    try:
        motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["up"]["motion_map"]))
        up = UpCam(cam, lens, make_light(lc, mr), motion, settle_s=lc["settle_s"],
                   ambient_subtract=lc["ambient_subtract"], keep_on=lc["keep_on"],
                   search_radius_mm=lc["search_radius_mm"])
        img = up.frame()
        cv2.imwrite("inspect_up.png", img)
        exp = tuple(a.size) if a.size else None
        res = up.inspect(exp, a.tol, EventLog(C.rel(cfg, cfg["log"]["db"])), img=img)
        print(json.dumps(res, indent=2), "\n(saved inspect_up.png)")
    finally:
        cam.stop()


def cmd_place_test(cfg, a):
    """You centre the down camera on the first part; it picks, inspects, places on the next
    target and measures, then finds the next part itself (looks like the first one)."""
    from .placetest import PartFinder, place_test, summarize
    mr = make_mr(cfg)
    ms_path = C.rel(cfg, "calibration/machine.json")
    ms = MS.load(ms_path)
    if not any(ms["up_cam_xy_mm"]) or not any(ms["down_cam_offset_mm"]):
        raise SystemExit("up camera centre / down-camera offset not calibrated yet")
    if a.inspect_z is None:
        a.inspect_z = ms.get("up_cam_focus_z") or 34.5
    for name, z in (("pick", a.pick_z), ("place", a.place_z), ("inspect", a.inspect_z)):
        if not (mr.safe_travel_z < z <= 40.0):
            raise SystemExit(f"--{name}-z {z} looks wrong (expected between Z{mr.safe_travel_z} and Z40)")
    step = np.array(a.step, float)
    targets = [np.array(a.target, float) + i * step for i in range(a.n)]
    lo, hi = np.min(targets, 0), np.max(targets, 0)
    wd = cfg["cameras"]["down"].get("working_distance_mm", 11.5)
    x0, y0, _ = mr.position()
    first = np.array([x0, y0])
    print(f"first part: the one under the down camera now, at {np.round(first, 3).tolist()}; the rest "
          f"are found automatically within the camera view from there (about +-12 mm X, +-6 mm Y); "
          f"part {a.size[0]}x{a.size[1]}x{a.thickness} mm")
    print(f"{a.n} targets from {np.round(targets[0], 3).tolist()} every {a.step} mm "
          f"(last {np.round(targets[-1], 3).tolist()}); box X {lo[0]:.2f}..{hi[0]:.2f} Y {lo[1]:.2f}..{hi[1]:.2f}")
    print(f"pick Z{a.pick_z}, place Z{a.place_z}, inspect Z{a.inspect_z}; up-camera correction "
          f"{'ON' if a.correct else 'OFF'}; accel {a.accel} mm/s^2; down_cam_offset {ms['down_cam_offset_mm']}")
    print("Visual home (VISION_HOME) must have been done since the last re-home, and the up light left on.")
    if input("Start? [y/N] ").strip().lower() != "y":
        return
    old_accel = mr.status("toolhead").get("max_accel")
    down_cam, down_lens = open_cam(cfg, "down")
    up_cam, up_lens = open_cam(cfg, "up")
    lc = cfg["lighting"]["up"]
    settle = cfg["moonraker"]["settle_s"]
    try:
        if a.accel:
            mr.gcode(f"SET_VELOCITY_LIMIT ACCEL={a.accel}")
        down_motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["down"]["motion_map"]))
        up_motion = MotionMap.load(C.rel(cfg, cfg["cameras"]["up"]["motion_map"]))
        up = UpCam(up_cam, up_lens, make_light(lc, mr), up_motion, settle_s=lc["settle_s"],
                   ambient_subtract=lc["ambient_subtract"], keep_on=True,
                   search_radius_mm=lc["search_radius_mm"])
        log = EventLog(C.rel(cfg, cfg["log"]["db"]))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        img_dir = f"place_test_{stamp}"
        os.makedirs(img_dir, exist_ok=True)
        shots = [0]

        def save(name, img):
            cv2.imwrite(os.path.join(img_dir, name + ".png"), img)

        def up_inspect(size):
            img = up.frame()
            shots[0] += 1
            save(f"up{shots[0]}", img)
            return up.inspect(tuple(size), a.tol, log, img=img)

        down_grab = lambda: down_lens.undistort(down_cam.fresh(settle_s=settle))
        principal = np.array([down_lens.newK[0, 2], down_lens.newK[1, 2]])
        k_height = wd / (wd - a.thickness)
        finder = PartFinder(mr, down_grab, down_motion, principal, tuple(a.size), k_height, first,
                            avoid=targets)
        save("first_part_template", finder.template)
        results = place_test(
            mr, down_grab, down_motion, principal, up_inspect, ms, finder, targets,
            size_mm=tuple(a.size), thickness_mm=a.thickness, working_distance_mm=wd,
            pick_z=a.pick_z, place_z=a.place_z, inspect_z=a.inspect_z,
            vac_ms=a.vac_ms, release_ms=a.release_ms, correct=a.correct, save=save)
        summary = summarize(results, ms["down_cam_offset_mm"], a.correct)
        print("\n" + json.dumps(summary, indent=2))
        path = C.rel(cfg, f"results/place_test_{stamp}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"settings": vars(a) | {"targets": [t.tolist() for t in targets]},
                       "machine": ms, "results": results, "summary": summary}, f, indent=2, default=str)
        print("saved", path, f"(images in {img_dir}/)")
        if a.apply_offset and summary.get("ok", 0) >= 3:
            MS.save(ms_path, down_cam_offset_mm=summary["suggested_down_cam_offset_mm"])
            print("down_cam_offset_mm updated to", summary["suggested_down_cam_offset_mm"])
        elif a.apply_offset:
            print("not applying the offset: need at least 3 good cycles")
    finally:
        try:
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        except Exception as e:
            print("could not lift the nozzle:", e)
        if a.accel and old_accel:
            mr.gcode(f"SET_VELOCITY_LIMIT ACCEL={old_accel}")
        down_cam.stop()
        up_cam.stop()


def main():
    ap = argparse.ArgumentParser(prog="pnpvision")
    ap.add_argument("--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-lens"); p.add_argument("--cam", default="both", choices=["up", "down", "both"])
    p = sub.add_parser("snapshot"); p.add_argument("--cam", required=True, choices=["up", "down"]); p.add_argument("--out", default="snap"); p.add_argument("--lit", action="store_true", help="(up) use the light rig")
    p = sub.add_parser("find-dot", help="no moves: show which round blobs the detector finds")
    p.add_argument("--cam", required=True, choices=["up", "down"])
    p.add_argument("--out", default="find_dot.png")
    add_dot_args(p)
    p = sub.add_parser("focus-sweep", help="up cam: step Z over the camera, score sharpness")
    p.add_argument("--z0", type=float, required=True, help="first Z (smaller = higher)")
    p.add_argument("--z1", type=float, required=True, help="last Z (max 40)")
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=2.0, help="scored area around the nozzle axis, mm")
    p = sub.add_parser("calibrate-motion")
    p.add_argument("--cam", required=True, choices=["up", "down"])
    p.add_argument("--step", type=float, default=1.0, help="jog size, mm")
    add_dot_args(p)
    p.add_argument("--find-center", action="store_true", help="(up) also record nozzle-over-camera XY")
    p.add_argument("--center-only", action="store_true",
                   help="(up) keep the motion map, only re-centre (e.g. after the nozzle was remounted)")
    p.add_argument("--update-offset", action="store_true",
                   help="(up) also shift down_cam_offset_mm by the change in the centre")
    p.add_argument("--keep-offset", action="store_true",
                   help="(up) save the new centre but leave down_cam_offset_mm alone (use when you "
                        "switch to a different feature, e.g. the real tip instead of the holder face)")
    p = sub.add_parser("calibrate-offset")
    add_dot_args(p)

    p = sub.add_parser("repeatability", help="down cam over a dot: go away and back N times, measure landing scatter")
    p.add_argument("--n", type=int, default=20, help="cycles")
    p.add_argument("--radius", type=float, nargs=2, default=[30.0, 80.0], metavar=("MIN", "MAX"),
                   help="how far away each cycle travels, mm")
    p.add_argument("--home-every", type=int, default=0, help="re-home X/Y every N cycles (0 = never)")
    p.add_argument("--feed", type=float, help="feedrate mm/min (default: config.toml)")
    p.add_argument("--seed", type=int, help="random seed for the travel distances")
    add_dot_args(p)

    p = sub.add_parser("home", help="visual homing: G28 X Y, then re-zero X/Y on the homing dot")
    p.add_argument("--teach", action="store_true",
                   help="record the dot under the crosshair as the homing reference (once)")
    p.add_argument("--no-g28", action="store_true", help="skip G28: only re-reference on the dot")
    p.add_argument("--notify", action="store_true", help="also report the result in the Mainsail console")
    add_dot_args(p)

    p = sub.add_parser("light-test"); p.add_argument("--n", type=int, default=3)
    p = sub.add_parser("inspect-up")
    p.add_argument("--size", type=float, nargs=2, metavar=("L", "W"), help="expected part body, mm")
    p.add_argument("--tol", type=float, default=0.3, help="allowed size error per side, mm")
    p = sub.add_parser("place-test", help="centre the down camera on a part first; then pick, inspect, "
                                          "place, measure, and find the next parts automatically")
    p.add_argument("--target", type=float, nargs=2, required=True, metavar=("X", "Y"),
                   help="first place target, down camera centred on it (toolhead XY)")
    p.add_argument("--step", type=float, nargs=2, default=[4.0, 0.0], metavar=("DX", "DY"),
                   help="target spacing, mm (default 4 0 = along +X)")
    p.add_argument("--n", type=int, default=5, help="parts to place (= targets)")
    p.add_argument("--size", type=float, nargs=2, default=[2.0, 1.25], metavar=("L", "W"))
    p.add_argument("--thickness", type=float, default=0.85, help="part height, mm (for the down-camera scale)")
    p.add_argument("--tol", type=float, default=0.3, help="allowed size error per side, mm")
    p.add_argument("--pick-z", type=float, default=37.8)
    p.add_argument("--place-z", type=float, default=37.7)
    p.add_argument("--inspect-z", type=float, help="default: up_cam_focus_z from calibration (else 34.5)")
    p.add_argument("--vac-ms", type=int, default=300, help="wait after VALVE_ON before lifting")
    p.add_argument("--release-ms", type=int, default=500, help="wait after VALVE_OFF before lifting")
    p.add_argument("--accel", type=float, default=2000, help="max accel during the test (0 = leave)")
    p.add_argument("--correct", action="store_true", help="apply the up-camera offset when placing")
    p.add_argument("--apply-offset", action="store_true",
                   help="save the suggested down_cam_offset_mm (needs >= 3 good cycles)")
    p = sub.add_parser("datum")
    p.add_argument("--teach", action="store_true"); p.add_argument("--reference", action="store_true")

    a = ap.parse_args()
    cfg = C.load(a.config)
    {"check-lens": cmd_check_lens, "snapshot": cmd_snapshot, "find-dot": cmd_find_dot, "focus-sweep": cmd_focus_sweep, "place-test": cmd_place_test,
     "calibrate-motion": cmd_calibrate_motion, "calibrate-offset": cmd_calibrate_offset, "datum": cmd_datum,
     "repeatability": cmd_repeatability, "home": cmd_home,
     "light-test": cmd_light_test, "inspect-up": cmd_inspect_up}[a.cmd](cfg, a)
