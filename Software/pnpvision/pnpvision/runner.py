"""Job runner: place an oshe-pnp-job/1 file on the aligned board, parts from the feeders.

Per placement:
  tip change?  -> park, ask you to fit the tip, re-centre the new tip over the up camera
  pick         -> feeder ready (servo stroke done / you advanced the manual track), lower onto the
                  part at the feeder's pick XY, vacuum, lift; start the next stroke right away
  inspect      -> up camera: empty = feed again and retry (up to `retries`); wrong size = reject
  turn         -> to job rot + board rotation (chips: shortest way mod 180; parts with a pin 1:
                  coarse turn from the feeder's tape angle, camera only fine-corrects), re-inspect
  place        -> board XY + down_cam_offset - part offset on the nozzle, Z from the tip and
                  the part height, release, slow lift
  verify       -> (optional) up camera again: the nozzle must be empty

Hardware is injected (callables) so the whole job runs against a simulator in the tests. All
positions are toolhead XY; board points are "down camera centred" (BoardPose), feeder pick
positions are "nozzle over the part" (taught with the nozzle; the down camera can't reach them).
"""
import time

import numpy as np

from .parts import MAX_FINE_DEG, PackageError, feeder_part_top_z, package_info, tip_for
from .rotation import rotate_xy, shortest_turn


class JobError(RuntimeError):
    pass


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def plan_job(job, feeders, tips, packages=None, only=None, need_feeders=True):
    """Resolve package, tip and feeder for every placement, grouped by tip (stable, in the order
    the tips first appear, so each tip goes on once). Returns (steps, problems)."""
    steps, problems = [], []
    for p in job["placements"]:
        if p.get("skip") or (only and p["ref"] not in only):
            continue
        try:
            pkg = package_info(p["package"], packages)
        except PackageError as e:
            problems.append(f"{p['ref']}: {e}")
            continue
        tip = tip_for(p, pkg)
        fd = feeders.for_placement(p) if feeders is not None else None
        if need_feeders:
            if not tip:
                problems.append(f"{p['ref']}: no nozzle tip for package {p['package']}")
            elif tip not in tips or not {"board_touch_z", "inspect_z"} <= set(tips[tip]):
                problems.append(f"{p['ref']}: tip {tip} not measured (needs board_touch_z, inspect_z)")
            if fd is None:
                problems.append(f"{p['ref']}: no feeder holds {p['package']} {p.get('value', '')}")
            elif not pkg["symmetric"] and "tape_angle_deg" not in fd.d:
                problems.append(f"{p['ref']}: feeder {fd.id} needs tape_angle_deg ({p['package']} has a pin 1)")
        steps.append(dict(p=p, pkg=pkg, tip=tip, feeder=fd))
    order = []
    for s in steps:
        if s["tip"] not in order:
            order.append(s["tip"])
    steps.sort(key=lambda s: order.index(s["tip"]))
    return steps, problems


def run_job(job, pose, ms, mr, up_inspect, rotator, feeders, tips, packages=None, only=None,
            current_tip=None, recentre=None, reject_xy=None, ask=input, say=print, log=None,
            pick_press=0.4, place_press=0.3, approach_mm=0.8, vac_ms=300, release_ms=500,
            retries=3, angle_tol_deg=1.0, verify_empty=True):
    """Returns a list of per-placement result dicts. ms: machine state (down_cam_offset_mm,
    up_cam_xy_mm; recentre(tip) may update them in place). up_inspect(body_mm) -> UpCam result."""
    log = log or (lambda event, ok=None, **d: None)
    steps, problems = plan_job(job, feeders, tips, packages, only)
    if problems:
        raise JobError("job can't run:\n  " + "\n  ".join(problems))
    for s in steps:
        z = tips[s["tip"]]["board_touch_z"] - s["p"]["height_mm"] + place_press
        zp = feeder_part_top_z(s["feeder"].d, s["tip"], tips) + pick_press
        for name, v in (("place", z), ("pick", zp), ("inspect", tips[s["tip"]]["inspect_z"])):
            if not (mr.safe_travel_z < v <= 40.0):
                raise JobError(f"{s['p']['ref']}: {name} Z {v:.2f} outside Z{mr.safe_travel_z}..40")

    def lower_and(valve, z, wait_ms, slow_lift=False):
        mr.move_abs(z=z - approach_mm, feedrate=300)
        mr.move_abs(z=z, feedrate=60)
        mr.gcode(f"{valve}\nG4 P{int(wait_ms)}")
        if slow_lift:
            mr.move_abs(z=z - 0.5, feedrate=60)
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)

    def look_up(tip, body):
        up = np.asarray(ms["up_cam_xy_mm"], float)
        mr.move_abs(x=float(up[0]), y=float(up[1]))
        mr.move_abs(z=tips[tip]["inspect_z"], feedrate=300)
        return up_inspect(body)

    def reject(why):
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        if reject_xy is not None:
            mr.move_abs(x=float(reject_xy[0]), y=float(reject_xy[1]))
            mr.gcode(f"VALVE_OFF\nG4 P{int(release_ms)}")
        else:
            # don't just let go: the nozzle may be over the up camera's lens
            ask(f"  ({why}) no reject bin taught: hold a hand under the nozzle and press Enter "
                f"to release the part... ")
            mr.gcode("VALVE_OFF")
        log("reject", ok=True, reason=why)

    rotator.zero()
    results = []
    t_start = time.time()
    for n, s in enumerate(steps, 1):
        p, pkg, tip, fd = s["p"], s["pkg"], s["tip"], s["feeder"]
        rec = {"ref": p["ref"], "package": p["package"], "tip": tip, "feeder": fd.id}
        results.append(rec)
        say(f"\n[{n}/{len(steps)}] {p['ref']} {p.get('value', '')} ({p['package']}) from {fd.id}, tip {tip}")
        if tip != current_tip:
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            ask(f"Fit nozzle tip {tip} (now: {current_tip or 'unknown'}), then press Enter... ")
            if recentre is not None:
                say(f"  re-centring tip {tip} over the up camera: {recentre(tip)}")
            current_tip = tip
        off = np.asarray(ms["down_cam_offset_mm"], float)
        pick_z = feeder_part_top_z(fd.d, tip, tips) + pick_press
        ins = None
        for attempt in range(1, retries + 1):
            fd.ready()
            xy = fd.pick_xy()
            mr.move_abs(x=float(xy[0]), y=float(xy[1]))
            lower_and("VALVE_ON", pick_z, vac_ms)
            fd.after_pick()
            ins = look_up(tip, pkg["body_mm"])
            if "offset_mm" not in ins:
                say(f"  pick {attempt}: nozzle empty ({ins.get('reason')})")
                mr.move_abs(z=mr.safe_travel_z, feedrate=600)
                mr.gcode("VALVE_OFF")
                log("pick", ok=False, ref=p["ref"], feeder=fd.id, attempt=attempt)
                ins = None
                continue
            if not ins["ok"]:
                say(f"  pick {attempt}: {ins['reason']} -> reject")
                reject(ins["reason"])
                ins = None
                continue
            fd.learn(ins["offset_mm"])
            log("pick", ok=True, ref=p["ref"], feeder=fd.id, attempt=attempt, offset_mm=ins["offset_mm"])
            break
        if ins is None:
            rec["status"] = f"no good part after {retries} picks"
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            a = ask(f"  {p['ref']}: feeder {fd.id} gave no good part {retries} times. Fix it and press "
                    f"Enter to go on with the next part, or type q to stop: ")
            if str(a).strip().lower() == "q":
                rec["status"] += " (stopped)"
                break
            continue
        u = np.asarray(ins["offset_mm"], float)
        say(f"  on nozzle: offset {np.round(u, 3).tolist()} mm, angle {ins['angle_deg']:.1f}")
        # turn
        target = pose.part_angle(p["rot"]) + pkg["axis_deg"]        # long-axis machine angle
        rec["target_angle_deg"] = target
        if not pkg["symmetric"]:
            coarse = wrap180(pose.part_angle(p["rot"]) - fd.d["tape_angle_deg"])
            rotator.turn_part(coarse)
            u = rotate_xy(u, coarse)
            ins = look_up(tip, pkg["body_mm"])
            if "offset_mm" not in ins:
                rec["status"] = "part lost while turning"
                reject(rec["status"])
                continue
            fine = shortest_turn(target, ins["angle_deg"], 180.0)
            if abs(fine) > MAX_FINE_DEG:
                rec["status"] = f"part sits {fine:.0f} deg off its tape angle -> rejected"
                say("  " + rec["status"])
                reject(rec["status"])
                continue
            u = np.asarray(ins["offset_mm"], float)
        lost = False
        for _ in range(2):
            d = shortest_turn(target, ins["angle_deg"], 180.0)
            if abs(d) <= angle_tol_deg:
                break
            rotator.turn_part(d)
            ins = look_up(tip, pkg["body_mm"])
            if "offset_mm" not in ins:
                lost = True
                break
            u = np.asarray(ins["offset_mm"], float)
        if lost:
            rec["status"] = "part lost while turning"
            reject(rec["status"])
            continue
        rec["placed_angle_deg"] = ins["angle_deg"]
        # place
        b = pose.to_machine([p["x"], p["y"]])
        nz = b + off - u
        place_z = tips[tip]["board_touch_z"] - p["height_mm"] + place_press
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        mr.move_abs(x=float(nz[0]), y=float(nz[1]))
        lower_and("VALVE_OFF", place_z, release_ms, slow_lift=True)
        rec.update(status="placed", board_xy=b.tolist(), nozzle_xy=nz.tolist(), part_offset_mm=u.tolist(),
                   place_z=place_z)
        log("place", ok=True, ref=p["ref"], xy=b.tolist(), angle=rec["placed_angle_deg"])
        say(f"  placed at {np.round(b, 3).tolist()}, angle {rec['placed_angle_deg']:.1f} (target "
            f"{target % 180:.1f} mod 180)")
        if verify_empty:
            chk = look_up(tip, None)
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            if chk.get("ok") or "offset_mm" in chk:
                rec["status"] = "still on the nozzle after placing"
                say(f"  {p['ref']}: part came back on the nozzle!")
                reject(rec["status"])
    done = sum(r.get("status") == "placed" for r in results)
    mr.move_abs(z=mr.safe_travel_z, feedrate=600)
    say(f"\njob done: {done}/{len(steps)} placed in {time.time() - t_start:.0f} s")
    return results


def dry_run(job, pose, look, packages=None, only=None, say=print):
    """No parts: centre the down camera over every placement (look(xy, name) takes a photo)."""
    steps, _ = plan_job(job, None, {}, packages, only, need_feeders=False)
    for n, s in enumerate(steps, 1):
        p = s["p"]
        xy = pose.to_machine([p["x"], p["y"]])
        say(f"[{n}/{len(steps)}] {p['ref']} ({p['package']}, rot {p['rot']}, tip {s['tip']}): "
            f"{np.round(xy, 3).tolist()}, machine angle {pose.part_angle(p['rot']):.1f}")
        look(xy, f"{n:03d}_{p['ref']}")
    return steps
