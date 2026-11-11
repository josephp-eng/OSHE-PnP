"""Place-and-measure test: find parts with the down camera, pick one, inspect it on the up
camera, place it on a target, then find it again with the down camera and report the error.

All positions are TOOLHEAD coordinates "with the down camera centred on the point" (what you
read in Mainsail when the reticle is on something). The nozzle goes over such a point at
point + down_cam_offset_mm.

Finding parts uses BEFORE/AFTER images (difference), not what parts look like: the pick area
is photographed empty once, then with the parts; at each target, empty and then after placing.
That works for any part colour and ignores copper pads, silkscreen and tape (on 2026-11-11 a
"find a part-sized blob" search on this maskless board picked a pad and aimed at part edges).

Why the numbers mean something (u = up-camera offset of the part on the nozzle, e = landing
error = where it landed minus the target):
  * without --correct, pick and place both use the same down_cam_offset, so an offset error
    cancels: e is only how far the part slid at pick-up, and (e - u) is the error a corrected
    job placement would have;
  * with --correct, e is that job placement error directly.
Either way, the mean of that "job error" is what down_cam_offset_mm should be reduced by.
Hardware is injected (callables), so the whole cycle runs against a simulator in the tests.
"""
import cv2
import numpy as np

from .detect import find_parts_by_difference


def _mean_frame(grab, n=2):
    return np.mean([grab().astype(np.float32) for _ in range(n)], axis=0).astype(np.uint8)


def place_test(mr, down_grab, down_motion, down_principal, up_inspect, ms, pick_view, targets,
               size_mm=(2.0, 1.25), thickness_mm=0.85, working_distance_mm=11.5,
               pick_z=37.8, place_z=37.7, inspect_z=34.5, approach_mm=0.8, vac_ms=300,
               release_ms=500, correct=False, wait_next=input, say=print, save=None, max_picks=None,
               park_xy=None, view_half_mm=(14.0, 7.0)):
    """Fill `targets` one by one with parts taken from the pick area seen from `pick_view`.
    Returns a list of per-attempt dicts (see the module docstring)."""
    off = np.array(ms["down_cam_offset_mm"], float)
    up_xy = np.array(ms["up_cam_xy_mm"], float)
    pick_view = np.asarray(pick_view, float)
    mpp = down_motion.mm_per_px
    exp_px = tuple(v / mpp for v in sorted(size_mm, reverse=True))
    # A part's top is closer to the camera than the board the motion map was made at, so it
    # looks bigger and an off-centre offset reads too far out by this factor.
    k_height = working_distance_mm / (working_distance_mm - thickness_mm)
    save = save or (lambda name, img: None)

    def to_machine(view_xy, px):
        return np.asarray(view_xy, float) + down_motion.offset_mm(np.asarray(px, float) - down_principal) / k_height

    def look(xy):
        mr.move_abs(x=float(xy[0]), y=float(xy[1]))
        return _mean_frame(down_grab)

    def park():
        # head out of the way so tweezers can reach the pick area
        if park_xy is not None:
            mr.move_abs(x=float(park_xy[0]), y=float(park_xy[1]))

    # -- reference image of the empty pick area, then the parts
    lo, hi = pick_view - view_half_mm, pick_view + view_half_mm
    area = f"X {lo[0]:.0f}-{hi[0]:.0f}, Y {lo[1]:.0f}-{hi[1]:.0f} (down-camera coordinates)"
    park()
    wait_next(f"Pick area = what the down camera sees from {np.round(pick_view, 2).tolist()}: {area}. "
              "Remove any loose parts from it, then Enter: ")
    ref_pick = look(pick_view)
    save("pick_empty", ref_pick)
    park()
    wait_next("Now put parts in the pick area, flat, about 3 mm apart, away from its edges "
              "(more than you need is fine). Enter when done: ")
    failed = []                     # positions that did not pick (skip them)
    results = []
    t_idx = 0
    picks = 0
    while t_idx < len(targets) and (max_picks is None or picks < max_picks):
        img = look(pick_view)
        cands = [c for c in find_parts_by_difference(ref_pick, img, exp_px)
                 if not any(np.linalg.norm(to_machine(pick_view, c.center_px) - f) < 1.0 for f in failed)]
        if not cands:
            park()
            ans = wait_next(f"No parts found in the pick area ({area}). Add more and Enter "
                            "(q + Enter to stop): ")
            if str(ans).strip().lower().startswith("q"):
                break
            continue
        cands.sort(key=lambda c: np.hypot(*(np.asarray(c.center_px) - down_principal)))
        part = to_machine(pick_view, cands[0].center_px)
        picks += 1
        target = np.asarray(targets[t_idx], float)
        rec = {"attempt": picks, "picked_from": part.tolist(), "target": target.tolist()}
        results.append(rec)
        say(f"\n--- part {picks}: at {np.round(part, 3).tolist()} -> target {t_idx + 1}/{len(targets)} "
            f"{np.round(target, 3).tolist()} ({len(cands)} part(s) in the pick area) ---")
        # pick
        n = part + off
        mr.move_abs(x=float(n[0]), y=float(n[1]))
        mr.move_abs(z=pick_z - approach_mm, feedrate=300)
        mr.move_abs(z=pick_z, feedrate=60)
        mr.gcode(f"VALVE_ON\nG4 P{int(vac_ms)}")
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        # inspect
        mr.move_abs(x=float(up_xy[0]), y=float(up_xy[1]))
        mr.move_abs(z=inspect_z, feedrate=300)
        ins = up_inspect(size_mm)
        rec["inspect"] = ins
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        if "offset_mm" not in ins:
            rec["status"] = "pick failed (nozzle empty)"
            say(f"  nozzle empty after the pick ({ins.get('reason')}); will try another part.")
            mr.gcode("VALVE_OFF")
            failed.append(part)
            continue
        u = np.array(ins["offset_mm"], float)
        say(f"  on nozzle: offset {np.round(u, 3).tolist()} mm, angle {ins['angle_deg']:.1f} deg, "
            f"size {np.round(ins['size_mm'], 2).tolist()} mm")
        if not ins["ok"]:
            # held, but not the part we expect (on its side, two parts, ...): put it back
            rec["status"] = f"bad part ({ins['reason']}), returned to the pick spot"
            say(f"  {ins['reason']} -> putting it back where it came from")
            n = part + off
            mr.move_abs(x=float(n[0]), y=float(n[1]))
            mr.move_abs(z=pick_z - approach_mm, feedrate=300)
            mr.move_abs(z=pick_z - 0.1, feedrate=60)
            mr.gcode(f"VALVE_OFF\nG4 P{int(release_ms)}")
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            failed.append(part)
            continue
        # empty target, then place
        ref_t = look(target)
        n = target + off - (u if correct else 0.0)
        mr.move_abs(x=float(n[0]), y=float(n[1]))
        mr.move_abs(z=place_z - approach_mm, feedrate=300)
        mr.move_abs(z=place_z, feedrate=60)
        mr.gcode(f"VALVE_OFF\nG4 P{int(release_ms)}")
        mr.move_abs(z=place_z - 0.5, feedrate=60)
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        t_idx += 1
        # measure
        after = look(target)
        found = [c for c in find_parts_by_difference(ref_t, after, exp_px)
                 if np.hypot(*(np.asarray(c.center_px) - down_principal)) * mpp < 2.0]
        save(f"target{t_idx}_empty", ref_t)
        save(f"target{t_idx}_after", after)
        if not found:
            rec["status"] = "placed part not found at the target (still on the nozzle?)"
            say("  NOT found at the target after placing - check the nozzle and the tape.")
            continue
        found.sort(key=lambda c: np.hypot(*(np.asarray(c.center_px) - down_principal)))
        landed = to_machine(target, found[0].center_px)
        e = landed - target
        job = e if correct else e - u
        rec.update(status="ok", landed=landed.tolist(), error_mm=e.tolist(), job_error_mm=job.tolist())
        say(f"  landed {np.round(landed, 3).tolist()}: error {np.round(e, 3).tolist()} mm "
            f"(|e| {np.linalg.norm(e):.3f}); job error {np.round(job, 3).tolist()} mm")
    return results


def summarize(results, off, correct):
    ok = [r for r in results if r.get("status") == "ok"]
    out = {"attempts": len(results), "ok": len(ok), "correct": correct,
           "failed": [r["status"] for r in results if r.get("status") != "ok"]}
    if ok:
        job = np.array([r["job_error_mm"] for r in ok])
        e = np.array([r["error_mm"] for r in ok])
        out.update(mean_job_error_mm=job.mean(0).round(4).tolist(),
                   std_job_error_mm=(job.std(0, ddof=1).round(4).tolist() if len(ok) > 1 else None),
                   mean_landing_error_mm=e.mean(0).round(4).tolist(),
                   max_landing_error_mm=float(np.linalg.norm(e, axis=1).max().round(4)),
                   suggested_down_cam_offset_mm=(np.asarray(off) - job.mean(0)).round(4).tolist())
    return out
