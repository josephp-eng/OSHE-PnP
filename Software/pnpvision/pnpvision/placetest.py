"""Place-and-measure test: find a part with the down camera, pick it, inspect it on the up
camera, place it on a target, then find it again with the down camera and report the error.

All positions are TOOLHEAD coordinates "with the down camera centred on the point" (what you
read in Mainsail when the reticle is on something). The nozzle goes over such a point at
point + down_cam_offset_mm.

Why the numbers mean something (u = up-camera offset of the part on the nozzle, e = landing
error = where it landed minus the target):
  * without --correct, pick and place both use the same down_cam_offset, so an offset error
    cancels: e is only how far the part slid at pick-up, and (e - u) is the error a corrected
    job placement would have;
  * with --correct, e is that job placement error directly.
Either way, the mean of that "job error" is what down_cam_offset_mm should be reduced by.
Hardware is injected (callables), so the whole cycle runs against a simulator in the tests.
"""
import time

import numpy as np

from .detect import measure_part
from .servo import center_on


def find_and_center(mr, grab, motion, principal, size_mm, search_mm, tol_mm=0.015):
    """Down camera: centre on a part-sized blob within search_mm of the reticle.
    Returns the toolhead XY with the part centred, or None if not found / not converged."""
    mpp = motion.mm_per_px
    exp_px = tuple(v / mpp for v in sorted(size_mm, reverse=True))
    first = [True]

    def locate(img):
        # first look: the whole search area; once centred, only a little beyond the part
        r_mm = search_mm if first[0] else 0.5 * float(np.hypot(*size_mm)) + 1.0
        first[0] = False
        return measure_part(img, principal, r_mm / mpp, expected_px=exp_px)

    res = center_on(grab, locate, mr.move_rel, motion, principal, tol_mm=tol_mm, max_step_mm=search_mm + 1)
    if not res["converged"]:
        return None, res
    x, y, _ = mr.position()
    return np.array([x, y]), res


def place_test(mr, down_grab, down_motion, down_principal, up_inspect, ms, pick_xy, targets,
               size_mm=(2.0, 1.25), search_mm=3.0, pick_z=37.8, place_z=37.7, inspect_z=34.5,
               approach_mm=0.8, vac_ms=300, release_ms=500, correct=False, wait_next=input,
               say=print):
    """Run len(targets) cycles. Returns a list of per-cycle dicts (see the module docstring)."""
    off = np.array(ms["down_cam_offset_mm"], float)
    up_xy = np.array(ms["up_cam_xy_mm"], float)
    results = []
    for k, target in enumerate(targets):
        target = np.asarray(target, float)
        rec = {"cycle": k + 1, "target": target.tolist()}
        results.append(rec)
        say(f"\n--- cycle {k + 1}/{len(targets)} ---")
        # 1. find the part at the pick spot
        mr.move_abs(x=float(pick_xy[0]), y=float(pick_xy[1]))
        part, res = find_and_center(mr, down_grab, down_motion, down_principal, size_mm, search_mm)
        if part is None:
            rec["status"] = "part not found at pick spot"
            say(f"  part not found near {np.round(pick_xy, 3)} ({res}).")
        else:
            rec["picked_from"] = part.tolist()
            # 2. pick
            n = part + off
            mr.move_abs(x=float(n[0]), y=float(n[1]))
            mr.move_abs(z=pick_z - approach_mm, feedrate=300)
            mr.move_abs(z=pick_z, feedrate=60)
            mr.gcode(f"VALVE_ON\nG4 P{int(vac_ms)}")
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            # 3. inspect on the up camera
            mr.move_abs(x=float(up_xy[0]), y=float(up_xy[1]))
            mr.move_abs(z=inspect_z, feedrate=300)
            ins = up_inspect(size_mm)
            rec["inspect"] = ins
            if "offset_mm" not in ins:
                rec["status"] = "nothing on the nozzle (pick failed)"
                say("  up camera: no part on the nozzle -> pick failed.")
                mr.move_abs(z=mr.safe_travel_z, feedrate=600)
                mr.gcode("VALVE_OFF")
            else:
                u = np.array(ins["offset_mm"], float)
                say(f"  on nozzle: offset {np.round(u, 3)} mm, angle {ins['angle_deg']:.1f} deg, "
                    f"size {np.round(ins['size_mm'], 2)} mm{'' if ins['ok'] else '  (' + ins['reason'] + ')'}")
                # 4. place
                n = target + off - (u if correct else 0.0)
                mr.move_abs(x=float(n[0]), y=float(n[1]))
                mr.move_abs(z=place_z - approach_mm, feedrate=300)
                mr.move_abs(z=place_z, feedrate=60)
                mr.gcode(f"VALVE_OFF\nG4 P{int(release_ms)}")
                mr.move_abs(z=place_z - 0.5, feedrate=60)
                mr.move_abs(z=mr.safe_travel_z, feedrate=600)
                # 5. measure where it landed
                mr.move_abs(x=float(target[0]), y=float(target[1]))
                landed, res = find_and_center(mr, down_grab, down_motion, down_principal, size_mm,
                                              search_mm=1.5)
                if landed is None:
                    rec["status"] = "placed part not found (still on the nozzle?)"
                    say("  placed part NOT found near the target - check whether it released.")
                else:
                    e = landed - target
                    job = e if correct else e - u
                    rec.update(status="ok", landed=landed.tolist(), error_mm=e.tolist(), job_error_mm=job.tolist())
                    say(f"  landed {np.round(landed, 3)}: error {np.round(e, 3)} mm "
                        f"(|e| {np.linalg.norm(e):.3f}); job error {np.round(job, 3)} mm")
        if k + 1 < len(targets):
            mr.move_abs(z=mr.safe_travel_z, feedrate=600)
            ans = wait_next(f"Put the next part at the pick spot (or the same part back), then Enter "
                            f"(q + Enter to stop): ")
            if str(ans).strip().lower().startswith("q"):
                break
    return results


def summarize(results, off, correct):
    ok = [r for r in results if r.get("status") == "ok"]
    out = {"cycles": len(results), "ok": len(ok), "correct": correct}
    if ok:
        job = np.array([r["job_error_mm"] for r in ok])
        e = np.array([r["error_mm"] for r in ok])
        out.update(mean_job_error_mm=job.mean(0).round(4).tolist(),
                   std_job_error_mm=(job.std(0, ddof=1).round(4).tolist() if len(ok) > 1 else None),
                   mean_landing_error_mm=e.mean(0).round(4).tolist(),
                   max_landing_error_mm=float(np.linalg.norm(e, axis=1).max().round(4)),
                   suggested_down_cam_offset_mm=(np.asarray(off) - job.mean(0)).round(4).tolist())
    return out
