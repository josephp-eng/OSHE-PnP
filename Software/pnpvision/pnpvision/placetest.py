"""Place-and-measure test: pick the part under the down camera, inspect it on the up camera,
place it on a target, then measure where it landed with the down camera.

You aim the picks: centre the down camera on a part (Mainsail reticle, by eye is fine), run the
command; after each part it asks you to jog onto the next one. Where it landed is measured
automatically by photographing the empty target before placing and again after: only what
changed counts, so part colour, copper pads, silkscreen and tape don't matter.

All positions are TOOLHEAD coordinates "with the down camera centred on the point" (what you
read in Mainsail when the reticle is on something). The nozzle goes over such a point at
point + down_cam_offset_mm.

Why the numbers mean something (u = up-camera offset of the part on the nozzle, e = landing
error = where it landed minus the target):
  * without --correct, pick and place both use the same down_cam_offset, so an offset error
    cancels: e is only how far off the part was picked (aim + slip), and (e - u) is the error
    a corrected job placement would have;
  * with --correct, e is that job placement error directly.
Either way, the mean of that "job error" is what down_cam_offset_mm should be reduced by.
Hardware is injected (callables), so the whole cycle runs against a simulator in the tests.
"""
import cv2
import numpy as np

from .detect import find_by_template, find_placed_part, measure_part


def _mean_frame(grab, n=2):
    return np.mean([grab().astype(np.float32) for _ in range(n)], axis=0).astype(np.uint8)


class PartFinder:
    """First part: wherever the down camera is centred when the test starts (you aim it).
    Next parts: found automatically from a picture of that first part (template) by searching
    the camera view from the first pick position for a shortlist, then checking candidates
    nearest first with the camera right over each one (same lighting as the template, and the
    part's height doesn't skew the scale there). A candidate that fails is not looked at again."""
    SHORTLIST = 0.55        # wide-view score to be worth a close look
    MAX_CHECKS = 8          # close looks per search (each is a short move + photo)

    def __init__(self, mr, grab, motion, principal, size_mm, k_height, origin, avoid=(),
                 threshold=0.65, say=print, save=None):
        self.save = save or (lambda name, img: None)
        self.searches = 0
        self.mr, self.grab, self.motion = mr, grab, motion
        self.principal = np.asarray(principal, float)
        self.k = k_height
        self.origin = np.asarray(origin, float)
        self.avoid = [np.asarray(p, float) for p in avoid]     # e.g. the targets (parts on tape)
        self.used = []                                          # picked or failed: skip
        self.rejected = []                                      # looked at close up: not a part
        self.threshold, self.say = threshold, say
        mpp = motion.mm_per_px
        half = int(round(0.5 * float(np.hypot(*size_mm)) * k_height / mpp * 1.1))
        img = self._look(self.origin)
        # You aimed by eye: measure the first part's real centre close to the reticle (only that
        # part is that close), so neither the first pick nor the template inherits the aim error.
        exp_px = tuple(v * k_height / mpp for v in sorted(size_mm, reverse=True))
        pm = measure_part(img, self.principal, half * 1.3, expected_px=exp_px)
        c = self.principal
        if pm is not None and np.linalg.norm(np.subtract(pm.center_px, self.principal)) * mpp < 0.6:
            c = np.asarray(pm.center_px, float)
            self.origin = self._to_machine(self.origin, c)
            say(f"  first part centre measured at {np.round(self.origin, 3).tolist()} "
                f"({np.linalg.norm(np.asarray(origin) - self.origin):.2f} mm from where you aimed)")
        else:
            say("  could not measure the first part's centre; using where you aimed")
        cx, cy = int(round(c[0])), int(round(c[1]))
        self.template = img[cy - half:cy + half + 1, cx - half:cx + half + 1].copy()
        # Match on the part only: its outline (slightly shrunk) as a mask, so the board around
        # the first part (pads, silkscreen) doesn't have to match around the others.
        self.mask = None
        if pm is not None and c is not self.principal:
            self.mask = np.zeros(self.template.shape[:2], np.uint8)
            box = cv2.boxPoints(((half + (c[0] - cx), half + (c[1] - cy)),
                                 (pm.long_px * 0.95, pm.short_px * 0.95), pm.angle_px_deg))
            cv2.fillPoly(self.mask, [np.round(box).astype(np.int32)], 255)
        self.first = True

    def _look(self, xy):
        self.mr.move_abs(x=float(xy[0]), y=float(xy[1]))
        return _mean_frame(self.grab)

    def _to_machine(self, view_xy, px):
        return np.asarray(view_xy, float) + self.motion.offset_mm(np.asarray(px, float) - self.principal) / self.k

    def __call__(self, last):
        if last is not None:
            self.used.append(np.asarray(last, float))
        if self.first:
            self.first = False
            return self.origin
        img = self._look(self.origin)
        # Wide view = shortlist only. Away from the middle the down light's glare changes how a
        # part looks (2026-11-11: real parts 0.64-0.78 there, tape/silkscreen up to 0.73), so
        # every candidate is judged again with the camera right over it, lit like the template.
        hits = find_by_template(img, self.template, threshold=self.SHORTLIST, scale=0.25, mask=self.mask)
        self.searches += 1
        self.save(f"search{self.searches}_raw", img)
        vis = img.copy()
        for px, score in hits:
            cv2.circle(vis, (int(px[0]), int(px[1])), 40, (0, 200, 255), 3)
            cv2.putText(vis, f"{score:.2f}", (int(px[0]) + 45, int(px[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        self.save(f"search{self.searches}", vis)
        cands = []
        for px, score in hits:
            p = self._to_machine(self.origin, px)
            if any(np.linalg.norm(p - q) < 1.0 for q in self.used + self.rejected) or \
               any(np.linalg.norm(p - q) < 2.5 for q in self.avoid):
                continue
            cands.append((np.linalg.norm(p - self.origin), p, score))
        cands.sort(key=lambda c: c[0])                     # nearest first
        r_px = 1.5 / self.motion.mm_per_px
        tried = []
        for _, p, score in cands[:self.MAX_CHECKS]:
            img = self._look(p)
            fine = find_by_template(img, self.template, center_px=self.principal, radius_px=r_px,
                                    threshold=0.0, mask=self.mask, scale=0.5)
            s = fine[0][1] if fine else 0.0
            tried.append(f"{s:.2f}")
            if fine and s >= self.threshold:
                p = self._to_machine(p, fine[0][0])
                self.say(f"  next part at {np.round(p, 3).tolist()} (wide view {score:.2f}, close up {s:.2f})")
                return p
            self.rejected.append(p)
        self.say(f"  no more parts found near the first pick position ({len(cands)} candidate(s); "
                 f"close-up scores {', '.join(tried) or 'none'}; need {self.threshold}).")
        return None


def place_test(mr, down_grab, down_motion, down_principal, up_inspect, ms, next_pick, targets,
               size_mm=(2.0, 1.25), thickness_mm=0.85, working_distance_mm=11.5,
               pick_z=37.8, place_z=37.7, inspect_z=34.5, approach_mm=0.8, vac_ms=300,
               release_ms=500, correct=False, say=print, save=None):
    """Fill `targets` one by one. next_pick(last_pick_xy or None) -> XY of the next part (down
    camera centred on it) or None to stop. Returns a list of per-attempt dicts."""
    off = np.array(ms["down_cam_offset_mm"], float)
    up_xy = np.array(ms["up_cam_xy_mm"], float)
    mpp = down_motion.mm_per_px
    exp_px = tuple(v / mpp for v in sorted(size_mm, reverse=True))
    # A part's top is closer to the camera than the board the motion map was made at, so an
    # off-centre offset reads too far out by this factor.
    k_height = working_distance_mm / (working_distance_mm - thickness_mm)
    save = save or (lambda name, img: None)

    def look(xy):
        mr.move_abs(x=float(xy[0]), y=float(xy[1]))
        return _mean_frame(down_grab)

    def lower_and(valve, z, ms_wait, slow_lift=False):
        mr.move_abs(z=z - approach_mm, feedrate=300)
        mr.move_abs(z=z, feedrate=60)
        mr.gcode(f"{valve}\nG4 P{int(ms_wait)}")
        if slow_lift:
            mr.move_abs(z=z - 0.5, feedrate=60)
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)

    results = []
    t_idx = 0
    part = None
    while t_idx < len(targets):
        part = next_pick(part)
        if part is None:
            break
        part = np.asarray(part, float)
        target = np.asarray(targets[t_idx], float)
        rec = {"attempt": len(results) + 1, "picked_from": part.tolist(), "target": target.tolist()}
        results.append(rec)
        say(f"\n--- part {len(results)}: from {np.round(part, 3).tolist()} -> target {t_idx + 1}/"
            f"{len(targets)} {np.round(target, 3).tolist()} ---")
        # pick
        n = part + off
        mr.move_abs(x=float(n[0]), y=float(n[1]))
        lower_and("VALVE_ON", pick_z, vac_ms)
        # inspect
        mr.move_abs(x=float(up_xy[0]), y=float(up_xy[1]))
        mr.move_abs(z=inspect_z, feedrate=300)
        ins = up_inspect(size_mm)
        rec["inspect"] = ins
        mr.move_abs(z=mr.safe_travel_z, feedrate=600)
        if "offset_mm" not in ins:
            rec["status"] = "pick failed (nozzle empty)"
            say(f"  nozzle empty after the pick ({ins.get('reason')}).")
            mr.gcode("VALVE_OFF")
            continue
        u = np.array(ins["offset_mm"], float)
        say(f"  on nozzle: offset {np.round(u, 3).tolist()} mm, angle {ins['angle_deg']:.1f} deg, "
            f"size {np.round(ins['size_mm'], 2).tolist()} mm")
        if not ins["ok"]:
            # held, but not the part we expect (on its side, two parts, ...): put it back
            rec["status"] = f"bad part ({ins['reason']}), put back"
            say(f"  {ins['reason']} -> putting it back where it came from")
            n = part + off
            mr.move_abs(x=float(n[0]), y=float(n[1]))
            lower_and("VALVE_OFF", pick_z - 0.1, release_ms)
            continue
        # photo of the empty target, place, photo again
        ref_t = look(target)
        n = target + off - (u if correct else 0.0)
        mr.move_abs(x=float(n[0]), y=float(n[1]))
        lower_and("VALVE_OFF", place_z, release_ms, slow_lift=True)
        t_idx += 1
        after = look(target)
        save(f"target{t_idx}_empty", ref_t)
        save(f"target{t_idx}_after", after)
        found = find_placed_part(ref_t, after, down_principal, 2.5 / mpp,
                                 tuple(v * k_height for v in exp_px))
        if found is None:
            rec["status"] = "placed part not found at the target (still on the nozzle?)"
            say("  NOT found at the target after placing - check the nozzle and the tape.")
            continue
        landed = target + down_motion.offset_mm(np.asarray(found.center_px) - down_principal) / k_height
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
