"""place_test against a simulated machine: toolhead, valve, parts that stick to the nozzle,
a down camera that sees parts (and copper pads) from above and an up camera that sees the
part on the nozzle."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pnpvision.motion import MotionMap
from pnpvision.detect import measure_part
from pnpvision.placetest import place_test, summarize

OFF_SET = np.array([8.328, 53.059])          # what the software believes
OFF_TRUE = OFF_SET + [0.12, -0.08]           # what the machine really has (calibration error)
UP_XY = np.array([85.59, 137.84])
DOWN_M = np.array([[-58.5, 0.4], [-0.3, 59.0]])     # camera moves (down cam, ~59 px/mm at the board)
UP_M = np.array([[0.27, -37.3], [38.6, 0.08]])      # target moves (up cam, 90 deg turned)
P_DOWN = np.array([960.0, 540.0])
P_UP = np.array([724.7, 382.6])
L_MM, W_MM, T_MM, WD = 2.0, 1.25, 0.85, 11.5
K = WD / (WD - T_MM)                                 # part tops look this much bigger / further out
PICK_VIEW = np.array([150.0, 253.7])
PADS = [PICK_VIEW + d for d in ((-6, 3), (-4, 3), (5, -4), (7, -4), (0, 6))] + \
       [np.array([155.39 - 4 * i, 267.7]) + d for i in range(4) for d in ((-1.0, 2.2), (1.0, 2.2))]


class World:
    """Positions in 'down camera centred' coordinates. The nozzle tip is at toolhead - OFF_TRUE."""
    def __init__(self, slip=(0.05, -0.03)):
        self.xyz = [100.0, 200.0, 30.0]
        self.safe_travel_z = 30.0
        self.parts = []                 # (xy, stuck) lying on the board
        self.held = None                # offset of the held part from the nozzle
        self.slip = np.array(slip)

    # -- Moonraker-like API used by place_test
    def position(self):
        return tuple(self.xyz)

    def move_abs(self, x=None, y=None, z=None, feedrate=None):
        if (x is not None or y is not None) and self.xyz[2] > self.safe_travel_z + 1e-6:
            self.xyz[2] = self.safe_travel_z        # same lift rule as Moonraker.move_abs
        if x is not None:
            self.xyz[0] = x
        if y is not None:
            self.xyz[1] = y
        if z is not None:
            self.xyz[2] = z

    def nozzle(self):
        return np.array(self.xyz[:2]) - OFF_TRUE

    def gcode(self, script):
        for line in script.splitlines():
            if line.startswith("VALVE_ON") and self.held is None and self.xyz[2] > 37.3:
                for i, (p, stuck) in enumerate(self.parts):
                    if np.linalg.norm(p - self.nozzle()) < 0.6 and not stuck:
                        self.held = p - self.nozzle() + self.slip
                        self.parts.pop(i)
                        break
            elif line.startswith("VALVE_OFF") and self.held is not None:
                if self.xyz[2] > 37.3:
                    self.parts.append((self.nozzle() + self.held, False))
                self.held = None

    # -- cameras
    @staticmethod
    def _draw(img, center_px, ax_px, ay_px, value):
        pts = np.array([center_px + s * ax_px / 2 + t * ay_px / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        cv2.fillPoly(img, [pts.astype(np.int32)], value)

    def down_frame(self):
        img = np.full((1080, 1920, 3), (60, 120, 40), np.uint8)      # green board (BGR)
        cam = np.array(self.xyz[:2])
        for p in PADS:                                                 # copper pads, part-like size
            self._draw(img, P_DOWN + DOWN_M @ (cam - p), DOWN_M @ np.array([0, -1.3]),
                       DOWN_M @ np.array([-1.0, 0]), (190, 200, 205))
        for p, _ in self.parts:                                        # tan parts along machine Y
            c = P_DOWN + K * (DOWN_M @ (cam - p))
            self._draw(img, c, K * (DOWN_M @ np.array([0, -L_MM])), K * (DOWN_M @ np.array([-W_MM, 0])),
                       (90, 140, 170))
        noise = np.random.default_rng(int(cam.sum() * 100) % 1000).normal(0, 3, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def up_inspect(self, size):
        img = np.full((720, 1280), 105, np.uint8)
        if self.held is not None and np.linalg.norm(np.array(self.xyz[:2]) - UP_XY) < 0.01:
            q = self.nozzle() + self.held - (UP_XY - OFF_TRUE)
            c = P_UP + UP_M @ q
            self._draw(img, c, UP_M @ np.array([0, L_MM]), UP_M @ np.array([W_MM, 0]), 200)
        um = MotionMap(UP_M, "target")
        pm = measure_part(img, P_UP, 3.0 / um.mm_per_px, expected_px=(L_MM / um.mm_per_px, W_MM / um.mm_per_px))
        if pm is None:
            return {"ok": False, "reason": "no part found"}
        return {"ok": True, "reason": "", "offset_mm": um.offset_mm(np.array(pm.center_px) - P_UP).tolist(),
                "angle_deg": 0.0, "size_mm": [pm.long_px * um.mm_per_px, pm.short_px * um.mm_per_px]}


def _run(correct):
    w = World()
    prompts = []

    def wait(prompt):
        prompts.append(prompt)
        if len(prompts) == 2:            # "now put parts in the view": more than needed, one stuck
            for d, stuck in (((-0.4, 0.3), True), ((3.5, -2.0), False), ((-5.0, 4.0), False),
                             ((8.0, 1.0), False), ((-9.0, -5.0), False)):
                w.parts.append((PICK_VIEW + d, stuck))
        return ""

    ms = {"down_cam_offset_mm": OFF_SET.tolist(), "up_cam_xy_mm": UP_XY.tolist()}
    targets = [np.array([155.39, 267.7]) + [-4.0 * i, 0] for i in range(3)]
    res = place_test(w, w.down_frame, MotionMap(DOWN_M, "camera"), P_DOWN, w.up_inspect, ms,
                     PICK_VIEW, targets, thickness_mm=T_MM, working_distance_mm=WD,
                     correct=correct, wait_next=wait, say=lambda *a: None)
    return w, res, summarize(res, OFF_SET, correct)


def test_place_test_uncorrected_measures_offset_error():
    w, res, s = _run(correct=False)
    assert s["ok"] == 3, res
    assert res[0]["status"].startswith("pick failed"), res[0]     # the stuck part, nearest the centre
    # uncorrected: landing error is only the pick slip; job error reveals the offset error
    assert np.allclose(s["mean_landing_error_mm"], [0.05, -0.03], atol=0.04), s
    assert np.allclose(s["suggested_down_cam_offset_mm"], OFF_TRUE, atol=0.04), s


def test_place_test_corrected_lands_off_by_offset_error():
    w, res, s = _run(correct=True)
    assert s["ok"] == 3, res
    assert np.allclose(s["mean_landing_error_mm"], OFF_SET - OFF_TRUE, atol=0.04), s
    assert np.allclose(s["suggested_down_cam_offset_mm"], OFF_TRUE, atol=0.04), s
