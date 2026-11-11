"""place_test against a simulated machine: toolhead, valve, a part that sticks to the nozzle,
a down camera that sees the part from above and an up camera that sees it on the nozzle."""
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
DOWN_M = np.array([[-58.5, 0.4], [-0.3, 59.0]])     # camera moves (down cam, ~59 px/mm)
UP_M = np.array([[0.27, -37.3], [38.6, 0.08]])      # target moves (up cam, 90 deg turned)
P_DOWN = np.array([960.0, 540.0])
P_UP = np.array([724.7, 382.6])
L_MM, W_MM = 2.0, 1.25


class World:
    """Positions in 'down camera centred' coordinates. The nozzle tip is at toolhead - OFF_TRUE."""
    def __init__(self, part_xy, slip=(0.05, -0.03)):
        self.xyz = [100.0, 200.0, 30.0]
        self.safe_travel_z = 30.0
        self.parts = [np.array(part_xy, float)]     # parts lying on the board
        self.held = None                            # offset of the held part from the nozzle
        self.slip = np.array(slip)
        self.vac = False

    # -- Moonraker-like API used by place_test / center_on
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

    def move_rel(self, dx=0.0, dy=0.0):
        self.move_abs(self.xyz[0] + dx, self.xyz[1] + dy)

    def nozzle(self):
        return np.array(self.xyz[:2]) - OFF_TRUE

    def gcode(self, script):
        for line in script.splitlines():
            if line.startswith("VALVE_ON"):
                self.vac = True
                if self.held is None and self.xyz[2] > 37.3:     # touching the board
                    for i, p in enumerate(self.parts):
                        if np.linalg.norm(p - self.nozzle()) < 0.6:
                            self.held = p - self.nozzle() + self.slip
                            self.parts.pop(i)
                            break
            elif line.startswith("VALVE_OFF"):
                self.vac = False
                if self.held is not None and self.xyz[2] > 37.3:
                    self.parts.append(self.nozzle() + self.held)
                    self.held = None

    # -- cameras
    def _draw(self, img, center_px, ax_px, ay_px):
        pts = np.array([center_px + s * ax_px / 2 + t * ay_px / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        cv2.fillPoly(img, [pts.astype(np.int32)], 30)

    def down_frame(self):
        img = np.full((1080, 1920), 120, np.uint8)        # green PCB in grey
        cam = np.array(self.xyz[:2])
        for p in self.parts:                                # parts lie along machine Y
            c = P_DOWN + DOWN_M @ (cam - p)
            self._draw(img, c, DOWN_M @ np.array([0, -L_MM]), DOWN_M @ np.array([-W_MM, 0]))
        return cv2.GaussianBlur(img, (5, 5), 0)

    def up_inspect(self, size):
        img = np.full((720, 1280), 105, np.uint8)
        if self.held is not None and np.linalg.norm(np.array(self.xyz[:2]) - UP_XY) < 0.01:
            q = self.nozzle() + self.held - (UP_XY - OFF_TRUE)   # part relative to the nozzle at the up cam
            c = P_UP + UP_M @ q
            self._draw(img, c, UP_M @ np.array([0, L_MM]), UP_M @ np.array([W_MM, 0]))
        um = MotionMap(UP_M, "target")
        pm = measure_part(img, P_UP, 3.0 / um.mm_per_px, expected_px=(L_MM / um.mm_per_px, W_MM / um.mm_per_px))
        if pm is None:
            return {"ok": False, "reason": "no part found"}
        return {"ok": True, "reason": "", "offset_mm": um.offset_mm(np.array(pm.center_px) - P_UP).tolist(),
                "angle_deg": 0.0, "size_mm": [pm.long_px * um.mm_per_px, pm.short_px * um.mm_per_px]}


def _run(correct):
    w = World(part_xy=(159.9, 253.2))
    supply = iter([(159.0, 254.1), (160.1, 253.3)])

    def next_part(_prompt):
        w.parts.append(np.array(next(supply)))      # "put the next part at the pick spot"
        return ""

    ms = {"down_cam_offset_mm": OFF_SET.tolist(), "up_cam_xy_mm": UP_XY.tolist()}
    targets = [np.array([159.39, 267.7]) + [4.0 * i, 0] for i in range(3)]
    res = place_test(w, w.down_frame, MotionMap(DOWN_M, "camera"), P_DOWN, w.up_inspect, ms,
                     np.array([159.39, 253.7]), targets, correct=correct, wait_next=next_part,
                     say=lambda *a: None)
    return res, summarize(res, OFF_SET, correct)


def test_place_test_uncorrected_measures_offset_error():
    res, s = _run(correct=False)
    assert s["ok"] == 3, res
    # uncorrected: landing error is only the pick slip; job error reveals the offset error
    assert np.allclose(s["mean_landing_error_mm"], [0.05, -0.03], atol=0.03), s
    assert np.allclose(s["suggested_down_cam_offset_mm"], OFF_TRUE, atol=0.03), s


def test_place_test_corrected_lands_off_by_offset_error():
    res, s = _run(correct=True)
    assert s["ok"] == 3, res
    # corrected: slip is removed, what remains is the offset error itself
    assert np.allclose(s["mean_landing_error_mm"], OFF_SET - OFF_TRUE, atol=0.03), s
    assert np.allclose(s["suggested_down_cam_offset_mm"], OFF_TRUE, atol=0.03), s
