"""Hardware-free tests: run with  python -m pytest -q   (or python tests/test_offline.py)"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, cv2

from pnpvision.lens import LensModel
from pnpvision.motion import MotionMap
from pnpvision.detect import find_circle, measure_part
from pnpvision.servo import calibrate_motion, center_on
from pnpvision.eventlog import EventLog

CAL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration")


def test_lens_loads_and_undistorts():
    for f, size in (("calibration_cam_down.npz", (1920, 1080)), ("calibration_cam_up.npz", (1280, 720))):
        lens = LensModel(os.path.join(CAL, f))
        assert lens.size == size
        out = lens.undistort(np.full((size[1], size[0], 3), 128, np.uint8))
        assert out.shape == (size[1], size[0], 3)
        print(f, lens.health())


def test_safe_travel_height():
    """Larger Z = lower nozzle. XY moves must lift to safe_travel_z first; Z goes down last."""
    from pnpvision.moonraker import Moonraker, MoveRefused

    class Fake(Moonraker):
        def __init__(self, pos, **kw):
            super().__init__(x_range=(0, 235), y_range=(0, 343), z_range=(0, 60), **kw)
            self.pos, self.sent = list(pos), []
        def status(self, obj="toolhead"):
            return {"position": self.pos + [0.0], "homed_axes": "xyz"}
        def gcode(self, script):
            self.sent.append(script)
            for line in script.splitlines():
                for w in line.split()[1:] if line.startswith("G1") else []:
                    if w[0] in "XYZ":
                        self.pos["XYZ".index(w[0])] = float(w[1:])

    lines = lambda m: [l for s in m.sent for l in s.splitlines() if l.startswith("G1")]

    m = Fake([50, 50, 45.0], safe_travel_z=30.0)          # nozzle low (near the plate)
    m.move_rel(5, 0)
    assert lines(m)[0].startswith("G1 Z30"), lines(m)     # lifted first...
    assert lines(m)[1].startswith("G1 X55"), lines(m)     # ...then moved sideways
    assert m.pos == [55, 50, 30.0]

    m = Fake([50, 50, 10.0], safe_travel_z=30.0)          # already high: no extra Z move
    m.move_rel(0, 3)
    assert lines(m) == ["G1 Y53.0000 F3000"], lines(m)

    m = Fake([50, 50, 10.0], safe_travel_z=30.0)          # XY then descend, never diagonal
    m.move_abs(x=80, y=60, z=38)
    assert lines(m) == ["G1 X80.0000 Y60.0000 F3000", "G1 Z38.0000 F3000"], lines(m)

    m = Fake([50, 50, 45.0], safe_travel_z=30.0)          # pure Z move: no lift needed
    m.move_abs(z=48)
    assert lines(m) == ["G1 Z48.0000 F3000"], lines(m)

    try:
        Fake([50, 50, 10.0], safe_travel_z=30.0).move_abs(x=300)
        raise AssertionError("out-of-range X accepted")
    except MoveRefused:
        pass


def test_down_light_switched_on_when_opened():
    """open_cam('down') must run VISION_LIGHT_ON (from config.toml) before the camera starts."""
    from pnpvision import cli, config as C
    cfg = C.load(os.path.join(os.path.dirname(CAL), "config.toml"))
    events = []

    class FakeMR:
        def gcode(self, script):
            events.append(("gcode", script))

    class FakeCam:
        def __init__(self, url, w, h, name): self.name = name
        def start(self):
            events.append(("camera", self.name)); return self

    real_mr, real_cam, real_sleep = cli.make_mr, cli.SnapshotCamera, cli.time.sleep
    cli.make_mr, cli.SnapshotCamera, cli.time.sleep = (lambda cfg: FakeMR()), FakeCam, (lambda s: None)
    try:
        cli.open_cam(cfg, "down")
        assert events[0] == ("gcode", "VISION_LIGHT_ON\nM400"), events
        assert events[1] == ("camera", "down"), events
        events.clear()
        cli.open_cam(cfg, "up")                       # up light not wired: no G-code at all
        assert events == [("camera", "up")], events
    finally:
        cli.make_mr, cli.SnapshotCamera, cli.time.sleep = real_mr, real_cam, real_sleep


def test_snapshot_camera_waits_for_a_new_frame():
    """Fake ustreamer: the cached frame only changes every 100 ms. fresh() must return a
    frame whose timestamp is newer than the one cached when it was called."""
    import http.server, threading, time
    from pnpvision.camera import SnapshotCamera
    state = {"t0": time.monotonic()}

    def current():
        n = int((time.monotonic() - state["t0"]) / 0.1)          # frame number, 10 fps
        img = np.full((72, 128, 3), n % 250, np.uint8)
        return cv2.imencode(".jpg", img)[1].tobytes(), n

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data, n = current()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Timestamp", f"{n * 0.1:.6f}"); self.end_headers()
            self.wfile.write(data)
        def log_message(self, *a): pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/webcam/?action=snapshot"
        cam = SnapshotCamera(url, 128, 72, name="fake").start()
        for _ in range(3):
            _, before = current()
            f = cam.fresh(settle_s=0.0)
            assert int(f[0, 0, 0]) > before, "returned the frame that was cached before the call"
        try:
            SnapshotCamera(url, 1920, 1080, name="fake").start()
            raise AssertionError("resolution mismatch was not caught")
        except RuntimeError as e:
            assert "1920x1080" in str(e)
    finally:
        srv.shutdown()


class World:
    """Simulated down camera over a dot. A camera move m shifts the scene by M @ m."""
    def __init__(self, M_true, dot_mm=(3.0, -2.0)):
        self.M = np.array(M_true); self.cam = np.zeros(2); self.dot = np.array(dot_mm)
    def move_rel(self, dx, dy): self.cam += (dx, dy)
    def grab(self):
        c = np.array([640, 360]) + self.M @ (self.cam - self.dot)
        img = np.full((720, 1280), 220, np.uint8)
        S = 4   # sub-pixel drawing
        cv2.circle(img, (int(round(c[0] * 2**S)), int(round(c[1] * 2**S))), 40 * 2**S, 30, -1,
                   cv2.LINE_AA, shift=S)
        return img


def test_circle_finder():
    img = np.full((720, 1280), 220, np.uint8); cv2.circle(img, (500, 300), 40, 30, -1, cv2.LINE_AA)
    c = find_circle(img, 15, 120)
    assert abs(c.center_px[0] - 500) < 0.6 and abs(c.center_px[1] - 300) < 0.6


def test_calibrate_and_center():
    # true: 40 px/mm, axes flipped in Y and rotated ~5 deg, scene moves opposite to camera
    th = np.radians(5); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M_true = -40.0 * R @ np.diag([1, -1])
    w = World(M_true)
    # World.grab: dot px = center + M @ (cam - dot)  (verified by shifts s = M m for a camera move m)
    mm = calibrate_motion(w.grab, lambda f: find_circle(f, 15, 120), w.move_rel, 1.0, "camera")
    assert np.allclose(mm.M, M_true, atol=1.0), mm.M
    assert abs(mm.mm_per_px - 1 / 40) < 5e-4
    res = center_on(w.grab, lambda f: find_circle(f, 15, 120), w.move_rel, mm, (640, 360), tol_mm=0.02)
    assert res["converged"], res
    assert np.linalg.norm(w.cam - w.dot) < 0.05, (w.cam, w.dot)


def test_white_dot_on_sparkly_surface():
    """Like the datum's white dot under the LEDs: a saturated dot on a mid-grey surface full of
    bright glints. Automatic (Otsu) threshold lets glints merge with the dot; fixed 230 works."""
    rng = np.random.default_rng(0)
    img = rng.normal(116, 35, (1080, 1920)).clip(0, 255).astype(np.uint8)
    for _ in range(4000):                                   # glints, some touching the dot
        x, y = rng.integers(0, 1920), rng.integers(0, 1080)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 9)), int(rng.integers(160, 235)), -1)
    cv2.circle(img, (878, 634), 80, 254, -1, cv2.LINE_AA)
    c = find_circle(img, 15, 120, polarity="bright", threshold=230)
    assert c is not None and abs(c.center_px[0] - 878) < 1 and abs(c.center_px[1] - 634) < 1, c
    assert abs(c.radius_px - 80) < 3, c


def test_repeatability_finds_backlash():
    """Simulated toolhead with 0.04 mm X backlash and no Y error: the test must report
    a ~0.04 mm X direction gap, ~0 in Y, and keep every move inside the limits."""
    from pnpvision.servo import repeatability_test
    M_true = np.array([[-58.7, 0.5], [0.6, 59.2]]) * (40 / 59)     # fits the 1280x720 sim
    w = World(M_true, dot_mm=(0.0, 0.0))
    state = {"cmd": np.array([100.0, 100.0]), "last_dx": 1.0, "homes": 0}
    BACKLASH = 0.04
    limits = {"x": (0.0, 235.0), "y": (0.0, 140.0)}

    def move_abs(x=None, y=None, z=None):
        new = np.array([x, y], float)
        assert limits["x"][0] <= x <= limits["x"][1] and limits["y"][0] <= y <= limits["y"][1], new
        dx = new[0] - state["cmd"][0]
        if abs(dx) > 1e-9:
            state["last_dx"] = dx
        state["cmd"] = new
        # true camera position (relative to the dot at the start XY): lags when moving -X
        w.cam = new - (100.0, 100.0) + (0.0 if state["last_dx"] > 0 else -BACKLASH, 0.0)

    def home():
        state["homes"] += 1

    move_abs(100.0, 100.0)
    res = repeatability_test(w.grab, lambda f: find_circle(f, 15, 120), MotionMap(M_true, "camera"),
                             move_abs, lambda: (*state["cmd"], 10.0), n=16, radius_mm=(30, 80),
                             limits=limits, home=home, home_every=4, seed=1, log=lambda *a: None)
    s = res["summary"]
    assert state["homes"] == 4
    gx, gy = s["direction_gap_mm"]
    # far point on +X side -> returns moving -X -> lands BACKLASH short
    assert abs(gx - (-BACKLASH)) < 0.01, s
    assert abs(gy) < 0.01, s
    assert s["noise_floor_mm"][0] < 0.005, s


def test_visual_home_removes_homing_offset():
    """Sim: each home() shifts the frame by up to 2 mm (like sensorless homing). After
    visual_home, commanding the taught XY must put the camera back over the dot."""
    from pnpvision.servo import visual_home
    M_true = np.array([[-58.7, 0.5], [0.6, 59.2]]) * (40 / 59)
    taught = (15.2, 104.2)
    w = World(M_true, dot_mm=taught)              # dot sits at the taught XY (frame at teach time)
    st = {"cmd": np.array([10.0, 10.0]), "off": np.zeros(2)}
    rng = np.random.default_rng(3)

    def move_abs(x=None, y=None, z=None):
        st["cmd"] = np.array([x, y], float)
        w.cam = st["cmd"] + st["off"]            # physical = commanded + frame offset

    def home():
        st["off"] = rng.uniform(-2, 2, 2)
        move_abs(10.0, 10.0)

    def set_xy(x, y):                              # SET_KINEMATIC_POSITION: same physical spot
        phys = st["cmd"] + st["off"]
        st["cmd"] = np.array([x, y], float)
        st["off"] = phys - st["cmd"]

    locate = lambda f: find_circle(f, 15, 120)
    for _ in range(3):
        res = visual_home(w.grab, locate, MotionMap(M_true, "camera"), (640, 360), move_abs,
                          lambda: (*st["cmd"], 5.0), set_xy, taught, home=home, log=lambda *a: None)
        assert np.linalg.norm(res["homing_error_mm"]) < 3, res
        assert np.linalg.norm(st["off"]) < 0.02, st["off"]      # frame restored
        move_abs(60.0, 80.0); move_abs(*taught)
        assert np.linalg.norm(w.cam - w.dot) < 0.02


def test_measure_part_up_camera():
    img = np.full((720, 1280), 60, np.uint8)
    rect = ((650.0, 350.0), (80.0, 40.0), 30.0)      # 80x40 px part rotated 30 deg
    cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], 200)
    p = measure_part(img, (640, 360), 200)
    assert p is not None
    assert abs(p.center_px[0] - 650) < 1 and abs(p.center_px[1] - 350) < 1
    assert abs(p.long_px - 80) < 2 and abs(p.short_px - 40) < 2
    assert abs(p.angle_px_deg - 30) < 2
    # px->machine with a mirrored up camera (target moves): 40 px/mm, y flipped
    mm = MotionMap(np.diag([40.0, -40.0]), "target")
    off = mm.offset_mm(np.array(p.center_px) - (640, 360))
    assert np.allclose(off, [0.25, 0.25], atol=0.03), off     # (+10px,-10px) -> (+0.25,+0.25) mm
    assert abs(mm.direction_mm(30) - (-30)) < 2                # mirror flips angle sense


def test_eventlog_cph():
    log = EventLog(os.path.join(tempfile.mkdtemp(), "t.sqlite"))
    for i in range(11): log.log("place", ok=True, ts=1000 + i * 4.0)   # 4 s per part
    assert abs(log.cph() - 900) < 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
