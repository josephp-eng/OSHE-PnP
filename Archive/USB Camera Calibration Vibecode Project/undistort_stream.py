#!/usr/bin/env python3
"""
Serve an undistorted copy of a crowsnest webcam stream, for viewing in Mainsail.

Reads the MJPEG stream crowsnest already serves, undistorts every frame with a
calibration .npz from calib_capture.py, and re-serves it as MJPEG:

    http://<pi>:8090/stream     live view   (Mainsail "Stream URL")
    http://<pi>:8090/snapshot   one JPEG    (Mainsail "Snapshot URL")

This is for humans looking at the dashboard. Vision code should read the RAW
crowsnest stream and apply the calibration itself at full resolution; don't
undistort this already-undistorted stream a second time.

Run:
    ./venv/bin/python undistort_stream.py \
        --calib captures/cam_down/calibration_cam_down.npz

It only touches the camera stream while someone is watching (or asked for a
snapshot in the last few seconds); otherwise it sleeps at ~0% CPU.

Options worth knowing:
    --scale 0.5   output size relative to the camera frame (less Pi CPU)
    --alpha 0     0 = crop to only valid pixels (no black edges),
                  1 = keep the whole field of view (curved black borders)
    --fps 10      cap on frames processed per second (keeps CPU free for Klipper)
"""

import argparse
import http.server
import socketserver
import threading
import time
import urllib.request

import cv2
import numpy as np

BOUNDARY = b"undistortframe"


def mjpeg_frames(url):
    """Yield raw JPEG bytes from an MJPEG-over-HTTP stream, reconnecting on errors."""
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                buf = b""
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        if start < 0 or end < 0:
                            if start > 0:
                                buf = buf[start:]      # drop junk before next frame
                            break
                        yield buf[start:end + 2]
                        buf = buf[end + 2:]
        except Exception as e:
            print(f"source stream error: {e}; retrying in 2 s", flush=True)
            time.sleep(2)


class Undistorter:
    def __init__(self, calib_path, scale, alpha):
        d = np.load(calib_path)
        self.K = d["camera_matrix"]
        self.dist = d["dist_coeffs"]
        self.calib_size = tuple(int(v) for v in d["image_size"])
        self.scale, self.alpha = scale, alpha
        self.maps, self.size = None, None

    def _build(self, w, h):
        K = self.K.copy()
        if (w, h) != self.calib_size:
            # Same camera mode at a different resolution: scale the intrinsics.
            print(f"WARNING: stream is {w}x{h} but calibration was {self.calib_size[0]}x"
                  f"{self.calib_size[1]}; scaling it. Recalibrate if the aspect "
                  "ratio differs (a different sensor crop).", flush=True)
            K[0] *= w / self.calib_size[0]
            K[1] *= h / self.calib_size[1]
        out = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        newK, _ = cv2.getOptimalNewCameraMatrix(K, self.dist, (w, h), self.alpha, out)
        self.maps = cv2.initUndistortRectifyMap(K, self.dist, None, newK, out, cv2.CV_16SC2)
        self.size = (w, h)
        print(f"undistorting {w}x{h} -> {out[0]}x{out[1]} (alpha={self.alpha})", flush=True)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self._build(w, h)
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)


class Latest:
    """Newest encoded frame, plus who currently wants frames.

    The worker only pulls and processes the camera stream while someone is
    watching /stream or asked for /snapshot in the last IDLE_AFTER seconds,
    so the service costs ~nothing while the dashboard isn't open.
    """

    IDLE_AFTER = 5.0

    def __init__(self):
        self.jpg, self.seq = None, 0
        self.viewers, self.last_demand = 0, -1e9
        self.cond = threading.Condition()

    def put(self, jpg):
        with self.cond:
            self.jpg, self.seq = jpg, self.seq + 1
            self.cond.notify_all()

    def wait_newer(self, seq, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.jpg, self.seq

    def _wanted(self):
        return self.viewers > 0 or time.monotonic() - self.last_demand < self.IDLE_AFTER

    def demand(self, viewer_delta=0):
        with self.cond:
            self.viewers += viewer_delta
            self.last_demand = time.monotonic()
            self.cond.notify_all()

    def wanted(self):
        with self.cond:
            return self._wanted()

    def wait_until_wanted(self):
        with self.cond:
            self.cond.wait_for(self._wanted)


def worker(args, latest):
    und = Undistorter(args.calib, args.scale, args.alpha)
    min_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    while True:
        latest.wait_until_wanted()           # sleep (0% CPU) until someone watches
        print("viewer connected: processing frames", flush=True)
        frames = mjpeg_frames(args.source)
        last = 0.0
        try:
            for raw in frames:
                if not latest.wanted():
                    break                    # nobody watching: disconnect and idle
                now = time.monotonic()
                if now - last < min_dt:
                    continue                 # skipped before decoding, so ~free
                last = now
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ok, jpg = cv2.imencode(".jpg", und(frame),
                                       [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    latest.put(jpg.tobytes())
        finally:
            frames.close()                   # closes the connection to crowsnest
        print("no viewers: idle", flush=True)


def make_handler(latest):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("", "/stream"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                seq = latest.seq             # don't replay a stale frame from before
                latest.demand(+1)
                try:
                    while True:
                        jpg, new = latest.wait_newer(seq)
                        if new == seq or jpg is None:
                            continue         # timed out waiting; keep the connection
                        seq = new
                        self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                         + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass                     # viewer closed the page
                finally:
                    latest.demand(-1)
            elif path == "/snapshot":
                seq = latest.seq
                latest.demand()              # wake the worker if it was idle
                jpg, new = latest.wait_newer(seq)    # a fresh frame, not a stale one
                if new == seq:
                    jpg = latest.jpg         # timed out: fall back to the last frame
                if jpg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calib", required=True, help="calibration .npz from calib_capture.py")
    ap.add_argument("--source", default="http://localhost/webcam/?action=stream",
                    help="crowsnest MJPEG stream (use /webcam2/ for the second camera)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    latest = Latest()
    threading.Thread(target=worker, args=(args, latest), daemon=True).start()
    srv = Server(("0.0.0.0", args.port), make_handler(latest))
    print(f"serving http://0.0.0.0:{args.port}/stream  and  /snapshot", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
