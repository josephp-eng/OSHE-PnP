#!/usr/bin/env python3
"""
ChArUco capture + OpenCV calibration for the PnP down camera (headless Pi).

Grabs stills from the crowsnest snapshot URL instead of opening the camera
directly, so crowsnest keeps running and you can aim the board using the live
Mainsail webcam view in your browser.

Workflow:
  1. Open the live stream in a browser (http://<pi-ip>/) to aim the board.
  2. Jog the toolhead over the board; vary tilt and position every shot.
  3. In this script's terminal:
        <Enter>  capture a still (saved only if enough ChArUco corners found)
        k        calibrate from everything captured this run
        q        quit
  4. Open captures/<name>/annotated/*.png in VS Code to verify detections.

Board (from the calibrx.io generator):
  7 x 5 squares, 8 mm squares, 6 mm markers, DICT_6X6_250, start id 0.
After printing, replace the sizes below with what you MEASURE with calipers.

Needs OpenCV 4.7+ (uses cv2.aruco.CharucoDetector). Check with:
    python3 -c "import cv2; print(cv2.__version__)"
"""

import argparse
import glob
import os
import time
import urllib.request

import cv2
import numpy as np

# ---- Board configuration (match your printed board) ------------------------
SQUARES_X, SQUARES_Y = 7, 5          # squares, not inner corners
SQUARE_SIZE_MM = 8.0                 # measure the printed square with calipers
MARKER_SIZE_MM = 6.0                 # measure the printed marker with calipers
ARUCO_DICT = cv2.aruco.DICT_6X6_250
LEGACY_BOARD = False                 # flip to True if corners won't detect
# ----------------------------------------------------------------------------

MIN_CORNERS = 8                      # min ChArUco corners to keep/use a frame


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE_MM, MARKER_SIZE_MM, dictionary
    )
    if LEGACY_BOARD:
        board.setLegacyPattern(True)
    return board, dictionary


def grab(url):
    """Fetch one JPEG snapshot and decode to BGR. Returns frame or None."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = r.read()
    except Exception as e:
        print(f"  snapshot fetch failed: {e}")
        return None
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("  could not decode snapshot (is the URL serving a JPEG?)")
    return frame


def annotate(frame, cc, ci, mc, mi):
    vis = frame.copy()
    if mi is not None and len(mi) > 0:
        cv2.aruco.drawDetectedMarkers(vis, mc, mi)
    if ci is not None and len(ci) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 0, 255))
    return vis


def calibrate(image_dir, board, detector):
    paths = sorted(glob.glob(os.path.join(image_dir, "capture_*.png")))
    if len(paths) < 5:
        print(f"Need ~5+ captures (15-25 is better); found {len(paths)}.")
        return

    all_obj, all_img, img_size, used = [], [], None, 0
    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        cc, ci, mc, mi = detector.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        if n >= MIN_CORNERS:
            obj, imgp = board.matchImagePoints(cc, ci)
            if obj is not None and len(obj) >= 6:
                all_obj.append(obj)
                all_img.append(imgp)
                used += 1
        else:
            print(f"  skipped ({n} corners): {os.path.basename(p)}")

    if used < 5:
        print(f"Only {used} usable images; capture more varied views.")
        return

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, img_size, None, None
    )

    total = 0.0
    for i in range(len(all_obj)):
        proj, _ = cv2.projectPoints(all_obj[i], rvecs[i], tvecs[i], K, dist)
        total += cv2.norm(all_img[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total / len(all_obj)

    print("\n=== Calibration result ===")
    print(f"Images used:        {used} of {len(paths)}")
    print(f"Image size:         {img_size}")
    print(f"RMS error:          {rms:.4f} px   (aim for < ~0.5-1.0)")
    print(f"Mean reproj. error: {mean_err:.4f} px")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coeffs:\n{dist.ravel()}")

    name = os.path.basename(os.path.normpath(image_dir))
    out = os.path.join(image_dir, f"calibration_{name}.npz")
    np.savez(out, camera_matrix=K, dist_coeffs=dist, image_size=img_size, rms=rms)
    print(f"\nSaved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost/webcam/?action=snapshot",
                    help="crowsnest snapshot URL (try /webcam2/ for the other cam)")
    ap.add_argument("--name", default=None,
                    help="run name (default: timestamp); re-running resumes it")
    args = ap.parse_args()

    if tuple(map(int, cv2.__version__.split(".")[:2])) < (4, 7):
        print(f"OpenCV {cv2.__version__} is < 4.7; this script uses the newer "
              "ChArUco API. Tell me your version and I'll adapt it.")
        return

    board, _ = make_board()
    detector = cv2.aruco.CharucoDetector(board)

    run_name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join("captures", run_name)
    os.makedirs(out_dir, exist_ok=True)
    count = len(glob.glob(os.path.join(out_dir, "capture_*.png")))

    print(f"Run folder: {out_dir}")
    print(f"Snapshot:   {args.url}")
    print("Aim with the live stream in your browser, then here:")
    print("  <Enter> capture | k calibrate | q quit\n")

    while True:
        cmd = input(f"[{count} saved] > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "k":
            calibrate(out_dir, board, detector)
            continue
        frame = grab(args.url)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = detector.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        if n >= MIN_CORNERS:
            count += 1
            raw = os.path.join(out_dir, f"capture_{count:03d}.png")
            cv2.imwrite(raw, frame)
            ann_dir = os.path.join(out_dir, "annotated")
            os.makedirs(ann_dir, exist_ok=True)
            cv2.imwrite(os.path.join(ann_dir, f"capture_{count:03d}.png"),
                        annotate(frame, cc, ci, mc, mi))
            print(f"  saved {raw}  ({n} corners)")
        else:
            print(f"  not saved: only {n} corners (need {MIN_CORNERS}); "
                  "reposition / check focus / try LEGACY_BOARD")


if __name__ == "__main__":
    main()