"""Detectors. All work on UNDISTORTED images and return sub-pixel pixel coordinates."""
from dataclasses import dataclass

import cv2
import numpy as np


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class Circle:
    center_px: tuple
    radius_px: float
    circularity: float


def find_circles(img, r_min_px, r_max_px, min_circularity=0.80,
                 polarity="both", external_only=False, threshold=None):
    """All round blobs, best circularity first. polarity: "bright" | "dark" | "both".
    external_only skips holes inside bigger shapes (e.g. the cells of the datum grid).
    threshold: None = automatic (Otsu) black/white cut-off for the whole frame; a number
    0-255 = fixed cut-off. Use a fixed high one (~230, polarity "bright") for a white dot on
    a sparkly or textured surface, where Otsu lets the glints merge into the dot."""
    g = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    h, w = g.shape
    modes = {"dark": [cv2.THRESH_BINARY_INV], "bright": [cv2.THRESH_BINARY],
             "both": [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]}[polarity]
    retr = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
    found = []
    for mode in modes:
        if threshold is None:
            _, bw = cv2.threshold(g, 0, 255, mode | cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(g, float(threshold), 255, mode)
        cnts, _ = cv2.findContours(bw, retr, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (np.pi * r_min_px**2 <= a <= np.pi * r_max_px**2):
                continue
            x, y, bw_, bh_ = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + bw_ >= w - 1 or y + bh_ >= h - 1:
                continue                       # touches border -> clipped
            p = cv2.arcLength(c, True)
            circ = 4 * np.pi * a / (p * p) if p else 0
            if circ < min_circularity:
                continue
            m = cv2.moments(c)
            found.append(Circle((m["m10"] / m["m00"], m["m01"] / m["m00"]),
                                float(np.sqrt(a / np.pi)), circ))
    return sorted(found, key=lambda c: -c.circularity)


def find_circle(img, r_min_px, r_max_px, min_circularity=0.80, polarity="both", threshold=None):
    """Most circular blob, dark-on-light or light-on-dark (see find_circles for options).
    Use for: a marker dot, the nozzle tip ring seen by the up camera, round fiducials."""
    found = find_circles(img, r_min_px, r_max_px, min_circularity, polarity, threshold=threshold)
    return found[0] if found else None


@dataclass
class PartMeasure:
    center_px: tuple
    long_px: float
    short_px: float
    angle_px_deg: float      # long-axis angle in image, (-90, 90]
    area_px: float


def find_parts_by_difference(ref, img, expected_px, threshold=None, size_tol=0.45, min_fill=0.6):
    """Parts that appeared between `ref` (scene without them) and `img`, taken from the SAME
    camera position. Works for any part colour on any background (pads, silkscreen, tape):
    only what changed counts. expected_px=(long, short) body size; returns PartMeasure list.
    Colour difference (largest channel change), so a tan part on green board counts even
    where the two have similar grey levels."""
    a = cv2.GaussianBlur(ref, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(img, (5, 5), 0).astype(np.int16)
    d = np.abs(b - a)
    d = d.max(axis=2) if d.ndim == 3 else d
    d = d.astype(np.uint8)
    t = threshold if threshold is not None else max(25.0, float(np.median(d)) + 20.0)
    mask = (d > t).astype(np.uint8) * 255
    el, es = expected_px
    kc = max(3, int(es * 0.3)) | 1                  # close gaps inside the part (e.g. its body
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  # matching the background's brightness)
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh, ang = rh, rw, ang + 90.0
        if rw <= 0 or rh <= 0:
            continue
        if abs(rw - el) / el > size_tol or abs(rh - es) / es > size_tol or area / (rw * rh) < min_fill:
            continue
        ang = (ang + 90.0) % 180.0 - 90.0
        out.append(PartMeasure((cx, cy), float(rw), float(rh), float(ang), float(area)))
    return out


def measure_part(img, nozzle_px, search_radius_px, min_area_px=200, expected_px=None):
    """Segment the part held on the nozzle (up camera) and fit a rotated rectangle.

    expected_px=(long, short) body size in pixels: try several black/white cut-offs and keep
    the blob whose size matches best. Needed for chip parts, whose bright metal ends and
    mid-grey body split apart under Otsu (it measured one end, 1.37 x 0.58 mm, of an 0805).
    The cut-offs are relative to the median of the crop (= the nozzle face behind the part).
    Without expected_px: Otsu, blob closest to the nozzle axis (first-cut behaviour)."""
    h, w = img.shape[:2]
    nx, ny = nozzle_px
    r = int(search_radius_px)
    x0, y0 = max(int(nx) - r, 0), max(int(ny) - r, 0)
    x1, y1 = min(int(nx) + r, w), min(int(ny) + r, h)
    crop = cv2.GaussianBlur(_gray(img)[y0:y1, x0:x1], (5, 5), 0)
    masks = []
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        masks.append(cv2.threshold(crop, 0, 255, mode | cv2.THRESH_OTSU)[1])
    if expected_px is not None:
        med = float(np.median(crop))
        k = np.ones((3, 3), np.uint8)
        for d in (8, 12, 16, 20, 25, 30, 40, 50):
            for mode, t in ((cv2.THRESH_BINARY, med + d), (cv2.THRESH_BINARY_INV, med - d)):
                if 0 < t < 255:
                    bw = cv2.threshold(crop, t, 255, mode)[1]
                    masks.append(cv2.morphologyEx(bw, cv2.MORPH_OPEN, k))
    best = None
    for bw in masks:
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area_px:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + cw >= crop.shape[1] - 1 or y + ch >= crop.shape[0] - 1:
                continue                       # blob is the background, not the part
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            if rw < rh:
                rw, rh, ang = rh, rw, ang + 90.0
            ang = (ang + 90.0) % 180.0 - 90.0
            d = np.hypot(cx + x0 - nx, cy + y0 - ny)
            if expected_px is None:
                score = d                      # prefer the blob closest to the nozzle axis
            else:
                el, es = expected_px
                fill = a / max(rw * rh, 1.0)   # 1 = solid rectangle; rings/crescents score low
                score = abs(rw - el) / el + abs(rh - es) / es + 0.2 * d / max(r, 1) + (1.0 - fill)
            if best is None or score < best[0]:
                best = (score, PartMeasure((cx + x0, cy + y0), float(rw), float(rh), float(ang), float(a)))
    return None if best is None else best[1]
