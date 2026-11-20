#!/usr/bin/env python3
"""Yards capstone: calibrated, single-can rim inspection prototype.

Install: python -m pip install opencv-python numpy
Run:     python can_rim_inspection.py --camera 0 --profile 12oz --reference-mm 50

CALIBRATION
Use a measured circular reference (50 mm above is ONLY an example), placed
at the SAME HEIGHT as the rim being inspected. Fix camera position, focus,
exposure, resolution and lighting first. Press C, then click the reference's
LEFT, RIGHT, TOP, BOTTOM edges in that order. ENTER saves; R restarts; ESC
cancels. Use a large, sharp reference spanning a similar area to the can.
Do not calibrate using an unknown production can or an assumed nominal size.
The saved scale is NEVER recomputed from a can under inspection.

Create separate profiles for 12oz / 16oz / 19.2oz. Recalibrate after any
camera, lens, focus, height, zoom or resolution change. A profile name alone
does not prove the physical setup matches. Lens distortion must be negligible
over the inspection region, or images must be rectified upstream.

Replay: python can_rim_inspection.py --image can.png --profile 12oz \
            --json result.json --save annotated.png
Optional: --roi 100 80 700 700   (x y width height in original pixels)
          --calibration my_calibration.json

DECISION SCOPE
PASS means the visible rim meets the CONFIGURED PROTOTYPE limits on this
frame. It does NOT certify interior cleanliness, sidewall condition, sealing
performance or fitness for production. No rejector is controlled here.
FAIL means at least one valid measurement is clearly outside its limit.
UNINSPECTABLE means calibration, image quality, geometry or uncertainty does
not support a decision. It must never be treated as PASS by downstream code.

Review LIMITS below with Yards before use. Flange +/-0.20 mm is provisional:
the team design document uses +/-0.25 mm. Outer-diameter, ovality and local
deformation limits below are engineering STARTING POINTS, not customer specs.
Local width here is projected radial separation of visible boundaries; confirm
that it matches the manufacturer's flange-width definition. A top view cannot
reliably establish every out-of-plane bend. Validate on labeled physical cans.

The inspector intentionally uses no smoothing between frames or cans.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Limits:
    outer_min_mm: float = 56.65  # PROVISIONAL; customer confirmation required
    outer_max_mm: float = 57.15
    inner_min_mm: float = 52.15
    inner_max_mm: float = 52.65
    flange_min_mm: float = 1.90  # Canvas: 2.10 +/-0.20; design doc differs
    flange_max_mm: float = 2.30
    max_ovality_mm: float = 0.50  # max axis minus min axis, near-normal view
    max_local_deviation_mm: float = 0.30  # PROVISIONAL
    edge_uncertainty_px: float = 1.0  # per edge; increase if repeatability worse
    min_laplacian_variance: float = 35.0  # starting image-quality threshold
    max_saturated_fraction: float = 0.15
    min_angular_coverage: float = 0.95


LIMITS = Limits()


def result(status, *reasons, measurements=None):
    return {"status": status, "scope": "visible rim only; prototype limits",
            "reasons": list(reasons), "measurements": measurements or {},
            "limits": asdict(LIMITS)}


def load_calibration(path, profile, shape):
    try:
        c = json.loads(Path(path).read_text())
        if c["version"] != 1 or c["profile"] != profile:
            raise ValueError("Calibration profile/version does not match")
        if c["image_size"] != [shape[1], shape[0]]:
            raise ValueError("Resolution changed: recalibrate at this resolution")
        if not np.isfinite(c["pixels_per_mm"]) or c["pixels_per_mm"] <= 0:
            raise ValueError("Invalid calibration scale")
        if not 0 < c["scale_relative_uncertainty"] < 0.1:
            raise ValueError("Invalid calibration uncertainty")
        return c, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"Calibration unavailable: {exc}"


def calibrate(frame, reference_mm, reference_tolerance_mm, profile, path):
    """User selects a KNOWN reference; inspection code cannot change scale."""
    points = []
    window = "Calibration - frozen reference image"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.setMouseCallback(window, click)
    message = "Click LEFT, RIGHT, TOP, BOTTOM. ENTER save; R reset; ESC cancel"
    try:
        while True:
            display = frame.copy()
            for i, p in enumerate(points):
                cv2.circle(display, p, 4, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), p, 0, 0.6, (0, 255, 255), 2)
            cv2.putText(display, message, (10, 25), 0, 0.45, (0, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 255
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord("r"):
                points.clear()
            if key not in (10, 13) or len(points) != 4:
                continue
            left, right, top, bottom = np.array(points, dtype=float)
            horizontal = np.linalg.norm(right - left)
            vertical = np.linalg.norm(bottom - top)
            diameter = (horizontal + vertical) / 2
            center_error = np.linalg.norm((left + right - top - bottom) / 2)
            if (right[0] <= left[0] or bottom[1] <= top[1] or diameter < 150
                    or abs(horizontal - vertical) / max(diameter, 1) > 0.015
                    or center_error > diameter * 0.015
                    or abs(right[1] - left[1]) > diameter * 0.02
                    or abs(bottom[0] - top[0]) > diameter * 0.02):
                message = "Reference too small, tilted or clicks inconsistent. R to retry"
                continue
            # Conservative starting bound: 1 pixel uncertainty at each endpoint,
            # plus uncertainty in the physically measured reference diameter.
            uncertainty = 2.0 / diameter + reference_tolerance_mm / reference_mm
            calibration = {
                "version": 1, "profile": profile,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "image_size": [frame.shape[1], frame.shape[0]],
                "reference_mm": reference_mm,
                "reference_tolerance_mm": reference_tolerance_mm,
                "reference_points": points,
                "pixels_per_mm": diameter / reference_mm,
                "scale_relative_uncertainty": uncertainty,
                "note": "Valid only for unchanged optical setup and rim plane",
            }
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(calibration, indent=2))
            print(f"Calibration saved: {target}")
            return True
    finally:
        cv2.destroyWindow(window)


def candidates(gray, ppm):
    """Find plausible boundaries, not just the two largest objects in frame."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    # No closing: filling edge gaps could conceal cracks or missing rim sections.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found = []
    for contour in contours:
        if len(contour) < 50 or cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= gray.shape[1] - 1 or y + h >= gray.shape[0] - 1:
            continue
        ellipse = cv2.fitEllipse(contour)
        axes = np.array(ellipse[1], dtype=float)
        diameter_mm = float(np.mean(axes) / ppm)
        if not 40 < diameter_mm < 72 or min(axes) <= 0:
            continue
        if max(axes) / min(axes) > 1.25:
            continue  # cannot distinguish gross deformation/tilt confidently
        # Canny frequently yields duplicate contours on the same edge.
        if any(np.linalg.norm(np.array(ellipse[0]) - np.array(e["ellipse"][0])) < 2
               and np.max(np.abs(axes - np.array(e["ellipse"][1]))) < 3
               for e in found):
            continue
        found.append({"ellipse": ellipse, "contour": contour, "diameter_mm": diameter_mm})
    return found


def radial_profile(contour, center, bins=180):
    points = contour[:, 0, :].astype(float) - np.array(center)
    angles = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    indices = np.floor(angles * bins / (2 * np.pi)).astype(int) % bins
    radii = np.linalg.norm(points, axis=1)
    profile = np.full(bins, np.nan)
    for i in np.unique(indices):
        profile[i] = np.median(radii[indices == i])
    return profile


def local_deviation(contour, ellipse, ppm):
    center, axes, angle = ellipse
    pts = contour[:, 0, :].astype(float) - np.array(center)
    theta = np.deg2rad(angle)
    x = pts[:, 0] * np.cos(theta) + pts[:, 1] * np.sin(theta)
    y = -pts[:, 0] * np.sin(theta) + pts[:, 1] * np.cos(theta)
    direction = np.arctan2(y, x)
    a, b = np.array(axes) / 2
    expected = 1 / np.sqrt((np.cos(direction) / a)**2 + (np.sin(direction) / b)**2)
    return float(np.max(np.abs(np.hypot(x, y) - expected)) / ppm)


def inspect(frame, calibration, roi=None):
    display = frame.copy()
    if calibration is None:
        return result("UNINSPECTABLE", "No valid calibration; press C using a known reference"), display
    ppm = calibration["pixels_per_mm"]
    height, width = frame.shape[:2]
    if calibration["image_size"] != [width, height]:
        return result("UNINSPECTABLE", "Calibration resolution mismatch"), display
    if roi is None:
        roi = (width // 10, height // 10, width * 8 // 10, height * 8 // 10)
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
        return result("UNINSPECTABLE", "ROI outside image"), display
    cv2.rectangle(display, (x, y), (x+w, y+h), (180, 180, 180), 1)
    gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    found = candidates(gray, ppm)
    pairs = []
    for outer in found:
        for inner in found:
            if outer["diameter_mm"] <= inner["diameter_mm"]:
                continue
            gap = (outer["diameter_mm"] - inner["diameter_mm"]) / 2
            distance = np.linalg.norm(np.array(outer["ellipse"][0]) - inner["ellipse"][0]) / ppm
            # Broad identification gates, deliberately wider than acceptance limits.
            if not 0.5 < gap < 5 or distance > 1.0:
                continue
            inner_points = inner["contour"][::max(1, len(inner["contour"]) // 50), 0]
            if not all(cv2.pointPolygonTest(outer["contour"], tuple(map(float, p)), False) >= 0
                       for p in inner_points):
                continue
            pairs.append((outer, inner))
    if len(pairs) != 1:
        return result("UNINSPECTABLE", f"Expected one clear rim pair; found {len(pairs)}. Check ROI/lighting"), display
    outer, inner = pairs[0]
    center = outer["ellipse"][0]
    ro = radial_profile(outer["contour"], center)
    ri = radial_profile(inner["contour"], center)
    covered = np.isfinite(ro) & np.isfinite(ri)
    coverage = float(np.mean(covered))
    if coverage < LIMITS.min_angular_coverage:
        return result("UNINSPECTABLE", "Incomplete visible rim", measurements={"coverage": coverage}), display
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [outer["contour"]], -1, 255, -1)
    cv2.drawContours(mask, [inner["contour"]], -1, 0, -1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(mask) < 100:
        return result("UNINSPECTABLE", "Too little rim area"), display
    sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)[mask]))
    saturated = float(np.mean(gray[mask] >= 250))
    if sharpness < LIMITS.min_laplacian_variance or saturated > LIMITS.max_saturated_fraction:
        return result("UNINSPECTABLE", "Rim is blurred or overexposed; adjust lighting/focus/exposure",
                      measurements={"sharpness": sharpness, "saturated_fraction": saturated}), display
    widths = (ro[covered] - ri[covered]) / ppm
    measurements = {
        "outer_diameter_mm": outer["diameter_mm"],
        "inner_diameter_mm": inner["diameter_mm"],
        "minimum_flange_mm": float(np.min(widths)),
        "maximum_flange_mm": float(np.max(widths)),
        "outer_ovality_mm": float(np.ptp(outer["ellipse"][1]) / ppm),
        "inner_ovality_mm": float(np.ptp(inner["ellipse"][1]) / ppm),
        "outer_local_deviation_mm": local_deviation(outer["contour"], outer["ellipse"], ppm),
        "inner_local_deviation_mm": local_deviation(inner["contour"], inner["ellipse"], ppm),
        "coverage": coverage, "sharpness": sharpness, "saturated_fraction": saturated,
    }
    failed, uncertain = [], []

    def check(name, value, low, high, edge_factor=2):
        # Conservative guard band; this is an estimated bound, not a validated
        # metrology uncertainty budget. Check repeatability on physical samples.
        uncertainty = edge_factor * LIMITS.edge_uncertainty_px / ppm + abs(value) * calibration["scale_relative_uncertainty"]
        measurements[name + "_uncertainty_mm"] = uncertainty
        if value + uncertainty < low or value - uncertainty > high:
            failed.append(f"{name}: {value:.3f} mm outside [{low:.3f}, {high:.3f}]")
        elif value - uncertainty < low or value + uncertainty > high:
            uncertain.append(f"{name}: too close to limit for measurement uncertainty")

    check("outer_diameter", measurements["outer_diameter_mm"], LIMITS.outer_min_mm, LIMITS.outer_max_mm)
    check("inner_diameter", measurements["inner_diameter_mm"], LIMITS.inner_min_mm, LIMITS.inner_max_mm)
    check("minimum_flange", measurements["minimum_flange_mm"], LIMITS.flange_min_mm, float("inf"))
    check("maximum_flange", measurements["maximum_flange_mm"], -float("inf"), LIMITS.flange_max_mm)
    for boundary in ("outer", "inner"):
        check(boundary + "_ovality", measurements[boundary + "_ovality_mm"], -float("inf"), LIMITS.max_ovality_mm, 4)
        check(boundary + "_local_deviation", measurements[boundary + "_local_deviation_mm"], -float("inf"), LIMITS.max_local_deviation_mm, 2)
        shifted = (np.array((outer if boundary == "outer" else inner)["contour"]) + (x, y)).astype(np.int32)
        cv2.drawContours(display, [shifted], -1, (255, 200, 0), 1)
    if failed:
        verdict = result("FAIL", *failed, *uncertain, measurements=measurements)
    elif uncertain:
        verdict = result("UNINSPECTABLE", *uncertain, measurements=measurements)
    else:
        verdict = result("PASS", "Visible rim meets configured prototype limits", measurements=measurements)
    return verdict, display


def overlay(display, verdict):
    colors = {"PASS": (0, 200, 0), "FAIL": (0, 0, 255), "UNINSPECTABLE": (0, 190, 255)}
    cv2.putText(display, verdict["status"] + " - RIM ONLY / PROTOTYPE", (10, 28), 0, 0.6,
                colors[verdict["status"]], 2)
    for index, reason in enumerate(verdict["reasons"][:4]):
        cv2.putText(display, reason[:105], (10, 50 + index * 20), 0, 0.40, (0, 190, 255), 1)
    return display


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--profile", choices=["12oz", "16oz", "19.2oz"], default="12oz")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reference-mm", type=float)
    parser.add_argument("--reference-tolerance-mm", type=float, default=0.02)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--json", type=Path, help="Save the single-image result")
    parser.add_argument("--save", type=Path, help="Save the single-image annotated output")
    args = parser.parse_args()
    if args.reference_mm is not None and (not np.isfinite(args.reference_mm) or args.reference_mm <= 0):
        parser.error("--reference-mm must be a positive measured diameter")
    if not np.isfinite(args.reference_tolerance_mm) or args.reference_tolerance_mm < 0:
        parser.error("--reference-tolerance-mm must be nonnegative")
    path = args.calibration or Path(__file__).resolve().parent / f"calibration_{args.profile}.json"
    print("PROTOTYPE: confirm thresholds and validate with real cans. No rejector output.")
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"Cannot read image: {args.image}")
        calibration, error = load_calibration(path, args.profile, frame.shape)
        verdict, display = inspect(frame, calibration, args.roi)
        if error:
            verdict["reasons"] = [error]
        verdict["profile"] = args.profile
        payload = json.dumps(verdict, indent=2, allow_nan=False)
        print(payload)
        if args.json:
            args.json.write_text(payload)
        if args.save and not cv2.imwrite(str(args.save), overlay(display, verdict)):
            raise OSError(f"Could not save {args.save}")
        return
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        parser.error("Cannot open camera; try --camera 1")
    print("C: calibrate with reference in view. Q: quit. Center ONE can in the ROI.")
    cv2.namedWindow("Can rim inspection", cv2.WINDOW_NORMAL)
    calibration, last_shape = None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(json.dumps(result("UNINSPECTABLE", "Camera read failed; inspection stopped")))
                break
            if frame.shape != last_shape:
                calibration, error = load_calibration(path, args.profile, frame.shape)
                last_shape = frame.shape
                if error:
                    print(error)
            verdict, display = inspect(frame, calibration, args.roi)
            cv2.imshow("Can rim inspection", overlay(display, verdict))
            key = cv2.waitKey(1) & 255  # exactly one key read per live iteration
            if key == ord("q") or cv2.getWindowProperty("Can rim inspection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                if args.reference_mm is None:
                    print("Restart with --reference-mm YOUR_MEASURED_REFERENCE_DIAMETER")
                elif calibrate(frame, args.reference_mm, args.reference_tolerance_mm, args.profile, path):
                    calibration, error = load_calibration(path, args.profile, frame.shape)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
