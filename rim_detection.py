"""Offline rim inspection prototype (OpenCV + NumPy).

Run: python rim_detection.py
Next photo: n, Space, or q. Previous: p. Exit: Escape.

Calibrate with an undamaged can/reference whose OUTER rim diameter you measured:
  python rim_detection.py --calibrate test_images/reference.jpg \
      --reference-diameter-mm MEASURED_VALUE --profile bench-12oz
Then inspect using the same camera, focus, resolution and rim height:
  python rim_detection.py --profile bench-12oz
Use a separate profile/recalibration whenever the setup or rim height changes.

Batch report: python rim_detection.py --headless --output-dir rim_results
Optional labels CSV: filename,label (PASS or FAIL); use --labels labels.csv.
Thresholds are prototype settings, not validated brewery acceptance limits.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
ANGLES = np.linspace(0, 2 * np.pi, 360, endpoint=False)


def longest_run(mask):
    """Longest circular run (including across zero degrees)."""
    best = current = 0
    for flag in np.tile(mask, 2):
        current = current + 1 if flag else 0
        best = max(best, current)
    return min(best, len(mask))


def sample_rays(gray, center, radii):
    x = center[0] + np.cos(ANGLES)[:, None] * radii[None, :]
    y = center[1] + np.sin(ANGLES)[:, None] * radii[None, :]
    return cv2.remap(gray, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def measure_ring(gray, circle, roi):
    """Locate the outer bright-to-dark rim edge, then retain local deviations."""
    x, y, seed_radius = map(float, circle)
    radii = np.arange(max(5, seed_radius * .75), seed_radius * 1.30, .5)
    rays = sample_rays(gray, (x, y), radii).astype(np.float32)
    # Positive means brightness drops moving outwards, as at the outer flange.
    drops = rays[:, :-4] - rays[:, 4:]
    rr = radii[2:-2]
    support = np.mean(drops > 8, axis=0)
    peaks = [i for i in range(1, len(rr)-1)
             if support[i] >= .55 and support[i] >= support[i-1]
             and support[i] >= support[i+1]]
    if not peaks:
        return None
    # Use the outermost supported boundary, not an inner reflection.
    radius = float(rr[peaks[-1]])
    band = np.abs(rr-radius) <= max(4, radius*.06)
    br = rr[band]
    bd = drops[:, band]
    indices = np.argmax(bd, axis=1)
    observed = br[indices]
    strengths = bd[np.arange(360), indices]
    valid = strengths > 8
    # Refine the reference circle with robust fitting; inspect ALL supported
    # original points afterward so that the fit cannot erase a local dent.
    pts = np.column_stack((x + observed*np.cos(ANGLES), y + observed*np.sin(ANGLES)))
    keep = valid.copy()
    cx, cy, radius_fit = x, y, radius
    for _ in range(3):
        p = pts[keep]
        if len(p) < 180:
            return None
        matrix = np.column_stack((2*p[:, 0], 2*p[:, 1], np.ones(len(p))))
        a, b, c = np.linalg.lstsq(matrix, np.sum(p*p, axis=1), rcond=None)[0]
        cx, cy = float(a), float(b)
        radius_fit = float(np.sqrt(max(0, c+a*a+b*b)))
        residual = np.hypot(pts[:, 0]-cx, pts[:, 1]-cy) - radius_fit
        keep = valid & (np.abs(residual) < max(2, radius_fit*.025))
    if radius_fit < 5:
        return None
    # An oblique circular rim appears elliptical. Fit a robust ellipse rather
    # than calling that perspective effect a dent; retain local residuals.
    ellipse_keep = valid.copy()
    for _ in range(3):
        if ellipse_keep.sum() < 180:
            return None
        ellipse = cv2.fitEllipse(pts[ellipse_keep].astype(np.float32))
        (cx, cy), (axis1, axis2), angle = ellipse
        if min(axis1, axis2) <= 0:
            return None
        radians = np.deg2rad(angle)
        delta = pts - (cx, cy)
        u = delta[:, 0]*np.cos(radians) + delta[:, 1]*np.sin(radians)
        v = -delta[:, 0]*np.sin(radians) + delta[:, 1]*np.cos(radians)
        radius_fit = max(axis1, axis2)/2
        residual = (np.sqrt((u/(axis1/2))**2 + (v/(axis2/2))**2)-1)*radius_fit
        ellipse_keep = valid & (np.abs(residual) < max(3, radius_fit*.03))
    if np.hypot(cx-x, cy-y) > seed_radius*.15:
        return None
    coverage = float(valid.mean())
    residual = np.median(np.vstack((np.roll(residual, -1), residual, np.roll(residual, 1))), axis=0)
    defect = valid & (np.abs(residual) > max(3, radius_fit*.03))
    x0, y0, x1, y1 = roi
    clipped = (cx-radius_fit-3 < x0 or cy-radius_fit-3 < y0 or
               cx+radius_fit+3 >= x1 or cy+radius_fit+3 >= y1)
    # Focus is checked around the rim, rather than on the sharp desk background.
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    ring_mask = np.abs(np.hypot(xx-cx, yy-cy)-radius_fit) < 5
    focus = float(cv2.Laplacian(gray, cv2.CV_32F)[ring_mask].var())
    return dict(center=[cx, cy], radius_px=radius_fit, coverage=coverage,
                missing_arc_deg=int(longest_run(~valid)),
                dent_arc_deg=int(longest_run(defect)),
                max_deviation_px=float(np.max(np.abs(residual[valid]))),
                focus=focus, clipped=bool(clipped), ellipse=ellipse,
                axis_ratio=float(min(axis1, axis2)/max(axis1, axis2)),
                points=pts[valid].astype(int).tolist(),
                defect_points=pts[defect].astype(int).tolist())


def detect(image, roi_fraction):
    h, w = image.shape[:2]
    roi = tuple(int(v*s) for v, s in zip(roi_fraction, (w, h, w, h)))
    x0, y0, x1, y1 = roi
    if min(x1-x0, y1-y0) < 32:
        return None, 'Inspection region is too small', roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    crop = smooth[y0:y1, x0:x1]
    scale = min(h, w)
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1.2,
                              minDist=max(20, scale*.20), param1=100, param2=50,
                              minRadius=max(10, int(scale*.075)),
                              maxRadius=int(scale*.24))
    candidates = []
    if circles is not None:
        for x, y, r in circles[0]:
            result = measure_ring(smooth, (x+x0, y+y0, r), roi)
            if result is not None:
                candidates.append(result)
    candidates.sort(key=lambda a: (a['coverage'], -a['missing_arc_deg']), reverse=True)
    distinct = []
    for candidate in candidates:
        if not any(np.linalg.norm(np.array(candidate['center'])-other['center'])
                   < .65 * max(candidate['radius_px'], other['radius_px']) for other in distinct):
            distinct.append(candidate)
    if not distinct:
        return None, 'No reliable outer rim found', roi
    if len(distinct) > 1:
        return None, 'Multiple possible cans: isolate one inspection region', roi
    ring = distinct[0]
    reason = ''
    if ring['clipped']:
        reason = 'Rim crosses inspection boundary'
    elif ring['coverage'] < .94 or ring['missing_arc_deg'] > 10:
        reason = 'Incomplete rim: possible obstruction or weak contrast'
    elif ring['axis_ratio'] < .90:
        reason = 'Rim is tilted or oval: use a perpendicular camera and review shape'
    elif ring['focus'] < 35:
        reason = 'Rim is too blurred for inspection'
    return ring, reason, roi


def inspect_image(image, roi_fraction, calibration=None, dent_arc=6):
    ring, reason, roi = detect(image, roi_fraction)
    result = dict(status='UNABLE TO INSPECT', reason=reason, roi=roi,
                  diameter_mm=None, ring=ring)
    if reason:
        return result
    if calibration is None:
        result['reason'] = 'Calibration required for this camera setup and rim height'
        return result
    if calibration['image_size'] != [image.shape[1], image.shape[0]]:
        result['reason'] = 'Image resolution differs from calibration'
        return result
    diameter = 2*ring['radius_px']/calibration['pixels_per_mm']
    result['diameter_mm'] = float(diameter)
    if ring['dent_arc_deg'] >= dent_arc:
        result.update(status='FAIL', reason='Sustained rim-edge deviation; check for a dent')
        return result
    if abs(diameter-calibration['target_diameter_mm']) > calibration['tolerance_mm']:
        result.update(status='FAIL', reason='Outer rim diameter outside configured tolerance')
    else:
        result.update(status='PASS', reason='Configured diameter and rim-shape checks passed')
    return result


def render(image, result, filename, index, total):
    vis = image.copy()
    colors = {'PASS': (40, 190, 40), 'FAIL': (40, 40, 230), 'UNABLE TO INSPECT': (0, 190, 255)}
    color = colors[result['status']]
    x0, y0, x1, y1 = result['roi']
    cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 180, 180), 1)
    ring = result['ring']
    if ring:
        center = tuple(round(v) for v in ring['center'])
        cv2.ellipse(vis, ring['ellipse'], color, 2)
        for p in ring['defect_points']:
            cv2.circle(vis, tuple(p), 2, (255, 0, 255), -1)
    # Fixed footer keeps findings readable and clear of the rim.
    vis = cv2.resize(vis, (min(1100, image.shape[1]), round(image.shape[0]*min(1100,image.shape[1])/image.shape[1])))
    footer = np.zeros((145, vis.shape[1], 3), np.uint8)
    lines = [f'{index+1}/{total}  {filename}  |  {result["status"]}', result['reason']]
    if ring:
        diameter = f'{result["diameter_mm"]:.2f} mm' if result['diameter_mm'] is not None else 'uncalibrated'
        lines.append(f'Outer diameter: {diameter} | Edge coverage: {ring["coverage"]:.0%} | Deviation arc: {ring["dent_arc_deg"]} deg')
    lines.append('n / Space / q: next    p: previous    Esc: quit | Prototype - validate with labeled cans')
    for i, line in enumerate(lines):
        cv2.putText(footer, line, (12, 28+i*32), cv2.FONT_HERSHEY_SIMPLEX, .55, color if i==0 else (235,235,235), 1, cv2.LINE_AA)
    return np.vstack((vis, footer))


def load_calibration(path, profile):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    config = data.get(profile)
    if config is None:
        return None
    for key in ('pixels_per_mm', 'target_diameter_mm', 'tolerance_mm'):
        if not isinstance(config.get(key), (int, float)) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid calibration field: {key}')
    if not isinstance(config.get('image_size'), list) or len(config['image_size']) != 2:
        raise ValueError('Invalid calibration image_size')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, default=BASE/'test_images')
    parser.add_argument('--roi', type=float, nargs=4, default=(.25, .05, .75, .85), metavar=('LEFT','TOP','RIGHT','BOTTOM'), help='Normalized inspection region, default central bench area; set for your conveyor')
    parser.add_argument('--profile', default='default', help='Calibration name for a fixed camera setup and rim height')
    parser.add_argument('--calibration', type=Path, default=BASE/'rim_calibration.json')
    parser.add_argument('--calibrate', type=Path, help='Photo of known undamaged outer rim at inspection height')
    parser.add_argument('--reference-diameter-mm', type=float)
    parser.add_argument('--target-diameter-mm', type=float, help='Defaults to measured reference diameter when calibrating')
    parser.add_argument('--tolerance-mm', type=float, default=1.0, help='Prototype diameter tolerance saved during calibration; confirm with sponsor')
    parser.add_argument('--dent-arc-deg', type=int, default=6, help='Minimum consecutive degrees of edge deviation to flag; experimental')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Save annotated photos and CSV results')
    parser.add_argument('--labels', type=Path, help='CSV columns filename,label with confirmed PASS/FAIL labels')
    args = parser.parse_args()
    l,t,r,b=args.roi
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        parser.error('ROI must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1')
    if not 1 <= args.dent_arc_deg <= 360:
        parser.error('dent-arc-deg must be between 1 and 360')
    if args.calibrate:
        if args.reference_diameter_mm is None or not np.isfinite(args.reference_diameter_mm) or args.reference_diameter_mm <= 0:
            parser.error('Calibration requires a positive measured --reference-diameter-mm')
        target = args.target_diameter_mm if args.target_diameter_mm is not None else args.reference_diameter_mm
        if not np.isfinite(target) or target <= 0 or not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
            parser.error('Target diameter and tolerance must be positive finite values')
        image = cv2.imread(str(args.calibrate))
        if image is None:
            parser.error('Cannot read calibration photo')
        ring, reason, _ = detect(image, args.roi)
        if reason or ring['dent_arc_deg'] >= args.dent_arc_deg:
            parser.error('Reference rejected: '+(reason or 'rim shape is irregular'))
        configs = json.loads(args.calibration.read_text()) if args.calibration.exists() else {}
        configs[args.profile] = dict(pixels_per_mm=2*ring['radius_px']/args.reference_diameter_mm,
            target_diameter_mm=target, tolerance_mm=args.tolerance_mm,
            image_size=[image.shape[1],image.shape[0]], reference_image=str(args.calibrate.resolve()),
            reference_diameter_mm=args.reference_diameter_mm)
        args.calibration.write_text(json.dumps(configs, indent=2)+'\n')
        print(f'Saved calibration profile {args.profile!r} in {args.calibration}')
        return
    try:
        calibration = load_calibration(args.calibration, args.profile)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if not args.images.is_dir():
        parser.error(f'Image folder not found: {args.images}')
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not files:
        parser.error(f'No supported images in {args.images}')
    labels = {}
    if args.labels:
        with args.labels.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('label') not in ('PASS', 'FAIL') or not row.get('filename'):
                    parser.error('Labels require filename,label columns and PASS or FAIL values')
                labels[row['filename']] = row['label']
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; cache={}; index=0
    try:
        while index < len(files):
            path=files[index]
            if index not in cache:
                image=cv2.imread(str(path))
                result = (inspect_image(image,args.roi,calibration,args.dent_arc_deg) if image is not None
                          else dict(status='UNABLE TO INSPECT',reason='Unreadable image',ring=None,diameter_mm=None,roi=(0,0,1,1)))
                canvas=render(image,result,path.name,index,len(files)) if image is not None else np.zeros((300,900,3),np.uint8)
                if image is None:
                    cv2.putText(canvas,f'UNABLE TO INSPECT: {path.name}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,190,255),1)
                cache[index]=(result,canvas)
                ring=result['ring'] or {}
                rows.append(dict(filename=path.name,status=result['status'],reason=result['reason'],diameter_mm=result['diameter_mm'],coverage=ring.get('coverage'),dent_arc_deg=ring.get('dent_arc_deg'),expected=labels.get(path.name,'')))
                print(f'{path.name}: {result["status"]} - {result["reason"]}')
                if args.output_dir:
                    cv2.imwrite(str(args.output_dir/(path.name+'.inspection.png')),canvas)
            if args.headless:
                index+=1
                continue
            cv2.imshow('Rim inspection',cache[index][1])
            key=cv2.waitKey(30)&0xff
            if key==27 or cv2.getWindowProperty('Rim inspection',cv2.WND_PROP_VISIBLE)<1:
                break
            if key in (ord('n'),ord('q'),32): index+=1
            elif key==ord('p'): index=max(0,index-1)
    finally:
        if not args.headless: cv2.destroyAllWindows()
        if args.output_dir and rows:
            with (args.output_dir/'results.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
    if labels:
        labeled=[row for row in rows if row['expected']]
        print(f'Labeled images inspected: {len(labeled)}; '
              f'damaged accepted: {sum(r["expected"]=="FAIL" and r["status"]=="PASS" for r in labeled)}; '
              f'good rejected: {sum(r["expected"]=="PASS" and r["status"]=="FAIL" for r in labeled)}; '
              f'unable to inspect: {sum(r["status"]=="UNABLE TO INSPECT" for r in labeled)}')


if __name__ == '__main__':
    main()
