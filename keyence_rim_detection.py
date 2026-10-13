"""Rim-only inspection for the Keyence demo images. Requires OpenCV and NumPy.

    python keyence_rim_detection.py
    python keyence_rim_detection.py --headless
    python keyence_rim_detection.py --zip /path/to/OneDrive_2026-10-01.zip
    python keyence_rim_detection.py --images /path/to/pngs --labels labels.csv

Default input: keyence_images beside this script, otherwise the named ZIP in
Downloads. Archives are read as data, never installed or executed. No calibration
is needed for relative rim shape; this does NOT measure physical diameter or
inspect interior/body defects. PASS means only that configured rim checks passed.

Keys: n, Space, q = next; p = previous; Escape = exit. Results and overlays are
saved to keyence_results (override with --output-dir). Optional supplied labels
are used ONLY for comparison after prediction. Demo thresholds are experimental;
validate on independent cans before production use. Camera tilt can resemble ovality.
"""
import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEFAULTS = dict(max_deviation=.012, broad_deviation=.008,
                min_axis_ratio=.97, min_defect_arc=2.0)


def circular_run(mask):
    best = current = 0
    for value in np.tile(mask, 2):
        current = current + 1 if value else 0
        best = max(best, current)
    return min(best, len(mask))


def image_from_bytes(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def load_inputs(folder=None, archive=None):
    """Recover labels by file-content hash, since exported filenames differ."""
    labels = defaultdict(set)
    samples = []
    if folder is not None:
        if not folder.is_dir():
            raise ValueError(f'Image folder not found: {folder}')
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_TYPES:
                samples.append((path.name, path.read_bytes()))
    else:
        if not archive.is_file():
            raise ValueError(f'Archive not found: {archive}; use --images or --zip')
        with ZipFile(archive) as outer:
            for entry in sorted(outer.namelist()):
                normalized = entry.replace('\\', '/')
                if Path(normalized).suffix.lower() in IMAGE_TYPES:
                    samples.append((Path(normalized).name, outer.read(entry)))
                elif normalized.endswith('PSU 9.26.26 demo.zip'):
                    # This contains Keyence project data, not Python dependencies.
                    with ZipFile(BytesIO(outer.read(entry))) as inner:
                        for name in inner.namelist():
                            parts = name.replace('\\', '/').split('/')
                            if (len(parts) >= 2 and parts[-2] in ('PASS', 'FAIL')
                                    and Path(parts[-1]).suffix.lower() in IMAGE_TYPES):
                                labels[sha256(inner.read(name)).hexdigest()].add(parts[-2])
    if not samples:
        raise ValueError('No supported image files found')
    if len({name for name, _ in samples}) != len(samples):
        raise ValueError('Duplicate basenames: use a folder with unique image names')
    expected = {}
    for name, data in samples:
        choices = labels.get(sha256(data).hexdigest(), set())
        if len(choices) == 1:
            expected[name] = next(iter(choices))
        elif len(choices) > 1:
            print(f'Conflicting source labels for {name}; excluded from comparison')
    return samples, expected


def ellipse_residual(points, ellipse):
    (cx, cy), (a, b), angle = ellipse
    t = np.deg2rad(angle)
    delta = points - (cx, cy)
    u = delta[:, 0]*np.cos(t) + delta[:, 1]*np.sin(t)
    v = -delta[:, 0]*np.sin(t) + delta[:, 1]*np.cos(t)
    return np.sqrt((u/(a/2))**2 + (v/(b/2))**2)-1


def inspect_rim(image, config=None):
    """Pure image inference: no filename, label, or reference-class lookup."""
    config = DEFAULTS if config is None else config
    result = dict(status='UNABLE TO INSPECT', reason='', ellipse=None, contour=None,
                  defect_points=None, axis_ratio=None, p95_deviation=None,
                  max_deviation=None, defect_arc_deg=None)
    if image is None or min(image.shape[:2]) < 100:
        result['reason'] = 'Unreadable or insufficient-resolution image'
        return result
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for i, contour in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(contour)
        # The lamp reaches the frame borders; the can rim is a smaller, closed
        # bright annulus surrounding a large dark hole, near the image center.
        if not (x > 2 and y > 2 and x+cw < w-2 and y+ch < h-2):
            continue
        if not (.35*min(h,w) < cw < .80*min(h,w)
                and .35*min(h,w) < ch < .80*min(h,w)):
            continue
        if not (.2*w < x+cw/2 < .8*w and .2*h < y+ch/2 < .8*h):
            continue
        child = hierarchy[0, i, 2]
        area = cv2.contourArea(contour)
        if child < 0 or area <= 0:
            continue
        child_area = cv2.contourArea(contours[child])
        if .70 < child_area/area < .995 and len(contour) >= 20:
            candidates.append(contour)
    if len(candidates) != 1:
        result['reason'] = ('No complete bright rim found; check framing, contrast, or obstruction'
                            if not candidates else 'Multiple rim candidates; isolate one can')
        return result
    contour = candidates[0]
    points = contour[:, 0, :].astype(np.float32)
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        ellipse = cv2.fitEllipse(points[keep])
        if min(ellipse[1]) <= 0 or not np.all(np.isfinite(np.array(ellipse[0]+ellipse[1]))):
            result['reason'] = 'Unstable rim fit'
            return result
        residual = ellipse_residual(points, ellipse)
        # Fit the reference with the most consistent 80%, but evaluate the
        # FULL outline below so local dents are not discarded from inspection.
        keep = np.abs(residual) <= np.quantile(np.abs(residual), .80)
    center, axes, _ = ellipse
    angles = np.mod(np.arctan2(points[:,1]-center[1], points[:,0]-center[0]), 2*np.pi)
    order = np.argsort(angles)
    grid = np.linspace(0, 2*np.pi, 720, endpoint=False)
    profile = np.interp(grid, angles[order], residual[order], period=2*np.pi)
    # 1.5-degree median removes isolated contour pixel noise.
    profile = np.median(np.vstack((np.roll(profile, -1), profile, np.roll(profile, 1))), axis=0)
    defect = np.abs(profile) > config['max_deviation']
    arc = circular_run(defect)*360/len(grid)
    p95 = float(np.quantile(np.abs(profile), .95))
    peak = float(np.max(np.abs(profile)))
    ratio = float(min(axes)/max(axes))
    result.update(ellipse=ellipse, contour=contour, axis_ratio=ratio,
                  p95_deviation=p95, max_deviation=peak, defect_arc_deg=float(arc),
                  defect_points=points[np.abs(residual)>config['max_deviation']].astype(int))
    reasons = []
    if ratio < config['min_axis_ratio']:
        reasons.append('oval/deformed rim (or camera tilt)')
    if arc >= config['min_defect_arc']:
        reasons.append('local dent/notch in rim outline')
    if p95 > config['broad_deviation']:
        reasons.append('broad rim-shape irregularity')
    result.update(status='FAIL' if reasons else 'PASS',
                  reason='; '.join(reasons) if reasons else 'Rim checks passed; interior/body NOT inspected')
    return result


def render(image, result, name, index, total, expected):
    vis = image.copy() if image is not None else np.zeros((600,800,3),np.uint8)
    colors = {'PASS':(60,210,70),'FAIL':(40,50,240),'UNABLE TO INSPECT':(0,190,255)}
    color = colors[result['status']]
    if result['contour'] is not None:
        cv2.drawContours(vis,[result['contour']],-1,(255,220,0),3)
        cv2.ellipse(vis,result['ellipse'],color,3)
        for point in result['defect_points']:
            cv2.circle(vis,tuple(point),3,(255,0,255),-1)
    target_width = min(1050,vis.shape[1])
    vis = cv2.resize(vis,(target_width,round(vis.shape[0]*target_width/vis.shape[1])))
    footer = np.full((195,target_width,3),20,np.uint8)
    lines = [f'{index+1}/{total}  RIM {result["status"]}  |  Source label: {expected or "unknown"}',
             name, result['reason']]
    if result['axis_ratio'] is not None:
        lines.append(f'Axis ratio {result["axis_ratio"]:.4f} | Deviation p95 {result["p95_deviation"]*100:.2f}% | Defect arc {result["defect_arc_deg"]:.1f} deg')
    lines.extend(['Cyan: measured rim | Colored ellipse: reference | Magenta: local deviations',
                  'n / Space / q: next | p: previous | Esc: quit | RIM ONLY - experimental thresholds'])
    for i,line in enumerate(lines):
        cv2.putText(footer,line,(12,25+29*i),cv2.FONT_HERSHEY_SIMPLEX,.49,
                    color if i==0 else (235,235,235),1,cv2.LINE_AA)
    return np.vstack((vis,footer))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--zip',type=Path)
    inputs.add_argument('--images',type=Path)
    parser.add_argument('--labels',type=Path,help='Comparison CSV: filename,label (PASS or FAIL)')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=BASE/'keyence_results')
    parser.add_argument('--max-deviation',type=float,default=DEFAULTS['max_deviation'],help='Fraction of fitted radius, default .012')
    parser.add_argument('--broad-deviation',type=float,default=DEFAULTS['broad_deviation'],help='95th percentile fractional deviation limit, default .008')
    parser.add_argument('--min-axis-ratio',type=float,default=DEFAULTS['min_axis_ratio'])
    parser.add_argument('--min-defect-arc',type=float,default=DEFAULTS['min_defect_arc'],help='Degrees, default 2')
    args=parser.parse_args()
    config={key:getattr(args,key) for key in DEFAULTS}
    if not all(np.isfinite(v) and v>0 for v in config.values()):
        parser.error('All thresholds must be positive finite numbers')
    if args.min_axis_ratio>1 or args.min_defect_arc>360:
        parser.error('Axis ratio must be <=1 and defect arc <=360')
    folder=args.images
    archive=args.zip
    if folder is None and archive is None:
        folder=BASE/'keyence_images' if (BASE/'keyence_images').is_dir() else None
        archive=Path.home()/'Downloads/OneDrive_2026-10-01.zip' if folder is None else None
    try:
        samples,expected=load_inputs(folder,archive)
        if args.labels:
            with args.labels.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    if not row.get('filename') or row.get('label') not in ('PASS','FAIL'):
                        raise ValueError('Labels require filename,label columns and PASS/FAIL values')
                    expected[row['filename']]=row['label']
    except (OSError,ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cache={};rows={};index=0
    window='Keyence - rim inspection'
    try:
        while index<len(samples):
            name,data=samples[index]
            if index not in cache:
                image=image_from_bytes(data)
                result=inspect_rim(image,config)
                canvas=render(image,result,name,index,len(samples),expected.get(name,''))
                cache[index]=canvas
                rows[index]={key:result[key] for key in ('status','reason','axis_ratio','p95_deviation','max_deviation','defect_arc_deg')}
                rows[index]={'filename':name,**rows[index],'source_label':expected.get(name,'')}
                if not cv2.imwrite(str(args.output_dir/(name+'.inspection.png')),canvas):
                    raise OSError('Could not save inspection image')
                print(f'{index+1}/{len(samples)} {name}: RIM {result["status"]} - {result["reason"]}')
            if args.headless:
                index+=1
                continue
            cv2.imshow(window,cache[index])
            key=cv2.waitKey(30)&0xff
            if key==27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
            if key in (ord('n'),ord('q'),32): index+=1
            elif key==ord('p'): index=max(0,index-1)
    finally:
        if not args.headless: cv2.destroyAllWindows()
        if rows:
            ordered=[rows[i] for i in sorted(rows)]
            with (args.output_dir/'results.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(ordered[0]))
                writer.writeheader();writer.writerows(ordered)
            comparison=Counter(f'{r["source_label"]} -> {r["status"]}' for r in ordered if r['source_label'])
            summary=dict(images_available=len(samples),images_inspected=len(rows),
                         statuses=dict(Counter(r['status'] for r in ordered)),
                         source_label_comparison=dict(comparison),thresholds=config,
                         scope='Rim shape only; no diameter calibration or interior/body inspection. Demo evaluation, not independent validation.')
            (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(summary,indent=2))
    print(f'Results saved to {args.output_dir}')


if __name__=='__main__':
    main()
