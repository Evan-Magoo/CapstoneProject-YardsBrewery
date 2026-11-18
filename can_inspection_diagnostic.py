#!/usr/bin/env python3
"""Pi 5 / Hailo-8 inspection: independent capture, vision, inference and GUI.

Requirements: numpy, cv2, hailo_platform, can_grading_model.hef.
IMPORTANT: verify model class names/order and RGB preprocessing on real samples.
This code is for inspection/monitoring; it does not control a reject actuator.
"""
import os
import sys
import time
import threading
from queue import Queue, Empty, Full

import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

# ---------------------- CONFIGURATION ----------------------
HEF_PATH = 'can_grading_model.hef'
CLASS_NAMES = ('bad_cans', 'good_cans', 'no_cans')  # Verify against original .pt
GOOD_CLASS_INDEX = 1  # Verify against original .pt
CAMERA_INDEX = 0
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 720, 120
CAMERA_EXPOSURE = -4
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
PANEL_WIDTH = 560
DISPLAY_FPS = 30
SHOW_DASHBOARD = True
# Diagnostic modes: default raw preview; --full original HMI; --no-vision isolate CPU/Hailo
RAW_PREVIEW = '--full' not in sys.argv
DISABLE_VISION = '--no-vision' in sys.argv
ROI_SIZE = 600             # Original camera pixel coordinates
ROI_OFFSET_X = 50
ROI_OFFSET_Y = 0
DETECTION_SIZE = 300       # Reduced-resolution contour processing
MIN_CONTOUR_AREA = 30000   # Area in ORIGINAL camera pixel coordinates
MIN_CIRCULARITY = 0.85
TRIGGER_START, TRIGGER_END = 0.35, 0.5
TRACK_MATCH_DISTANCE = 70  # Original camera pixel coordinates
TRACK_MISSING_FRAMES = 7
HOLD_SECONDS = 5.0
INFERENCE_QUEUE_SIZE = 8
WINDOW_NAME = 'Live Can Inspection Dashboard'
# -----------------------------------------------------------

cv2.setNumThreads(2)
stop_event = threading.Event()
frame_queue = Queue(maxsize=1)  # Newest camera image (intentional drops)
inference_queue = Queue(maxsize=INFERENCE_QUEUE_SIZE)  # Every inspection job matters
result_queue = Queue()  # Never discard inference results
shared_lock = threading.Lock()
shared = {
    'capture_fps': 0.0, 'vision_fps': 0.0, 'display_fps': 0.0,
    'frame': None, 'frame_at': 0.0, 'vision_frame_at': 0.0,
    'roi_box': None, 'circles': (), 'thumb': None,
    'track_count': 0,
    'state': dict(total=0, passed=0, failed=0, pending=0, skipped=0,
                  status='Awaiting Can...', color=(140, 140, 160),
                  prediction='SCANNING...', confidence=0., area=0.,
                  circularity=0., latency=0.),
}


def keep_latest(queue, item):
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass


def camera_worker(cap):
    count, start = 0, time.monotonic()
    while not stop_event.is_set():
        ok, frame = cap.read()
        now = time.monotonic()
        if ok and frame is not None:
            count += 1
            # Publish live video DIRECTLY from capture. Rendering must never wait
            # for image analysis to complete; frames are immutable after read().
            with shared_lock:
                shared['frame'] = frame
                shared['frame_at'] = now
            keep_latest(frame_queue, (now, frame))
        else:
            time.sleep(.02)
        if now - start >= 1.0:
            with shared_lock:
                shared['capture_fps'] = count / (now - start)
            count, start = 0, now


def get_probabilities(raw):
    vals = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vals.size != len(CLASS_NAMES):
        raise ValueError(f'Expected {len(CLASS_NAMES)} classes; got {vals.shape}')
    if not np.all(np.isfinite(vals)):
        raise ValueError('Non-finite classifier output')
    if np.all(vals >= -1e-4) and np.all(vals <= 1.0001) and np.isclose(vals.sum(), 1., atol=.015):
        vals = np.clip(vals, 0, 1)
        return vals / max(float(vals.sum()), 1e-9)
    stable = vals - vals.max()
    probs = np.exp(stable)
    return probs / probs.sum()


def inference_worker(group, input_params, output_params, input_info, output_info):
    try:
        height, width, channels = map(int, input_info.shape)
        if channels != 3:
            raise ValueError(f'Expected 3-channel RGB NHWC HEF input, got {input_info.shape}')
        with group.activate():
            with InferVStreams(group, input_params, output_params) as pipeline:
                print('[INFO] Hailo pipeline ready', flush=True)
                while not stop_event.is_set():
                    try:
                        job_id, crop = inference_queue.get(timeout=.1)
                    except Empty:
                        continue
                    start = time.monotonic()
                    try:
                        img = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        tensor = np.ascontiguousarray(img[None, ...], dtype=np.uint8)
                        result = pipeline.infer({input_info.name: tensor})[output_info.name]
                        probabilities = get_probabilities(result)
                        result_queue.put((job_id, probabilities, (time.monotonic()-start)*1000, None))
                    except Exception as exc:
                        result_queue.put((job_id, None, (time.monotonic()-start)*1000, str(exc)))
                    finally:
                        inference_queue.task_done()
    except Exception as exc:
        result_queue.put((-1, None, 0., str(exc)))
        stop_event.set()


def vision_worker():
    """Owns tracks and counters; never calls imshow/waitKey."""
    state = dict(shared['state'])
    tracks, pending = {}, {}
    next_track_id = next_job_id = 0
    last_thumb, hold_until = None, 0.
    last_frame, last_roi_box, last_circles = None, None, ()
    vision_count, vision_window = 0, time.monotonic()
    vision_fps = 0.
    while not stop_event.is_set():
        now = time.monotonic()
        # Process all completed inference jobs; results are never dropped.
        while True:
            try:
                job_id, probs, latency, error = result_queue.get_nowait()
            except Empty:
                break
            if job_id == -1:
                print(f'[FATAL] Hailo worker: {error}', flush=True)
                stop_event.set()
                break
            if job_id not in pending:
                continue
            del pending[job_id]
            state['pending'] = len(pending)
            state['latency'] = latency if state['latency'] == 0 else .8*state['latency'] + .2*latency
            if error:
                state.update(status='FAIL - AI ERROR', prediction='AI ERROR',
                             color=(0, 60, 255), confidence=0.)
                state['total'] += 1
                state['failed'] += 1
                print(f'[ERROR] Inspection {job_id}: {error}', flush=True)
            else:
                idx = int(np.argmax(probs))
                state['confidence'] = float(probs[idx])*100
                state['prediction'] = CLASS_NAMES[idx].upper()
                if GOOD_CLASS_INDEX is None:
                    state.update(status='REVIEW - CLASS UNMAPPED', color=(140, 140, 160))
                elif idx == GOOD_CLASS_INDEX:
                    state['total'] += 1
                    state['passed'] += 1
                    state.update(status='PASS - GOOD CAN', color=(0, 255, 120))
                else:
                    state['total'] += 1
                    state['failed'] += 1
                    state.update(status='FAIL - BAD CLASS', color=(0, 60, 255))
            hold_until = now + HOLD_SECONDS

        try:
            captured_at, frame = frame_queue.get(timeout=.02)
            # Drain any backlog and process only the newest camera frame.
            while True:
                try:
                    captured_at, frame = frame_queue.get_nowait()
                except Empty:
                    break
        except Empty:
            frame = None
            captured_at = 0.0

        if frame is not None:
            vision_count += 1
            if now - vision_window >= 1.0:
                vision_fps = vision_count / (now - vision_window)
                vision_count, vision_window = 0, now
            raw_h, raw_w = frame.shape[:2]
            side = min(ROI_SIZE, raw_w, raw_h)

            roi_x = (raw_w - side) // 2 + ROI_OFFSET_X
            roi_y = (raw_h - side) // 2 + ROI_OFFSET_Y

            roi_x = max(0, min(roi_x, raw_w - side))
            roi_y = max(0, min(roi_y, raw_h - side))
            raw_roi = frame[roi_y:roi_y+side, roi_x:roi_x+side]
            detect_side = min(DETECTION_SIZE, side)
            # Track coordinates and thresholds remain in ORIGINAL ROI pixels.
            small = cv2.resize(raw_roi, (detect_side, detect_side), interpolation=cv2.INTER_AREA) if detect_side < side else raw_roi
            scale = side / detect_side
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 1.2)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            found = []
            for contour in contours:
                small_area = cv2.contourArea(contour)
                if small_area * scale * scale < MIN_CONTOUR_AREA:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                found.append((cx*scale, cy*scale, radius*scale, contour,
                              small_area*scale*scale, scale))

            old_ids = set(tracks)
            candidates = []
            for index, (cx, cy, _, _, _, _) in enumerate(found):
                for tid in old_ids:
                    track = tracks[tid]
                    # Increasing per-frame distance may be needed for fast-moving cans.
                    dist = np.hypot(cx-track['cx'], cy-track['cy'])
                    if dist < TRACK_MATCH_DISTANCE:
                        candidates.append((dist, index, tid))
            candidates.sort()
            assigned, matched = set(), set()
            for _, index, tid in candidates:
                if index in assigned or tid in matched:
                    continue
                cx, cy, radius, contour, area, s = found[index]
                tracks[tid].update(cx=cx, cy=cy, r=radius, contour=contour,
                                   area=area, scale=s, missing=0)
                assigned.add(index)
                matched.add(tid)
            for tid in old_ids-matched:
                tracks[tid]['missing'] += 1
                if tracks[tid]['missing'] > TRACK_MISSING_FRAMES:
                    del tracks[tid]
            for index, (cx, cy, radius, contour, area, s) in enumerate(found):
                if index not in assigned:
                    tracks[next_track_id] = dict(cx=cx, cy=cy, r=radius,
                                                 contour=contour, area=area,
                                                 scale=s, missing=0, graded=False)
                    next_track_id += 1

            lower, upper = side*TRIGGER_START, side*TRIGGER_END
            for tid, track in tracks.items():
                if track['missing'] or track['graded'] or not lower <= track['cy'] <= upper:
                    continue
                track['graded'] = True
                area = track['area']
                perimeter = cv2.arcLength(track['contour'], True)
                # Circularity is invariant under uniform resizing.
                area_small = cv2.contourArea(track['contour'])
                circ = 4*np.pi*area_small/(perimeter**2) if perimeter > 0 else 0.
                state.update(area=area, circularity=circ)
                last_thumb = raw_roi.copy()  # Full resolution for Hailo.
                hold_until = now+HOLD_SECONDS
                if circ < MIN_CIRCULARITY:
                    #state['total'] += 1
                    #state['failed'] += 1
                    #state.update(status='FAIL - DEFECTIVE RIM',
                    #             color=(0, 60, 255), prediction='DEFECTIVE RIM', confidence=0.)
                    continue
                else:
                    next_job_id += 1
                    try:
                        inference_queue.put_nowait((next_job_id, last_thumb))
                        pending[next_job_id] = tid
                        state['pending'] = len(pending)
                        state.update(status='AI INSPECTION PENDING',
                                     color=(140, 140, 160), prediction='PROCESSING...', confidence=0.)
                    except Full:
                        state['skipped'] += 1
                        state['total'] += 1
                        state['failed'] += 1
                        state.update(status='FAIL - AI QUEUE FULL', prediction='NOT INSPECTED',
                                     color=(0, 60, 255), confidence=0.)
            last_frame = frame  # Reference only; camera creates new frames.
            last_roi_box = (roi_x, roi_y, side)
            last_circles = tuple((tr['cx'], tr['cy'], tr['r']) for tr in tracks.values() if tr['missing'] == 0)

        if now > hold_until and not tracks and not pending:
            state.update(status='Awaiting Can...', prediction='SCANNING...',
                         color=(140, 140, 160), confidence=0.)
            last_thumb = None

        # Snapshot at most once per vision iteration, without large frame copying.
        with shared_lock:
            # Never overwrite the live camera frame from the slower vision worker.
            shared.update(roi_box=last_roi_box, circles=last_circles,
                          thumb=last_thumb, track_count=sum(t['missing']==0 for t in tracks.values()),
                          state=dict(state), vision_fps=vision_fps)
            if frame is not None:
                shared['vision_frame_at'] = captured_at


def make_panel_background():
    """One-time render of static panel borders, section titles and metric boxes."""
    p = np.empty((SCREEN_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    p[:] = (18, 18, 24)
    margin, width = 22, PANEL_WIDTH-44
    cv2.rectangle(p, (10,10), (PANEL_WIDTH-10,SCREEN_HEIGHT-10), (60,60,80), 2)
    cv2.putText(p, 'QUALITY INSPECTION HMI', (margin,44), cv2.FONT_HERSHEY_DUPLEX,.82,(255,255,255),2,cv2.LINE_AA)
    cv2.line(p,(margin,58),(PANEL_WIDTH-margin,58),(70,70,90),2)
    headings = [('SYSTEM STATUS',88),('PRODUCTION METRICS',196),
                ('PASS RATIO DISTRIBUTION',296),('OPENCV MEASUREMENTS',354),
                ('HAILO-8 AI CLASSIFIER',449),('SYSTEM PERFORMANCE',544),
                ('MODEL INPUT PREVIEW',639)]
    for txt,y in headings:
        cv2.putText(p,txt,(margin,y),cv2.FONT_HERSHEY_SIMPLEX,.52,(160,160,190),1,cv2.LINE_AA)
    third = (width-16)//3
    half = (width-10)//2
    boxes = [(margin+i*(third+8),206,third,70,title) for i,title in enumerate(('TOTAL','PASSED','FAILED'))]
    for y,titles in ((364,('CONTOUR AREA','CIRCULARITY')),
                     (459,('PREDICTION','CONFIDENCE')),
                     (554,('INFERENCE TIME','ACTIVE TRACKERS'))):
        boxes.extend([(margin+i*(half+10),y,half,65,title) for i,title in enumerate(titles)])
    for x,y,w,h,title in boxes:
        cv2.rectangle(p,(x,y),(x+w,y+h),(42,42,58),-1)
        cv2.rectangle(p,(x,y),(x+w,y+h),(75,75,95),1)
        cv2.putText(p,title,(x+9,y+19),cv2.FONT_HERSHEY_SIMPLEX,.38,(170,170,195),1,cv2.LINE_AA)
    cv2.rectangle(p,(margin,304),(PANEL_WIDTH-margin,334),(42,32,52),-1)
    cv2.rectangle(p,(margin,304),(PANEL_WIDTH-margin,334),(75,75,95),1)
    return p


def draw_panel(base, st, thumb, camera_fps, vision_fps, display_fps, active_count):
    panel = base.copy()  # Static text/layout drawn only once.
    margin, total_w = 22, PANEL_WIDTH-44
    color = st['color']
    green, red = color == (0,255,120), color == (0,60,255)
    bg = (20,60,35) if green else (30,30,75) if red else (35,35,48)
    border = (0,230,110) if green else (0,60,230) if red else (80,80,105)
    cv2.rectangle(panel,(margin,100),(PANEL_WIDTH-margin,172),bg,-1)
    cv2.rectangle(panel,(margin,100),(PANEL_WIDTH-margin,172),border,2)
    cv2.circle(panel,(margin+30,136),10,color,-1)
    status, size = st['status'], .72
    while size > .4 and cv2.getTextSize(status,cv2.FONT_HERSHEY_DUPLEX,size,2)[0][0] > total_w-70:
        size -= .04
    cv2.putText(panel,status,(margin+56,145),cv2.FONT_HERSHEY_DUPLEX,size,(255,255,255),2,cv2.LINE_AA)

    def value(x,y,w,h,val,col,scale=.65):
        val = str(val)
        while scale > .35 and cv2.getTextSize(val,cv2.FONT_HERSHEY_DUPLEX,scale,2)[0][0] > w-16:
            scale -= .04
        cv2.putText(panel,val,(x+9,y+h-17),cv2.FONT_HERSHEY_DUPLEX,scale,col,2,cv2.LINE_AA)

    third = (total_w-16)//3
    for i,(v,col) in enumerate(((st['total'],(240,240,250)),
                                (st['passed'],(60,255,120)),
                                (st['failed'],(100,100,255)))):
        value(margin+i*(third+8),206,third,70,v,col,.72)
    completed = st['passed']+st['failed']
    ratio = st['passed']/completed if completed else 0
    if st['passed']:
        cv2.rectangle(panel,(margin,304),(margin+int(total_w*ratio),334),(24,165,82),-1)
    ratio_text = f'Pass Rate: {ratio*100:.1f}%  ({st["passed"]} / {completed})' if completed else 'Pass Rate: --% (0 / 0)'
    tw = cv2.getTextSize(ratio_text,cv2.FONT_HERSHEY_DUPLEX,.48,1)[0][0]
    cv2.putText(panel,ratio_text,(margin+(total_w-tw)//2,324),cv2.FONT_HERSHEY_DUPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    half = (total_w-10)//2
    value(margin,364,half,65,f'{st["area"]:.0f}',(255,230,120))
    value(margin+half+10,364,half,65,f'{st["circularity"]:.2f}',(255,230,120))
    value(margin,459,half,65,st['prediction'],(110,230,255),.62)
    conf = f'{st["confidence"]:.1f}%' if st['confidence'] > 0 else '--'
    value(margin+half+10,459,half,65,conf,(110,230,255))
    value(margin,554,half,65,f'{st["latency"]:.1f} ms',(255,160,110),.62)
    value(margin+half+10,554,half,65,active_count,(255,160,110))
    size = min(total_w,SCREEN_HEIGHT-(639+45))
    x,y = margin+(total_w-size)//2,647
    cv2.rectangle(panel,(x,y),(x+size,y+size),(25,25,35),-1)
    if thumb is not None:
        panel[y:y+size,x:x+size] = cv2.resize(thumb,(size,size),interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel,(x,y),(x+size,y+size),color,3)
    else:
        cv2.putText(panel,'WAITING FOR CAN...',(x+8,y+size-8),cv2.FONT_HERSHEY_SIMPLEX,.52,(130,130,150),2)
    cv2.putText(panel,f'Capture {camera_fps:.1f} | Vision {vision_fps:.1f} | Display {display_fps:.1f} FPS',
                (margin,SCREEN_HEIGHT-36),cv2.FONT_HERSHEY_SIMPLEX,.44,(140,140,165),1,cv2.LINE_AA)
    cv2.putText(panel,f'Pending: {st["pending"]}  |  Skipped: {st["skipped"]}  |  Q: exit',
                (margin,SCREEN_HEIGHT-18),cv2.FONT_HERSHEY_SIMPLEX,.44,(140,140,165),1,cv2.LINE_AA)
    return panel


def render_camera(frame, roi_box, circles, display_fps):
    cam_w = SCREEN_WIDTH-PANEL_WIDTH
    h,w = frame.shape[:2]
    aspect = cam_w/SCREEN_HEIGHT
    if w/h > aspect:
        crop_w = max(1,int(h*aspect))
        ox,oy = (w-crop_w)//2,0
        crop = frame[:,ox:ox+crop_w]
    else:
        crop_h = max(1,int(w/aspect))
        ox,oy = 0,(h-crop_h)//2
        crop = frame[oy:oy+crop_h,:]
    ch,cw = crop.shape[:2]
    sx,sy = cam_w/cw, SCREEN_HEIGHT/ch
    canvas = cv2.resize(crop,(cam_w,SCREEN_HEIGHT),interpolation=cv2.INTER_LINEAR)
    if roi_box is not None:
        rx,ry,side = roi_box
        cv2.rectangle(canvas,(round((rx-ox)*sx),round((ry-oy)*sy)),
                      (round((rx+side-ox)*sx),round((ry+side-oy)*sy)),(0,255,120),2)
        for cx,cy,r in circles:
            cv2.circle(canvas,(round((rx+cx-ox)*sx),round((ry+cy-oy)*sy)),
                       max(1,round(r*sx)),(0,255,255),2)
    label = f'FPS: {display_fps:.1f}'
    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,.8,2)
    cv2.rectangle(canvas,(25,25),(53+tw,49+th),(20,20,25),-1)
    cv2.rectangle(canvas,(25,25),(53+tw,49+th),(70,70,95),2)
    cv2.putText(canvas,label,(39,33+th),cv2.FONT_HERSHEY_DUPLEX,.8,(0,255,120),2,cv2.LINE_AA)
    return canvas


def main():
    if not os.path.isfile(HEF_PATH):
        raise FileNotFoundError(f'HEF not found: {os.path.abspath(HEF_PATH)}')
    hef = HEF(HEF_PATH)
    inputs, outputs = hef.get_input_vstream_infos(), hef.get_output_vstream_infos()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError('Expected one input and one output VStream')
    input_info, output_info = inputs[0], outputs[0]
    if len(input_info.shape) != 3 or int(input_info.shape[2]) != 3:
        raise RuntimeError(f'Expected NHWC RGB input: {input_info.shape}')
    if int(np.prod(output_info.shape)) != len(CLASS_NAMES):
        raise RuntimeError(f'HEF output {output_info.shape} does not match CLASS_NAMES')
    if GOOD_CLASS_INDEX is not None and (GOOD_CLASS_INDEX < 0 or GOOD_CLASS_INDEX >= len(CLASS_NAMES)
                                          or CLASS_NAMES[GOOD_CLASS_INDEX] != 'good_cans'):
        raise RuntimeError('GOOD_CLASS_INDEX must refer to good_cans, or be None')
    print(f'[INFO] HEF input {input_info.shape}, output {output_info.shape}', flush=True)
    print('[WARNING] Verify class order and preprocessing on labeled cans before production.',flush=True)
    device = VDevice()
    cfg = ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe)
    group = device.configure(hef,cfg)[0]
    input_params = InputVStreamParams.make_from_network_group(group,quantized=False,format_type=FormatType.UINT8)
    output_params = OutputVStreamParams.make_from_network_group(group,quantized=False,format_type=FormatType.FLOAT32)
    cap = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')
    try:
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,.25)
        cap.set(cv2.CAP_PROP_EXPOSURE,CAMERA_EXPOSURE)
        print(f'[INFO] Camera requested {CAMERA_FPS} FPS, reports {cap.get(cv2.CAP_PROP_FPS):.1f} FPS')
        workers = [
            threading.Thread(target=camera_worker,args=(cap,),daemon=True,name='camera'),
            threading.Thread(target=inference_worker,
                             args=(group,input_params,output_params,input_info,output_info),
                             daemon=True,name='hailo'),
            *([] if DISABLE_VISION else [threading.Thread(target=vision_worker,daemon=True,name='vision')]),
        ]
        for worker in workers:
            worker.start()
        if SHOW_DASHBOARD:
            cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
            if not RAW_PREVIEW:
                cv2.setWindowProperty(WINDOW_NAME,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        base_panel = make_panel_background() if SHOW_DASHBOARD else None
        display_count, display_start, last_display = 0,time.monotonic(),0.
        display_fps = 0.
        interval = 1/max(DISPLAY_FPS,1)
        last_stats = 0.
        last_image_at = -1.
        newest_age_ms = 0.
        paint_ms = show_ms = key_ms = 0.
        actual_new_frames = 0
        print('[DIAG] Mode:', 'RAW 1280x720' if RAW_PREVIEW else 'FULL 1920x1080', '| Vision:', 'OFF' if DISABLE_VISION else 'ON', flush=True)
        # The main thread alone owns OpenCV GUI operations.
        while not stop_event.is_set():
            now = time.monotonic()
            if SHOW_DASHBOARD and now-last_display >= interval:
                with shared_lock:
                    frame = shared['frame']
                    frame_at = shared['frame_at']
                    vision_frame_at = shared['vision_frame_at']
                    roi_box = shared['roi_box']
                    circles = shared['circles']
                    thumb = shared['thumb']
                    st = shared['state']
                    capture_fps = shared['capture_fps']
                    vision_fps = shared['vision_fps']
                    active = shared['track_count']
                if frame is not None:
                    build_start = time.monotonic()
                    newest_age_ms = max(0., (build_start - frame_at) * 1000.)
                    if frame_at != last_image_at:
                        actual_new_frames += 1
                        last_image_at = frame_at
                    if RAW_PREVIEW:
                        # Deliberately avoid upscaling, hstack, draw_panel and interpolation.
                        canvas = frame.copy()  # Only one 1280x720 copy for diagnostic text.
                        cv2.putText(canvas, f'RAW / age {newest_age_ms:.0f} ms / display {display_fps:.1f} FPS',
                                    (15,35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                    else:
                        overlay_age = max(0.0, frame_at - vision_frame_at)
                        if overlay_age > 0.25:
                            roi_box, circles = None, ()
                        camera_canvas = render_camera(frame,roi_box,circles,display_fps)
                        panel = draw_panel(base_panel,st,thumb,capture_fps,vision_fps,display_fps,active)
                        canvas = np.hstack((camera_canvas,panel))
                    paint_ms = (time.monotonic() - build_start)*1000.
                    t_show = time.monotonic()
                    cv2.imshow(WINDOW_NAME,canvas)
                    show_ms = (time.monotonic() - t_show)*1000.
                    display_count += 1
                    if t_show-display_start >= 1.0:
                        display_fps = display_count/(t_show-display_start)
                        display_count,display_start = 0,t_show
                    last_display = time.monotonic()
            if SHOW_DASHBOARD:
                t_key = time.monotonic()
                key = cv2.waitKey(1) & 0xff
                key_ms = (time.monotonic()-t_key)*1000.
                if key == ord('q'):
                    break
                if time.monotonic()-last_stats >= 1.:
                    with shared_lock:
                        cfps = shared['capture_fps']
                        vfps = shared['vision_fps']
                        capture_age_ms = max(0., (time.monotonic()-shared['frame_at'])*1000.) if shared['frame_at'] else -1.
                    print(f'[PERF] capture={cfps:.1f} vision={vfps:.1f} display={display_fps:.1f} '
                          f'fresh_preview_frames={actual_new_frames}/s frame_age={capture_age_ms:.0f}ms '
                          f'build={paint_ms:.1f}ms imshow={show_ms:.1f}ms waitKey={key_ms:.1f}ms',flush=True)
                    actual_new_frames = 0
                    last_stats = time.monotonic()
            else:
                if now-last_stats >= 1.:
                    with shared_lock:
                        capture_fps,vision_fps = shared['capture_fps'],shared['vision_fps']
                    print(f'[PERF] Capture {capture_fps:.1f} FPS | Vision {vision_fps:.1f} FPS',flush=True)
                    last_stats = now
                time.sleep(.005)
    finally:
        stop_event.set()
        if 'workers' in locals():
            for worker in workers:
                worker.join(timeout=1.5)
        cap.release()
        if SHOW_DASHBOARD:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INFO] Stopped')
    finally:
        stop_event.set()
