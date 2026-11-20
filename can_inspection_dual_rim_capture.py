import json
import csv
import ctypes
import os
import sys
import time
import threading
import subprocess
import re
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full

import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

# [DB] inspection_db.py lives next to this file. It is imported INSIDE the
# db_writer thread (not here), because sqlite3 connections belong to the thread
# that opens them and inspection_db opens its connection at import time.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------- CONFIGURATION ----------------------
HEF_PATH = 'can_grading_model.hef'
CLASS_NAMES = ('bad_cans', 'good_cans', 'no_cans')  # Verify against original .pt
GOOD_CLASS_INDEX = 1  # Verify against original .pt
CAMERA_INDEX = 0
# Preserve the original 1920x1080 camera framing and 900x900 model ROI.
# ELP verified on this Pi: MJPEG 1920x1080 at approximately 94.77 decoded FPS with buffer=4.
# Optional overrides: --camera-size 1920x1080 --camera-fps 120
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1920, 1080, 120
CAMERA_FORMAT = "MJPG"
for _arg_index, _arg in enumerate(sys.argv):
    if _arg == '--camera-size' and _arg_index + 1 < len(sys.argv):
        _parts = sys.argv[_arg_index + 1].lower().split('x')
        if len(_parts) != 2:
            raise ValueError('--camera-size expects WIDTHxHEIGHT')
        CAMERA_WIDTH, CAMERA_HEIGHT = map(int, _parts)
        if CAMERA_HEIGHT < 900:
            raise ValueError('Camera height must be >=900 to preserve the 900x900 ROI')
    elif _arg == '--camera-fps' and _arg_index + 1 < len(sys.argv):
        CAMERA_FPS = int(sys.argv[_arg_index + 1])
CAMERA_EXPOSURE = None  # Do not force -4: it can cause long exposure / low FPS on V4L2
CAMERA_TEST = '--camera-test' in sys.argv
CAMERA_AUTO_MODE = False  # Never silently change the trained camera framing
CAMERA_EXPOSURE_US = None
if '--exposure-us' in sys.argv:
    CAMERA_EXPOSURE_US = int(sys.argv[sys.argv.index('--exposure-us') + 1])
    if CAMERA_EXPOSURE_US <= 0:
        raise ValueError('--exposure-us must be positive')
SCREEN_WIDTH, SCREEN_HEIGHT = 1024, 600  # Fallback only; auto-detected below
PANEL_WIDTH = 360
DISPLAY_FPS = 24  # Limit costly GUI resizing to leave CPU time for detection
SHOW_DASHBOARD = True
# Diagnostic modes: default raw preview; --full original HMI; --no-vision isolate CPU/Hailo
RAW_PREVIEW = '--full' not in sys.argv
DISABLE_VISION = '--no-vision' in sys.argv
DB_ENABLED = '--no-db' not in sys.argv   # [DB] --no-db runs without SQLite logging
SAVE_MODEL_IMAGES = '--no-save-images' not in sys.argv  # Save exact RGB model inputs as PNG
RIM_FAIL_ENABLED = True    # [DB] True = low circularity counts + logs as FAIL 'DEFECTIVE RIM' (like v2)
ROI_SIZE = 900             # Original camera pixel coordinates
ROI_OFFSET_X = 50
ROI_OFFSET_Y = 0
DETECTION_SIZE = 300       # Reduced-resolution contour processing
MIN_CONTOUR_AREA = 14000   # Accept partial/damaged rims without requiring a complete circle
MAX_CONTOUR_AREA = 145000  # Allow some reflection variation without accepting giant edges
MIN_CAN_RADIUS = 45        # Accept smaller partial outlines at high conveyor speeds
MAX_CAN_RADIUS = 190       # Still reject oversized yellow contours
MIN_CONTOUR_FILL = 0.24   # Damaged/partly missing rim edges can have a lower fill ratio
# Outer-rim check is independent from the small inner/bottom-rim check.
# Areas and radii below are expressed in ORIGINAL 900x900 ROI pixels.
OUTER_MIN_AREA = 220000
OUTER_MAX_AREA = 620000
OUTER_MIN_RADIUS = 215
OUTER_MAX_RADIUS = 465
RIM_CENTER_ALIGNMENT_PX = 45  # Maximum distance between independently detected rim centers
INNER_ONLY_REVIEW_FRAMES = 3  # Flag persistent missing outer rim; never automatically fail solely on absence
MIN_CIRCULARITY = 0.85
# Virtual counting line in normalized ROI coordinates.
# +1 = cans move toward larger image Y (top to bottom); -1 = bottom to top.
COUNT_LINE_FRACTION = 0.43
AI_CAPTURE_FRACTION = 0.50  # Full original ROI captured closest to its actual center.
COUNT_DIRECTION = 1 if "--direction-down" in sys.argv else -1
# Used only for tracks first detected on/near the line.
COUNT_INITIAL_MOTION_PIXELS = 10
COUNT_ENTRY_BAND_PIXELS = 36  # New tracks starting near the line can still be counted after confirmation
VISION_QUEUE_SIZE = 6  # Bound queue age to reduce delayed counting / preview lag.
MAX_TRACK_ASSOCIATION_AGE = 0.45  # Old tracks kept for stop/start but don't hijack new cans.
CAPTURE_UPDATE_PIXELS = 0.0  # No smoothing/deadband in selecting the best frame.
CENTER_CAPTURE_TOLERANCE = 30  # Pixels around the ROI center that qualify as centered.

TRACK_MATCH_DISTANCE = 155  # Smaller search avoids jumping between adjacent cans
TRACK_MISSING_SECONDS = 5.0  # Keep graded identity through stops/brief contour loss
TRACK_SMOOTHING = 0.70  # Display only; adaptive smoothing keeps fast cans responsive
TRACK_CONFIRM_FRAMES = 2   # Reject single-frame noise; keep for stationary cans
# No global cooldown: counts are based on each can's motion, not passage of time.
HOLD_SECONDS = 5.0
INFERENCE_QUEUE_SIZE = 8
WINDOW_NAME = 'Live Can Inspection Dashboard'
# -----------------------------------------------------------

cv2.setNumThreads(2)
stop_event = threading.Event()
frame_queue = Queue(maxsize=VISION_QUEUE_SIZE)  # FIFO: vision never skips queued frames
frame_queue_stats_lock = threading.Lock()
frame_queue_stats = {'overflow': 0, 'high_watermark': 0}
inference_queue = Queue(maxsize=INFERENCE_QUEUE_SIZE)  # Every inspection job matters
result_queue = Queue()  # Never discard inference results
db_queue = Queue()  # [DB] Rows waiting for the SQLite writer thread (never dropped)
image_save_queue = Queue(maxsize=256)  # Dedicated disk writer; backpressure instead of silent image loss
run_info = {'session_start': None, 'fatal_error': None, 'prior_resets': dict(graded=0, passed=0, failed=0)}  # [DB] session report inputs
shared_lock = threading.Lock()
shared = {
    'capture_fps': 0.0, 'vision_fps': 0.0, 'display_fps': 0.0,
    'frame': None, 'frame_at': 0.0, 'vision_frame_at': 0.0,
    'roi_box': None, 'circles': (), 'rim_marks': (), 'thumb': None,
    'track_count': 0,
    'state': dict(total=0, passed=0, failed=0, pending=0, skipped=0,
                  status='Awaiting Can...', color=(140, 140, 160),
                  prediction='SCANNING...', confidence=0., area=0.,
                  circularity=0., latency=0.,
                  graded=0),  # [DB] cans that reached the trigger window = image_count
}


# ---------------------- CLICKABLE HMI SETTINGS ----------------------
# These limits are deliberately bounded to reduce accidental unsafe tuning.
# The settings are changed live, without restarting the Hailo pipeline.
SETTINGS_SPEC = (
    ('ROI_OFFSET_X',        'ROI horizontal',       10,   -500, 500),
    ('ROI_OFFSET_Y',        'ROI vertical',         10,   -500, 500),
    ('COUNT_LINE_FRACTION', 'Count line %',        .02,    .10, .90),
    ('AI_CAPTURE_FRACTION', 'AI capture %',         .02,    .10, .90),
    ('MIN_CONTOUR_AREA',    'Min inner area',   2000,   1000, 140000),
    ('MAX_CONTOUR_AREA',    'Max inner area',   5000,  20000, 300000),
    ('MIN_CAN_RADIUS',      'Min inner radius',       5,     10, 185),
    ('MAX_CAN_RADIUS',      'Max inner radius',     5,     60, 300),
    ('OUTER_MIN_AREA',      'Outer min area',   10000,  120000, 600000),
    ('OUTER_MAX_AREA',      'Outer max area',   10000,  320000, 800000),
    ('OUTER_MIN_RADIUS',    'Outer min radius',    10,     150, 450),
    ('OUTER_MAX_RADIUS',    'Outer max radius',    10,     250, 550),
    ('RIM_CENTER_ALIGNMENT_PX','Rim align tolerance',5, 10, 120),
    ('INNER_ONLY_REVIEW_FRAMES','Inner only frames', 1, 1, 15),
    ('TRACK_MATCH_DISTANCE','Track distance',     10,     40, 300),
    ('MIN_CIRCULARITY',     'Rim circularity',    .02,    .40, 1.00),
    ('TRACK_CONFIRM_FRAMES','Confirm frames',      1,      1, 5),
)
DEFAULT_SETTINGS = {key: globals()[key] for key, *_ in SETTINGS_SPEC}
ui_state = {'tab': 'inspection', 'reset_confirm_until': 0.0,
            'defaults_confirm_until': 0.0,
            'notice': '', 'notice_until': 0.0,
            'selected_setting': 'COUNT_LINE_FRACTION'}
reset_request = threading.Event()


def show_ui_notice(message, duration=3.0):
    ui_state['notice'] = message
    ui_state['notice_until'] = time.monotonic() + duration


def settings_geometry():
    # Comfortable spacing on 1080p and all options visible on 1024x600.
    top = 102 if SCREEN_HEIGHT <= 850 else 124
    # Fit both inner- and outer-rim controls above the reset buttons.
    step = max(18, min(44 if SCREEN_HEIGHT > 850 else 26,
                       (SCREEN_HEIGHT - 245 - top) // max(1, len(SETTINGS_SPEC)-1)))
    return top, step


def on_dashboard_mouse(event, x, y, flags, userdata):
    """OpenCV mouse callback. All inspection counters are owned by vision_worker."""
    if event != cv2.EVENT_LBUTTONDOWN or RAW_PREVIEW:
        return
    local_x = x - (SCREEN_WIDTH-PANEL_WIDTH)
    if not (0 <= local_x < PANEL_WIDTH):
        return
    if 9 <= y <= 49:
        ui_state['tab'] = 'inspection' if local_x < PANEL_WIDTH//2 else 'settings'
        return
    if ui_state['tab'] != 'settings':
        return
    top, step = settings_geometry()
    for i, (key, label, delta, minimum, maximum) in enumerate(SETTINGS_SPEC):
        center = top+i*step
        if not center-max(9,step//2) <= y <= center+max(9,step//2):
            continue
        # Two actual button regions at the right; ignore clicks on labels.
        minus_center, plus_center = PANEL_WIDTH-87, PANEL_WIDTH-30
        if abs(local_x-minus_center) > 21 and abs(local_x-plus_center) > 21:
            return
        direction = -1 if abs(local_x-minus_center) <= 21 else 1
        ui_state['selected_setting'] = key
        old = globals()[key]
        new = max(minimum, min(maximum, old + direction*delta))
        if key in ('MIN_CONTOUR_AREA','MAX_CONTOUR_AREA','OUTER_MIN_AREA','OUTER_MAX_AREA'):
            new = int(new)
            if key == 'MIN_CONTOUR_AREA' and new >= MAX_CONTOUR_AREA:
                show_ui_notice('Min area must be below max area')
                return
            if key == 'MAX_CONTOUR_AREA' and new <= MIN_CONTOUR_AREA:
                show_ui_notice('Max area must exceed min area')
                return
            if key == 'OUTER_MIN_AREA' and new >= OUTER_MAX_AREA:
                show_ui_notice('Outer minimum must be below maximum')
                return
            if key == 'OUTER_MAX_AREA' and new <= OUTER_MIN_AREA:
                show_ui_notice('Outer maximum must exceed minimum')
                return
        if key in ('MIN_CAN_RADIUS','MAX_CAN_RADIUS','OUTER_MIN_RADIUS','OUTER_MAX_RADIUS'):
            new = int(new)
            if key == 'MIN_CAN_RADIUS' and new >= MAX_CAN_RADIUS:
                show_ui_notice('Min radius must be below max radius')
                return
            if key == 'MAX_CAN_RADIUS' and new <= MIN_CAN_RADIUS:
                show_ui_notice('Max radius must exceed min radius')
                return
            if key == 'OUTER_MIN_RADIUS' and new >= OUTER_MAX_RADIUS:
                show_ui_notice('Outer minimum radius must be lower')
                return
            if key == 'OUTER_MAX_RADIUS' and new <= OUTER_MIN_RADIUS:
                show_ui_notice('Outer maximum radius must be higher')
                return
        if key in ('TRACK_CONFIRM_FRAMES','INNER_ONLY_REVIEW_FRAMES','RIM_CENTER_ALIGNMENT_PX'):
            new = int(new)
        if isinstance(new, float):
            new = round(new, 3)
        globals()[key] = new
        show_ui_notice(f'{label}: {format_setting_value(key, new)}', 2)
        print(f'[SETTING] {key} {old} -> {new}', flush=True)
        return
    defaults_y = SCREEN_HEIGHT-115
    if defaults_y-16 <= y <= defaults_y+16 and 14 <= local_x <= PANEL_WIDTH-14:
        now = time.monotonic()
        if now < ui_state['defaults_confirm_until']:
            changes = []
            for key, default in DEFAULT_SETTINGS.items():
                current = globals()[key]
                if current != default:
                    globals()[key] = default
                    changes.append(f'{key}: {current} -> {default}')
            ui_state['defaults_confirm_until'] = 0.0
            ui_state['selected_setting'] = 'COUNT_LINE_FRACTION'
            print('[SETTING] Restored factory defaults' + (': ' + '; '.join(changes) if changes else ' (already at defaults)'), flush=True)
            show_ui_notice('Default settings restored', 3)
        else:
            ui_state['defaults_confirm_until'] = now + 5
            ui_state['reset_confirm_until'] = 0.0
            show_ui_notice('Click RESTORE DEFAULTS again to confirm', 4)
        return
    reset_y = SCREEN_HEIGHT-77
    if reset_y-20 <= y <= reset_y+20 and 14 <= local_x <= PANEL_WIDTH-14:
        now = time.monotonic()
        if now < ui_state['reset_confirm_until']:
            reset_request.set()
            ui_state['reset_confirm_until'] = 0.0
            show_ui_notice('Reset requested - waiting for pending AI', 5)
        else:
            ui_state['reset_confirm_until'] = now+5
            ui_state['defaults_confirm_until'] = 0.0
            show_ui_notice('Click RESET again within 5 seconds', 5)


def format_setting_value(key, value):
    if key in ('COUNT_LINE_FRACTION', 'AI_CAPTURE_FRACTION'):
        return f'{value*100:.0f}%'
    if key == 'MIN_CIRCULARITY':
        return f'{value:.2f}'
    return str(value)


def draw_tab_bar(panel):
    h = 51
    cv2.rectangle(panel, (0,0), (PANEL_WIDTH-1,h), (18,18,24), -1)
    for idx, (tab, text) in enumerate((('inspection','INSPECTION'), ('settings','SETTINGS'))):
        left = 8+idx*(PANEL_WIDTH//2)
        right = PANEL_WIDTH//2-2 if idx == 0 else PANEL_WIDTH-8
        selected = ui_state['tab'] == tab
        fill = (73,65,42) if selected else (37,37,47)
        cv2.rectangle(panel,(left,8),(right,44),fill,-1)
        cv2.rectangle(panel,(left,8),(right,44),(110,110,135) if selected else (60,60,80),1)
        scale = .49 if PANEL_WIDTH < 360 else .56
        (tw,_),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        cv2.putText(panel,text,((left+right-tw)//2,32),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,(255,255,255) if selected else (175,175,190),1,cv2.LINE_AA)
    return panel


def draw_settings_panel():
    panel = np.full((SCREEN_HEIGHT, PANEL_WIDTH,3),(18,18,24),np.uint8)
    m=15
    cv2.putText(panel,'INNER + OUTER RIM SETTINGS',(m,75),cv2.FONT_HERSHEY_SIMPLEX,
                .55,(235,235,240),1,cv2.LINE_AA)
    cv2.putText(panel,'BLUE outer / YELLOW inner | - / + to adjust',(m,91),cv2.FONT_HERSHEY_SIMPLEX,
                .36,(150,165,190),1,cv2.LINE_AA)
    top,step = settings_geometry()
    font = .34 if SCREEN_HEIGHT <= 850 else (.40 if PANEL_WIDTH < 380 else .49)
    for i,(key,label,delta,minimum,maximum) in enumerate(SETTINGS_SPEC):
        y=top+i*step
        rowhalf=max(9,min(17,step//2))
        if key == ui_state['selected_setting']:
            cv2.rectangle(panel,(8,y-rowhalf),(PANEL_WIDTH-8,y+rowhalf),(48,49,76),-1)
            cv2.rectangle(panel,(8,y-rowhalf),(PANEL_WIDTH-8,y+rowhalf),(245,200,80),1)
        elif i % 2 == 0:
            cv2.rectangle(panel,(8,y-rowhalf),(PANEL_WIDTH-8,y+rowhalf),(29,29,39),-1)
        cv2.putText(panel,label,(m,y+5),cv2.FONT_HERSHEY_SIMPLEX,font,(205,205,215),1,cv2.LINE_AA)
        val=format_setting_value(key,globals()[key])
        value_x=PANEL_WIDTH-151
        cv2.putText(panel,val,(value_x,y+5),cv2.FONT_HERSHEY_SIMPLEX,
                    .42,(110,230,255),1,cv2.LINE_AA)
        for cx,sign in ((PANEL_WIDTH-87,'-'),(PANEL_WIDTH-30,'+')):
            cv2.rectangle(panel,(cx-20,y-rowhalf),(cx+20,y+rowhalf),(59,57,73),-1)
            cv2.rectangle(panel,(cx-20,y-rowhalf),(cx+20,y+rowhalf),(99,99,120),1)
            (tw,_),_=cv2.getTextSize(sign,cv2.FONT_HERSHEY_SIMPLEX,.75,2)
            cv2.putText(panel,sign,(cx-tw//2,y+7),cv2.FONT_HERSHEY_SIMPLEX,
                        .75,(245,245,250),2,cv2.LINE_AA)
    defaults_y = SCREEN_HEIGHT-115
    defaults_confirm = time.monotonic() < ui_state['defaults_confirm_until']
    defaults_label = 'CONFIRM RESTORE DEFAULTS' if defaults_confirm else 'RESTORE DEFAULT SETTINGS'
    defaults_color = (34, 90, 150) if defaults_confirm else (55, 75, 80)
    cv2.rectangle(panel,(m,defaults_y-16),(PANEL_WIDTH-m,defaults_y+16),defaults_color,-1)
    cv2.rectangle(panel,(m,defaults_y-16),(PANEL_WIDTH-m,defaults_y+16),(115,165,175),1)
    defaults_scale = .43 if PANEL_WIDTH < 380 else .52
    dtw = cv2.getTextSize(defaults_label,cv2.FONT_HERSHEY_SIMPLEX,defaults_scale,1)[0][0]
    cv2.putText(panel,defaults_label,((PANEL_WIDTH-dtw)//2,defaults_y+5),
                cv2.FONT_HERSHEY_SIMPLEX,defaults_scale,(255,255,255),1,cv2.LINE_AA)
    reset_y=SCREEN_HEIGHT-77
    waiting=reset_request.is_set()
    confirm=time.monotonic()<ui_state['reset_confirm_until']
    label='RESET QUEUED' if waiting else ('CONFIRM RESET COUNTS' if confirm else 'RESET COUNTS')
    color=(65,65,100) if waiting else ((35,70,170) if confirm else (43,55,112))
    cv2.rectangle(panel,(m,reset_y-20),(PANEL_WIDTH-m,reset_y+20),color,-1)
    cv2.rectangle(panel,(m,reset_y-20),(PANEL_WIDTH-m,reset_y+20),(120,130,180),1)
    tw=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.53,1)[0][0]
    cv2.putText(panel,label,((PANEL_WIDTH-tw)//2,reset_y+6),cv2.FONT_HERSHEY_SIMPLEX,
                .53,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(panel,'Count direction: UP (unchanged)',(m,SCREEN_HEIGHT-34),
                cv2.FONT_HERSHEY_SIMPLEX,.42,(155,190,175),1,cv2.LINE_AA)
    cv2.putText(panel,'Q: Exit  |  Changes last until restart',(m,SCREEN_HEIGHT-16),
                cv2.FONT_HERSHEY_SIMPLEX,.36,(160,160,180),1,cv2.LINE_AA)
    return panel


def draw_hmi_panel(base, st, thumb, camera_fps, vision_fps, display_fps, active_count):
    if ui_state['tab'] == 'settings':
        panel = draw_settings_panel()
    else:
        panel = draw_panel(base,st,thumb,camera_fps,vision_fps,display_fps,active_count)
    draw_tab_bar(panel)
    if time.monotonic() < ui_state['notice_until']:
        text=ui_state['notice']
        scale=.38
        while scale > .26 and cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0] > PANEL_WIDTH-20:
            scale -= .02
        y = SCREEN_HEIGHT-31 if ui_state['tab'] == 'settings' else SCREEN_HEIGHT-47
        cv2.rectangle(panel,(7,y-18),(PANEL_WIDTH-7,y+7),(49,58,80),-1)
        cv2.putText(panel,text,(11,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
    return panel
# --------------------------------------------------------------------

# ---------------------- STREAMLIT LIVE METRICS ----------------------
# Same JSON schema and atomic replacement as the original YOLO publisher.
METRICS_DIRECTORY = Path(__file__).resolve().parent / 'shared_data'
METRICS_FILE = METRICS_DIRECTORY / 'live_metrics.json'
METRICS_TEMP = METRICS_DIRECTORY / 'live_metrics.tmp'


def publish_metrics(image_count, pass_count, fail_count):
    """Publish a complete, consistent snapshot for the Streamlit dashboard."""
    image_count = int(image_count)
    pass_count = int(pass_count)
    fail_count = int(fail_count)
    graded_count = pass_count + fail_count
    metrics = {
        'total_inspected': image_count,
        'graded_count': graded_count,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'pending_count': max(0, image_count - graded_count),
        'pass_rate': round(100.0 * pass_count / graded_count, 2) if graded_count else 0.0,
        'last_updated': datetime.now().astimezone().isoformat(),
    }
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with METRICS_TEMP.open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=4)
    os.replace(METRICS_TEMP, METRICS_FILE)


def publish_shared_metrics():
    """Read all counters under one lock and publish without holding the lock."""
    with shared_lock:
        st = dict(shared['state'])
    try:
        publish_metrics(st['graded'], st['passed'], st['failed'])
    except OSError as exc:
        print(f'[ERROR] Could not publish Streamlit metrics: {exc}', file=sys.stderr, flush=True)


# ---------------------- SESSION TEXT REPORT ----------------------
def get_desktop_path():
    """Resolve the real Windows Desktop, including OneDrive redirection."""
    if sys.platform == "win32":
        buffer = ctypes.create_unicode_buffer(32768)
        # CSIDL_DESKTOPDIRECTORY = 0x10; current configured folder = 0.
        result = ctypes.windll.shell32.SHGetFolderPathW(
            None, 0x10, None, 0, buffer
        )
        if result != 0 or not buffer.value:
            raise OSError(f"Could not locate Windows Desktop (HRESULT {result})")
        return Path(buffer.value)
    return Path.home() / "Desktop"


def save_session_log(start_time, end_time, image_count, pass_count,
                     fail_count, exit_status, error_message):
    """Write one uniquely named report for this inspection session."""
    pass_rate = pass_count / image_count * 100 if image_count else 0.0
    desktop = get_desktop_path()
    filename = f"Can_Inspection_Log_{start_time:%Y-%m-%d_%H-%M-%S_%f}.txt"
    desktop.mkdir(parents=True, exist_ok=True)
    log_path = desktop / filename
    report = (
        "YARDS BREWING - CAN INSPECTION LOG\n\n"
        "SESSION INFORMATION\n"
        "------------------------------\n"
        f"Start Time: {start_time.isoformat(sep=' ', timespec='seconds')}\n"
        f"End Time:   {end_time.isoformat(sep=' ', timespec='seconds')}\n"
        f"Duration:   {end_time - start_time}\n"
        f"Exit Status: {exit_status}\n"
        f"Error: {error_message or 'None'}\n\n"
        "PRODUCTION METRICS\n"
        "------------------------------\n"
        f"Total Cans Inspected (image_count): {image_count}\n"
        f"Passed Cans: {pass_count}\n"
        f"Failed Cans: {fail_count}\n"
        f"Pass Rate: {pass_rate:.2f}%\n"
        f"Inspected Without Final Result: {image_count - pass_count - fail_count}\n"
        + ("Note: No cans inspected; pass rate shown as 0.00%.\n" if not image_count else "")
        + "\nEND OF SESSION REPORT\n"
    )
    with log_path.open("x", encoding="utf-8") as log_file:
        log_file.write(report)
    print(f"[INFO] Session log saved to: {log_path}")
    return log_path

# ---------------------- RETRAINING IMAGE DATASET ----------------------
# Images are the actual, resized RGB uint8 inputs sent into Hailo, not
# screenshot captures or raw full-resolution frames. Stored as lossless PNG.
# Predicted classes are NOT ground-truth labels; review before retraining.
def model_image_writer(session_directory):
    """Single disk-writing thread; saving does not run inside Hailo inference."""
    session_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = session_directory / 'manifest.csv'
    saved = 0
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('job_id', 'time', 'image_path', 'predicted_class',
                         'confidence_percent', 'inference_error', 'width', 'height',
                         'capture_quality', 'center_error_px', 'target_y_px',
                         'outer_present','inner_present','rim_alignment_px'))
        while True:
            record = image_save_queue.get()
            try:
                if record is None:
                    break
                job_id, timestamp, model_rgb, predicted, confidence, error, capture_meta = record
                category = predicted if predicted in CLASS_NAMES else 'inference_error'
                output_dir = session_directory / category
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f'input_{job_id:07d}.png'
                path = output_dir / filename
                # cv2.imwrite expects BGR, but the model receives RGB.
                bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f'cv2.imwrite returned False: {path}')
                writer.writerow((job_id, timestamp, str(path.relative_to(session_directory)),
                                 predicted, f'{confidence:.4f}' if confidence is not None else '',
                                 error or '', model_rgb.shape[1], model_rgb.shape[0],
                                 capture_meta.get('quality', 'UNKNOWN'),
                                 capture_meta.get('center_error_px', ''),
                                 capture_meta.get('target_y_px', ''),
                                 capture_meta.get('outer_present',''),
                                 capture_meta.get('inner_present',''),
                                 capture_meta.get('alignment_px','')))
                f.flush()
                saved += 1
            except Exception as exc:
                print(f'[IMAGES ERROR] Could not write model image: {exc}', file=sys.stderr, flush=True)
            finally:
                image_save_queue.task_done()
    print(f'[IMAGES] Saved {saved} model input images in {session_directory}', flush=True)


# ---------------------- [DB] SQLITE LOGGING ----------------------
def db_writer():
    """Only thread that touches SQLite. Commits never block camera/vision/display."""
    try:
        import inspection_db  # opens the connection in THIS thread
        print(f'[INFO] SQLite logging to {inspection_db.DB_PATH}', flush=True)
    except Exception as exc:
        print(f'[ERROR] SQLite logging disabled: {exc}', file=sys.stderr, flush=True)
        inspection_db = None
    while True:
        job = db_queue.get()
        if job is None:  # shutdown marker, sent after the session row
            break
        if inspection_db is None:
            continue  # keep draining so queued ROI images are freed
        kind, args = job
        try:
            if kind == 'can':
                inspection_db.log_can(*args)
            else:
                inspection_db.log_session(*args)
        except Exception as exc:
            print(f'[ERROR] Could not save {kind} to database: {exc}', file=sys.stderr, flush=True)


def log_can(can_number, frame_time, verdict, defect_type, roi, circularity,
            class_name=None, confidence=None):
    """Same arguments as inspection_db.log_can in can_tester_v2.py, but queued."""
    if DB_ENABLED:
        db_queue.put(('can', (run_info['session_start'], can_number, frame_time, verdict,
                              defect_type, roi, circularity, class_name, confidence)))


def wall_time(monotonic_ts):
    """Convert a time.monotonic() capture stamp to a local datetime for the DB."""
    return datetime.fromtimestamp(time.time() - (time.monotonic() - monotonic_ts)).astimezone()
# -----------------------------------------------------------------


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


def discover_camera_modes(device='/dev/video0'):
    """List advertised selected-format camera modes (width, height, maximum FPS)."""
    try:
        proc = subprocess.run(['v4l2-ctl', '-d', device, '--list-formats-ext'],
                              capture_output=True, text=True, timeout=4, check=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []
    modes = {}
    is_selected_format = False
    size = None
    for line in proc.stdout.splitlines():
        if re.search(r"^\s*\[\d+\]:\s+'", line):
            is_selected_format = f"'{CAMERA_FORMAT}'" in line
            size = None
            continue
        match_size = re.search(r'Size:\s+Discrete\s+(\d+)x(\d+)', line)
        if match_size:
            size = tuple(map(int, match_size.groups())) if is_selected_format else None
            continue
        match_fps = re.search(r'Interval:\s+Discrete.*?\((\d+(?:\.\d+)?)\s+fps\)', line)
        if size and match_fps:
            modes[size] = max(modes.get(size, 0.), float(match_fps.group(1)))
    return [(w, h, fps) for (w, h), fps in modes.items()]


def select_camera_mode():
    """Always keep the requested camera resolution; report selected-format support."""
    modes = discover_camera_modes(f'/dev/video{CAMERA_INDEX}')
    if '--list-camera-modes' in sys.argv:
        if modes:
            print(f'[CAMERA] Advertised {CAMERA_FORMAT} modes:', flush=True)
            for w, h, fps in sorted(modes, key=lambda m: (m[0]*m[1], m[2]), reverse=True):
                print(f'  {w}x{h} @ up to {fps:g} FPS', flush=True)
        else:
            print(f'[CAMERA] Could not enumerate {CAMERA_FORMAT} modes (v4l-utils required).', flush=True)
    if modes:
        match = next((max_fps for w, h, max_fps in modes
                      if (w, h) == (CAMERA_WIDTH, CAMERA_HEIGHT)), None)
        if match is None:
            print(f'[WARNING] {CAMERA_WIDTH}x{CAMERA_HEIGHT} not advertised for {CAMERA_FORMAT}.', flush=True)
        elif CAMERA_FPS > match:
            print(f'[WARNING] Requested {CAMERA_FPS:g} FPS but {CAMERA_FORMAT} '
                  f'{CAMERA_WIDTH}x{CAMERA_HEIGHT} advertises at most {match:g} FPS.', flush=True)
    return CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


def open_configured_camera():
    width, height, fps = select_camera_mode()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera /dev/video{CAMERA_INDEX}')
    try:
        # Use the tested 1080p MJPEG camera mode. Buffer=4 avoids the observed FPS halving.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FORMAT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.set(cv2.CAP_PROP_BUFFERSIZE, 4):
            print('[WARNING] Could not set V4L2 camera buffers to 4.', flush=True)
        if CAMERA_EXPOSURE_US is not None:
            # Many USB UVC cameras use exposure_absolute in 100-microsecond ticks.
            # Some devices don't support it; report errors instead of assuming success.
            controls = ['v4l2-ctl', '-d', f'/dev/video{CAMERA_INDEX}',
                        '-c', 'exposure_auto=1', '-c', f'exposure_absolute={max(1, round(CAMERA_EXPOSURE_US / 100))}']
            try:
                response = subprocess.run(controls, capture_output=True, text=True, timeout=3)
                if response.returncode != 0:
                    print(f'[WARNING] Exposure control not applied: {response.stderr.strip()}', flush=True)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print('[WARNING] Install v4l-utils to set --exposure-us.', flush=True)
        aw = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join(chr((fourcc_int >> (8*i)) & 255) for i in range(4))
        print(f'[CAMERA] Requested {width}x{height}@{fps:g}; driver reports '
              f'{aw}x{ah}@{afps:.1f} FPS, pixel format {fourcc}', flush=True)
        if (aw, ah) != (width, height):
            raise RuntimeError(f'Camera delivered {aw}x{ah} instead of requested {width}x{height}; refusing to alter trained ROI framing.')
        if fourcc != CAMERA_FORMAT:
            raise RuntimeError(f'Camera delivered {fourcc} instead of {CAMERA_FORMAT}; MJPEG configuration failed.')
        if afps < fps - 2:
            print(f'[WARNING] Driver reports {afps:.1f} FPS, below requested {fps:g} FPS.', flush=True)
        return cap
    except BaseException:
        cap.release()
        raise


def run_camera_test(seconds=6.0):
    """Measure actual cap.read() throughput without Hailo, dashboard or Canny."""
    cap = open_configured_camera()
    try:
        count = 0
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            ok, frame = cap.read()
            if ok and frame is not None:
                count += 1
        elapsed = time.monotonic() - start
        print(f'[CAMERA TEST] Delivered {count} frames in {elapsed:.2f}s: '
              f'{count/elapsed:.1f} FPS (independent of display/vision).', flush=True)
    finally:
        cap.release()


def camera_worker(cap):
    """Never block camera capture on vision processing.

    When vision is slower than capture, remove the OLDEST queued frame. This
    avoids multi-second latency but means overload can miss fast cans. Counters
    and warnings make that tradeoff visible; real lossless operation requires
    vision FPS >= delivered camera FPS (or a sensor trigger).
    """
    count, start = 0, time.monotonic()
    last_overflow_warning = start
    while not stop_event.is_set():
        ok, frame = cap.read()
        now = time.monotonic()
        if ok and frame is not None:
            count += 1
            with shared_lock:
                shared['frame'] = frame
                shared['frame_at'] = now
            try:
                frame_queue.put_nowait((now, frame))
            except Full:
                try:
                    frame_queue.get_nowait()  # drop stale frame, never block capture
                except Empty:
                    pass
                try:
                    frame_queue.put_nowait((now, frame))
                except Full:
                    pass
                with frame_queue_stats_lock:
                    frame_queue_stats['overflow'] += 1
                if now-last_overflow_warning >= 2.0:
                    print('[WARNING] Vision slower than camera: oldest queued frames dropped. '
                          'Missed cans possible; reduce requested FPS or optimize vision.',flush=True)
                    last_overflow_warning = now
            with frame_queue_stats_lock:
                frame_queue_stats['high_watermark'] = max(
                    frame_queue_stats['high_watermark'], frame_queue.qsize())
        else:
            time.sleep(.005)
        if now-start >= 1.0:
            with shared_lock:
                shared['capture_fps'] = count/(now-start)
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
                        job_id, crop, capture_meta = inference_queue.get(timeout=.1)
                    except Empty:
                        continue
                    start = time.monotonic()
                    img = None
                    try:
                        img = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        tensor = np.ascontiguousarray(img[None, ...], dtype=np.uint8)
                        result = pipeline.infer({input_info.name: tensor})[output_info.name]
                        probabilities = get_probabilities(result)
                        if SAVE_MODEL_IMAGES:
                            label_idx = int(np.argmax(probabilities))
                            image_save_queue.put((job_id, datetime.now().astimezone().isoformat(),
                                                  img, CLASS_NAMES[label_idx],
                                                  float(probabilities[label_idx]) * 100, None, capture_meta))
                        result_queue.put((job_id, probabilities, (time.monotonic()-start)*1000, None))
                    except Exception as exc:
                        if SAVE_MODEL_IMAGES and img is not None:
                            # Save even if inference fails: this input was prepared for Hailo.
                            image_save_queue.put((job_id, datetime.now().astimezone().isoformat(),
                                                  img, 'inference_error', None, str(exc), capture_meta))
                        result_queue.put((job_id, None, (time.monotonic()-start)*1000, str(exc)))
                    finally:
                        inference_queue.task_done()
    except Exception as exc:
        run_info['fatal_error'] = f'Hailo worker: {exc}'  # [DB] for the sessions row
        result_queue.put((-1, None, 0., str(exc)))
        stop_event.set()


# ---------------------- DUAL-RIM DETECTION ----------------------
def detect_dual_rims(contours, scale):
    """Return one detection per can, pairing independently detected rims.

    Find contours at all hierarchy levels; RETR_EXTERNAL hides the inner edge.
    A pair is permitted only when centers are close and the outer is larger.
    Neither rim is required to be perfectly circular. A single inner rim can
    continue tracking/counting, but is never a qualified centered photo.
    """
    inner, outer = [], []
    for c in contours:
        area_small = cv2.contourArea(c)
        area = area_small * scale * scale
        (cx, cy), radius_small = cv2.minEnclosingCircle(c)
        r = radius_small * scale
        if radius_small < 1:
            continue
        fill = area_small / max(np.pi*radius_small*radius_small, 1.)
        # Moment centers can shift on deformed contours, so use the bounding
        # extent's center for both classes consistently.
        bx, by, bw, bh = cv2.boundingRect(c)
        center_x = (bx + bw * 0.5) * scale
        center_y = (by + bh * 0.5) * scale
        item = dict(cx=center_x, cy=center_y, r=r, area=area,
                    contour=c, sf=scale, fill=fill)
        if (OUTER_MIN_AREA <= area <= OUTER_MAX_AREA and
                OUTER_MIN_RADIUS <= r <= OUTER_MAX_RADIUS and fill >= .20):
            outer.append(item)
        elif (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA and
              MIN_CAN_RADIUS <= r <= MAX_CAN_RADIUS and fill >= MIN_CONTOUR_FILL):
            inner.append(item)
    # Prefer largest outer contour, then pair with nearest inner rim.
    outer.sort(key=lambda o: -o['area'])
    inner.sort(key=lambda i: -i['area'])
    used_inner=set()
    found=[]
    for o in outer:
        choices=[]
        for j,i in enumerate(inner):
            if j in used_inner or o['r'] <= i['r']*1.35:
                continue
            delta=float(np.hypot(o['cx']-i['cx'],o['cy']-i['cy']))
            if delta <= RIM_CENTER_ALIGNMENT_PX:
                choices.append((delta,j))
        if choices:
            delta,j=min(choices)
            i=inner[j]
            used_inner.add(j)
            # Use mean Y only when independently detected centers agree.
            o=dict(o, has_outer=True, has_inner=True,
                   aligned=True, alignment_px=delta,
                   inner_x=i['cx'], inner_y=i['cy'], inner_r=i['r'])
        else:
            o=dict(o, has_outer=True, has_inner=False,
                   aligned=False, alignment_px=None)
        # A contour that is merely a duplicate nested outer edge should not
        # create a second physical can track.
        if any(np.hypot(o['cx']-d['cx'],o['cy']-d['cy']) < 35 for d in found):
            continue
        found.append(o)
    for j,i in enumerate(inner):
        if j in used_inner:
            continue
        if any(np.hypot(i['cx']-o['cx'],i['cy']-o['cy']) < max(45,o['r']*.55)
               for o in found):
            continue
        found.append(dict(i, has_outer=False, has_inner=True,
                          aligned=False, alignment_px=None,
                          inner_x=i['cx'],inner_y=i['cy'],inner_r=i['r']))
    return found, len(inner), len(outer)


# ---------------------- KALMAN MULTI-CAN TRACKER ----------------------
# Lightweight Canny-centroid matching (no Kalman and no SciPy required).
# One-to-one greedy matching, predicted only a short distance along observed
# velocity. Raw measured centroids drive counting/capture; only overlay is smoothed.
def associate_cans(tracks, detections, timestamp, max_distance):
    ids = tuple(tracks)
    if not ids or not detections:
        return [], set(range(len(detections))), set(ids)
    candidates = []
    for tid in ids:
        tr = tracks[tid]
        age = max(0., timestamp-tr['last_seen_at'])
        if age > MAX_TRACK_ASSOCIATION_AGE:
            continue
        dt = min(age, .12)
        px = tr['cx'] + tr.get('vx',0.) * dt
        py = tr['cy'] + tr.get('vy',0.) * dt
        for j,det in enumerate(detections):
            cx,cy,radius,area = det['cx'],det['cy'],det['r'],det['area']
            radius_ratio = radius/max(tr['r'],1.)
            area_ratio = area/max(tr['area'],1.)
            changing_rim = det['has_outer'] != tr.get('has_outer', False)
            if not changing_rim and not (.58 <= radius_ratio <= 1.75 and .30 <= area_ratio <= 3.2):
                continue
            distance = float(np.hypot(cx-px,cy-py))
            gate = min(max_distance, max(75., tr['r']*.92))
            if distance > gate:
                continue
            cost = distance + (0 if changing_rim else 18.*abs(np.log(radius_ratio)) + 5.*abs(np.log(area_ratio)))
            # Prefer recently seen tracks over stale ones when equally close.
            candidates.append((cost+age*35, tid, j))
    candidates.sort()
    used_ids, used_j = set(), set()
    pairs = []
    for _cost,tid,j in candidates:
        if tid not in used_ids and j not in used_j:
            used_ids.add(tid)
            used_j.add(j)
            pairs.append((tid,j))
    return pairs, set(range(len(detections)))-used_j, set(ids)-used_ids


def select_capture_candidate(track, raw_roi, target_y, roi_side, radius):
    """Only a paired/aligned inner+outer rim qualifies as centered.

    The image is NEVER cropped or translated. Keep best observed full 900x900
    camera frame as a fallback; flag missing outer/poor alignment in metadata.
    """
    distance = abs(track['cy'] - target_y)
    margin = min(max(0.0, radius), roi_side / 2)
    visible = (margin <= track['cy'] <= roi_side-margin and
               margin <= track['cx'] <= roi_side-margin)
    has_outer=bool(track.get('has_outer'))
    has_inner=bool(track.get('has_inner'))
    aligned=bool(track.get('aligned'))
    qualified = visible and has_outer and has_inner and aligned and distance <= CENTER_CAPTURE_TOLERANCE
    # Prefer verified frames over any inner-rim-only fallback.
    old_dist = track.get('best_capture_distance', float('inf'))
    old_ok = track.get('best_capture_centered', False)
    if ((qualified and not old_ok) or (qualified == old_ok and distance < old_dist)):
        track['best_capture_roi'] = raw_roi.copy()
        track['best_capture_distance'] = distance
        track['best_capture_y'] = track['cy']
        track['best_capture_centered'] = qualified
        track['best_capture_has_outer'] = has_outer
        track['best_capture_has_inner'] = has_inner
        track['best_capture_alignment_px'] = track.get('alignment_px')
    return qualified


def capture_metadata(track, target_y):
    distance = float(track.get('best_capture_distance', float('inf')))
    centered = bool(track.get('best_capture_centered', False))
    has_outer = bool(track.get('best_capture_has_outer',False))
    has_inner = bool(track.get('best_capture_has_inner',False))
    quality = ('CENTERED_BOTH_RIMS' if centered else
               'INNER_ONLY_OUTER_SUSPECT' if has_inner and not has_outer else
               'RIMS_NOT_ALIGNED' if has_outer and has_inner else
               'OFF_CENTER_FALLBACK')
    return dict(quality=quality, center_error_px=round(distance, 2),
                target_y_px=round(float(target_y), 2),
                outer_present=has_outer, inner_present=has_inner,
                alignment_px=track.get('best_capture_alignment_px'))


def vision_worker():
    """Owns tracks and counters; never calls imshow/waitKey."""
    state = dict(shared['state'])
    tracks, pending = {}, {}
    deferred_captures = {}  # Counted cans awaiting their independent AI capture line.
    next_track_id = next_job_id = 0
    can_count = 0  # Lifetime unique can number for SQLite; never reset.
    count_offset = 0  # Display/Streamlit counts since most recent reset.
    last_thumb, hold_until = None, 0.
    last_frame, last_roi_box, last_circles, last_rim_marks = None, None, (), ()
    vision_count, vision_window = 0, time.monotonic()
    vision_fps = 0.
    diag_last = time.monotonic()
    diag_rejected_area = diag_rejected_radius = diag_rejected_fill = 0
    diag_detected = 0
    def submit_ai_image(image, can_number, frame_time, circularity, capture_meta=None):
        """Send a counted can to Hailo with the selected capture image."""
        nonlocal next_job_id, last_thumb, hold_until
        last_thumb = image
        hold_until = time.monotonic() + HOLD_SECONDS
        next_job_id += 1
        try:
            inference_queue.put_nowait((next_job_id, image, capture_meta or {}))
            pending[next_job_id] = dict(can=can_number, frame_time=frame_time,
                                        roi=image, circ=circularity)
            state['pending'] = len(pending)
            state.update(status='AI INSPECTION PENDING', color=(140, 140, 160),
                         prediction='PROCESSING...', confidence=0.)
        except Full:
            state['skipped'] += 1
            state['total'] += 1
            state['failed'] += 1
            state.update(status='FAIL - AI QUEUE FULL', prediction='NOT INSPECTED',
                         color=(0, 60, 255), confidence=0.)
            log_can(can_number, frame_time, 'FAIL', 'NOT INSPECTED', image, circularity)

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
            job = pending.pop(job_id)  # [DB] can number, frame time, ROI, circularity
            state['pending'] = len(pending)
            state['latency'] = latency if state['latency'] == 0 else .8*state['latency'] + .2*latency
            if error:
                state.update(status='FAIL - AI ERROR', prediction='AI ERROR',
                             color=(0, 60, 255), confidence=0.)
                state['total'] += 1
                state['failed'] += 1
                print(f'[ERROR] Inspection {job_id}: {error}', flush=True)
                # [DB] save this can (Hailo failed, so no class or confidence)
                log_can(job['can'], job['frame_time'], 'FAIL', 'AI ERROR', job['roi'], job['circ'])
            else:
                idx = int(np.argmax(probs))
                state['confidence'] = float(probs[idx])*100
                state['prediction'] = CLASS_NAMES[idx].upper()
                if GOOD_CLASS_INDEX is None:
                    state.update(status='REVIEW - CLASS UNMAPPED', color=(140, 140, 160))
                    # [DB] no verdict: shows up as no_result_count in the sessions row
                elif idx == GOOD_CLASS_INDEX:
                    state['total'] += 1
                    state['passed'] += 1
                    state.update(status='PASS - GOOD CAN', color=(0, 255, 120))
                    # [DB] save this can
                    log_can(job['can'], job['frame_time'], 'PASS', None, job['roi'], job['circ'],
                            CLASS_NAMES[idx], state['confidence'])
                else:
                    state['total'] += 1
                    state['failed'] += 1
                    state.update(status='FAIL - BAD CLASS', color=(0, 60, 255))
                    # [DB] save this can
                    log_can(job['can'], job['frame_time'], 'FAIL', 'BAD CLASS', job['roi'], job['circ'],
                            CLASS_NAMES[idx], state['confidence'])
                    # Will's code here: the absolute can number is job['can']
            hold_until = now + HOLD_SECONDS

        # Reset only once outstanding AI results have been processed; no historical
        # database entries are discarded and already graded tracks stay graded.
        if reset_request.is_set() and not pending and not deferred_captures and inference_queue.empty() and result_queue.empty():
            run_info['prior_resets']['graded'] += state['graded']
            run_info['prior_resets']['passed'] += state['passed']
            run_info['prior_resets']['failed'] += state['failed']
            count_offset = can_count
            state.update(graded=0,total=0,passed=0,failed=0,pending=0,skipped=0,
                         status='COUNTS RESET',prediction='SCANNING...',confidence=0.,
                         color=(140,140,160))
            reset_request.clear()
            print('[RESET] Display/Streamlit counters reset; SQLite can IDs retained', flush=True)
            with shared_lock:
                shared['state'] = dict(state)
            publish_shared_metrics()

        try:
            captured_at, frame = frame_queue.get(timeout=.02)
            # IMPORTANT: process FIFO frames individually. Draining to latest
            # skips the exact frames where high-speed cans cross the line.
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
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)  # Smaller blur; faster, less contour drift
            edges = cv2.Canny(gray, 50, 150)
            # RETR_TREE: both concentric rims must be visible to inspection.
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found, inner_hits, outer_hits = detect_dual_rims(contours, scale)
            diag_detected += len(found)

            # Optional once-per-second diagnostics: run with --debug-counting.
            # "accepted" refers to contours, not cans; high rejection numbers
            # help identify whether the problem is detection or tracking.
            if '--debug-counting' in sys.argv and now - diag_last >= 1.0:
                print(f'[VISION] contours accepted={diag_detected} '
                      f'inner_candidates={inner_hits} outer_candidates={outer_hits} '
                      f'rejected_radius={diag_rejected_radius} '
                      f'rejected_shape={diag_rejected_fill} '
                      f'tracks={len(tracks)} cans={can_count} '
                      f'queue={frame_queue.qsize()}/{VISION_QUEUE_SIZE} '
                      f'overflow={frame_queue_stats["overflow"]}', flush=True)
                diag_detected = diag_rejected_area = diag_rejected_radius = diag_rejected_fill = 0
                diag_last = now

            # Lightweight Canny centroid matching with one-to-one assignment. Tracks
            # remain alive when contours flicker or the conveyor is stopped.
            pairs, unassigned, missing_ids = associate_cans(
                tracks, found, captured_at, TRACK_MATCH_DISTANCE)
            for tid, index in pairs:
                det=found[index]
                cx,cy,radius,contour,area,sf = (det['cx'],det['cy'],det['r'],det['contour'],det['area'],det['sf'])
                tr = tracks[tid]
                dt = max(0.001, captured_at - tr['last_seen_at'])
                old_x, old_y = tr['cx'], tr['cy']
                # Use raw measured positions for
                # center selection and line crossings (no filter latency).
                # No Kalman update: raw Canny centroid controls crossing and capture.
                tr['prev_cy'] = old_y
                tr['vx'] = .65*tr['vx'] + .35*((cx-old_x)/dt)
                tr['vy'] = .65*tr['vy'] + .35*((cy-old_y)/dt)
                # Adaptive low-lag overlay: faster motion -> less smoothing.
                step = float(np.hypot(cx-old_x, cy-old_y))
                alpha = 1.0 if step >= 40 else max(TRACK_SMOOTHING, .70 + .30*(step/40.))
                display_x = tr['display_cx'] + alpha*(cx-tr['display_cx'])
                display_y = tr['display_cy'] + alpha*(cy-tr['display_cy'])
                display_r = tr['display_r'] + .75*(radius-tr['display_r'])
                tr.update(cx=cx, cy=cy, r=radius, contour=contour,
                          area=area, scale=sf, missing=0,
                          last_seen_at=captured_at, seen=tr['seen']+1,
                          display_cx=display_x, display_cy=display_y,
                          display_r=display_r, has_outer=det['has_outer'],
                          has_inner=det['has_inner'], aligned=det['aligned'],
                          alignment_px=det['alignment_px'],
                          outer_seen=tr.get('outer_seen',0)+int(det['has_outer']),
                          inner_only_seen=tr.get('inner_only_seen',0)+int(det['has_inner'] and not det['has_outer']))

            for tid in missing_ids:
                tr = tracks[tid]
                tr['missing'] += 1
                if captured_at - tr['last_seen_at'] > TRACK_MISSING_SECONDS:
                    if tid in deferred_captures:
                        job = deferred_captures.pop(tid)
                        chosen = job['selection']
                        meta = capture_metadata(chosen, ROI_SIZE * AI_CAPTURE_FRACTION)
                        submit_ai_image(chosen['best_capture_roi'], job['can'], job['frame_time'], job['circ'], meta)
                        print(f'[CAPTURE] Can #{job["can"]}: track lost, {meta["quality"]}, error={meta["center_error_px"]:.1f}px', flush=True)
                    if '--debug-counting' in sys.argv and not tr['graded']:
                        print(f'[TRACK LOST] id={tid} seen={tr["seen"]} '
                              f'first_y={tr["first_cy"]:.1f} last_y={tr["cy"]:.1f} '
                              f'crossing_pending={tr.get("crossing_pending", False)}', flush=True)
                    del tracks[tid]

            for index in sorted(unassigned):
                det=found[index]
                cx,cy,radius,contour,area,sf = (det['cx'],det['cy'],det['r'],det['contour'],det['area'],det['sf'])
                # Suppress alternate contours of *the same visible can*, not
                # nearby cans. Existing tracks (even temporarily missing ones)
                # remain the primary protection against duplicate counting.
                duplicate_edge = any(
                    tr['missing'] <= 2 and
                    float(np.hypot(cx-tr['cx'], cy-tr['cy'])) <
                        max(12., min(radius, tr['r']) * .16) and
                    (.25 <= radius / max(tr['r'], 1) <= 4.0)
                    for tr in tracks.values()
                )
                if duplicate_edge:
                    continue
                tracks[next_track_id] = dict(cx=cx, cy=cy, r=radius,
                                             contour=contour, area=area,
                                             scale=sf, missing=0, graded=False,
                                             seen=1, first_cy=cy,
                                             prev_cy=cy, vx=0., vy=0.,
                                             last_seen_at=captured_at,
                                             display_cx=cx, display_cy=cy,
                                             display_r=radius, crossing_pending=False,
                                             has_outer=det['has_outer'],has_inner=det['has_inner'],
                                             aligned=det['aligned'],alignment_px=det['alignment_px'],
                                             outer_seen=int(det['has_outer']),
                                             inner_only_seen=int(det['has_inner'] and not det['has_outer']))
                next_track_id += 1

            line_y = side * COUNT_LINE_FRACTION
            capture_y = side * AI_CAPTURE_FRACTION
            # Count and image capture are independent. Retain an unchanged
            # full ROI ONLY from a real camera frame. Centered images require
            # actual can detection within +/-30 px of the desired Y and the
            # estimated rim fully inside the ROI. Unqualified images are kept
            # as fallback only and flagged in the training manifest.
            for tid, track in tracks.items():
                if track['missing']:
                    continue
                if not track['graded']:
                    # On Settings changes, do not reuse images measured against
                    # a different capture target.
                    if track.get('capture_fraction') != AI_CAPTURE_FRACTION:
                        for key in ('best_capture_roi', 'best_capture_distance',
                                    'best_capture_y', 'best_capture_centered'):
                            track.pop(key, None)
                        track['capture_fraction'] = AI_CAPTURE_FRACTION
                    select_capture_candidate(track, raw_roi, capture_y, side, track['r'])
                elif tid in deferred_captures:
                    job = deferred_captures[tid]
                    # Use same selection policy after counting if the operator
                    # moves capture position past the counting line.
                    job['selection']['cx'] = track['cx']
                    job['selection']['cy'] = track['cy']
                    select_capture_candidate(job['selection'], raw_roi,
                                             capture_y, side, track['r'])
                    previous_y = track['prev_cy']
                    crossed_capture = (COUNT_DIRECTION * (previous_y-capture_y) < 0 <=
                                       COUNT_DIRECTION * (track['cy']-capture_y))
                    if crossed_capture or abs(track['cy'] - capture_y) <= CENTER_CAPTURE_TOLERANCE:
                        deferred_captures.pop(tid)
                        chosen = job['selection']
                        meta = capture_metadata(chosen, capture_y)
                        print(f'[CAPTURE] Can #{job["can"]}: {meta["quality"]} '
                              f'error={meta["center_error_px"]:.1f}px', flush=True)
                        submit_ai_image(chosen['best_capture_roi'], job['can'],
                                        job['frame_time'], job['circ'], meta)
            # Remember a crossing even when it happened on a track's FIRST
            # detection or BEFORE the track reached TRACK_CONFIRM_FRAMES.
            # A fast can may cross between frame 1 and frame 2: gating first
            # and checking only the current frame permanently loses it.
            # Require forward motion to avoid counting a stationary can.
            for tid, track in list(tracks.items()):
                if track['missing'] or track['graded']:
                    continue
                previous_y, current_y = track['prev_cy'], track['cy']
                first_y = track['first_cy']
                signed_prev = COUNT_DIRECTION * (previous_y - line_y)
                signed_now = COUNT_DIRECTION * (current_y - line_y)
                signed_first = COUNT_DIRECTION * (first_y - line_y)
                signed_travel = COUNT_DIRECTION * (current_y - first_y)
                step_forward = COUNT_DIRECTION * (current_y - previous_y)
                crossed_line = signed_prev < 0 <= signed_now and step_forward > 0
                # Newly acquired tracks may start just past the line because
                # the rim was previously obscured. Only permit this within a
                # narrow entry band, after actual forward movement is seen.
                started_near_line = (abs(signed_first) <= COUNT_ENTRY_BAND_PIXELS and
                                     signed_now >= 0 and
                                     signed_travel >= COUNT_INITIAL_MOTION_PIXELS)
                if crossed_line or started_near_line:
                    track['crossing_pending'] = True
                if track['seen'] < TRACK_CONFIRM_FRAMES or not track.get('crossing_pending', False):
                    continue
                track['graded'] = True
                track['crossing_pending'] = False
                area = track['area']
                perimeter = cv2.arcLength(track['contour'], True)
                # Circularity is invariant under uniform resizing.
                area_small = cv2.contourArea(track['contour'])
                circ = 4*np.pi*area_small/(perimeter**2) if perimeter > 0 else 0.
                # Inner-only is a review flag, not conclusive proof of a defect.
                suspected_outer = (track.get('inner_only_seen',0) >= INNER_ONLY_REVIEW_FRAMES
                                   and track.get('outer_seen',0) == 0)
                # Do not run the outer-rim circularity threshold on an inner rim.
                outer_circ_valid = bool(track.get('has_outer'))
                state.update(area=area, circularity=circ if outer_circ_valid else 0.)
                # Best available crop is independent of the count position.
                capture_image = track.get('best_capture_roi', raw_roi).copy()
                capture_meta = capture_metadata(track, capture_y)
                last_thumb = capture_image
                hold_until = now+HOLD_SECONDS
                if outer_circ_valid and circ < MIN_CIRCULARITY and not RIM_FAIL_ENABLED:
                    track['graded'] = True  # Do not keep reconsidering this rim
                    continue  # Explicitly configured to ignore this kind of rim
                # [DB] this can gets a number and a capture time, like image_count/frame_time in v2
                can_count += 1
                print(f'[COUNT] Can #{can_count} track={tid} triggered, outer={track.get("has_outer")} inner={track.get("has_inner")} circularity={circ:.3f} center_error={capture_meta["center_error_px"]:.1f}px quality={capture_meta["quality"]}',flush=True)
                state['graded'] = can_count - count_offset
                frame_time = wall_time(captured_at)
                if outer_circ_valid and circ < MIN_CIRCULARITY:
                    state['total'] += 1
                    state['failed'] += 1
                    state.update(status='FAIL - DEFECTIVE RIM',
                                 color=(0, 60, 255), prediction='DEFECTIVE RIM', confidence=0.)
                    # [DB] save this can (Hailo was skipped, so no class or confidence)
                    log_can(can_count, frame_time, 'FAIL', 'DEFECTIVE RIM', last_thumb, circ)
                    # Will's code here: the absolute can number is can_count
                    continue
                else:
                    if suspected_outer:
                        print(f'[RIM REVIEW] Can #{can_count}: inner rim detected repeatedly, '
                              'outer rim not detected; suspected outer-rim defect, not confirmed',flush=True)
                    # If capture position is BEFORE counting in conveyor travel,
                    # the best-centered crop is already cached. Otherwise wait
                    # for the same counted track to reach the capture position.
                    capture_after_count = COUNT_DIRECTION * (capture_y-line_y) > 0
                    if capture_after_count and abs(track['cy']-capture_y) > max(8, side*.012):
                        selection = dict(cx=track['cx'], cy=track['cy'],
                                         best_capture_roi=capture_image,
                                         best_capture_distance=capture_meta['center_error_px'],
                                         best_capture_centered=capture_meta['quality'] == 'CENTERED_BOTH_RIMS',
                                         best_capture_has_outer=capture_meta.get('outer_present',False),
                                         best_capture_has_inner=capture_meta.get('inner_present',False),
                                         best_capture_alignment_px=capture_meta.get('alignment_px'))
                        deferred_captures[tid] = dict(can=can_count, frame_time=frame_time,
                            circ=circ, selection=selection, counted_at=now)
                        state.update(status='AWAITING AI CAPTURE',
                                     color=(140,140,160), prediction='WAITING FOR CAPTURE...',
                                     confidence=0.)
                        print(f'[CAPTURE] Can #{can_count}: waiting for {AI_CAPTURE_FRACTION*100:.0f}% capture position',flush=True)
                    else:
                        print(f'[CAPTURE] Can #{can_count}: {capture_meta["quality"]} '
                              f'error={capture_meta["center_error_px"]:.1f}px', flush=True)
                        submit_ai_image(capture_image, can_count, frame_time, circ, capture_meta)
            last_frame = frame  # Reference only; camera creates new frames.
            last_roi_box = (roi_x, roi_y, side)
            last_rim_marks = tuple((det['cx'],det['cy'],det['r'], 'outer' if det['has_outer'] else 'inner')
                                   for det in found) + tuple((det['inner_x'],det['inner_y'],det['inner_r'],'inner')
                                   for det in found if det['has_outer'] and det['has_inner'])
            last_circles = tuple((tr['display_cx'], tr['display_cy'], tr['display_r'])
                                 for tr in tracks.values()
                                 if tr['missing'] <= 2 and tr['seen'] >= TRACK_CONFIRM_FRAMES)

        # If the conveyor stops after counting but before the AI capture line,
        # finalize with the best observed image after a bounded wait.
        for tid, job in list(deferred_captures.items()):
            if now - job['counted_at'] >= 5.0:
                deferred_captures.pop(tid)
                chosen = job['selection']
                meta = capture_metadata(chosen, ROI_SIZE * AI_CAPTURE_FRACTION)
                submit_ai_image(chosen['best_capture_roi'], job['can'], job['frame_time'], job['circ'], meta)
                print(f'[CAPTURE] Can #{job["can"]}: timeout, {meta["quality"]}, error={meta["center_error_px"]:.1f}px', flush=True)
        if now > hold_until and not tracks and not pending and not deferred_captures:
            state.update(status='Awaiting Can...', prediction='SCANNING...',
                         color=(140, 140, 160), confidence=0.)
            last_thumb = None

        # Snapshot at most once per vision iteration, without large frame copying.
        with shared_lock:
            # Never overwrite the live camera frame from the slower vision worker.
            shared.update(roi_box=last_roi_box, circles=last_circles, rim_marks=last_rim_marks,
                          thumb=last_thumb, track_count=sum(t['missing']==0 for t in tracks.values()),
                          state=dict(state), vision_fps=vision_fps)
            if frame is not None:
                shared['vision_frame_at'] = captured_at


def detect_screen_size():
    """Read Linux desktop resolution; --screen WIDTHxHEIGHT overrides detection."""
    for i, arg in enumerate(sys.argv):
        if arg == '--screen' and i + 1 < len(sys.argv):
            match = re.fullmatch(r'(\d{3,5})[xX](\d{3,5})', sys.argv[i + 1])
            if not match:
                raise ValueError('Use --screen 1024x600 or --screen 1920x1080')
            return tuple(map(int, match.groups()))
        if arg.startswith('--screen='):
            match = re.fullmatch(r'(\d{3,5})[xX](\d{3,5})', arg.split('=', 1)[1])
            if not match:
                raise ValueError('Use --screen=1024x600 or --screen=1920x1080')
            return tuple(map(int, match.groups()))
    try:
        output = subprocess.check_output(['xrandr', '--current'],
                                         text=True, stderr=subprocess.DEVNULL, timeout=2)
        # Prefer the active mode of the primary output (not the virtual desktop size).
        blocks = re.split(r'(?=^\S+ connected\b)', output, flags=re.MULTILINE)
        connected = [b for b in blocks if re.match(r'^\S+ connected', b)]
        connected.sort(key=lambda b: ' primary ' not in b.splitlines()[0])
        for block in connected:
            head = block.splitlines()[0]
            geometry = re.search(r'\b(\d{3,5})x(\d{3,5})\+\d+\+\d+', head)
            if geometry:
                return int(geometry[1]), int(geometry[2])
            active = re.search(r'^\s*(\d{3,5})x(\d{3,5})\s+.*\*', block, re.MULTILINE)
            if active:
                return int(active[1]), int(active[2])
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        result = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return result
    except Exception:
        return 1024, 600


def configure_display():
    global SCREEN_WIDTH, SCREEN_HEIGHT, PANEL_WIDTH
    SCREEN_WIDTH, SCREEN_HEIGHT = detect_screen_size()
    if SCREEN_WIDTH < 640 or SCREEN_HEIGHT < 400:
        raise ValueError(f'Screen {SCREEN_WIDTH}x{SCREEN_HEIGHT} is too small')
    PANEL_WIDTH = min(max(290, round(SCREEN_WIDTH * .32)), 470)
    PANEL_WIDTH = min(PANEL_WIDTH, SCREEN_WIDTH - 320)
    print(f'[INFO] Display {SCREEN_WIDTH}x{SCREEN_HEIGHT}; '
          f'{"compact" if SCREEN_HEIGHT <= 850 else "expanded"} dashboard; '
          f'panel {PANEL_WIDTH}px', flush=True)


def compact_label(img, text, x, y, scale=.48, color=(175, 180, 200), thickness=1):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def compact_value(img, text, x, y, max_width, scale=.65, color=(240, 240, 250)):
    text = str(text)
    while scale > .3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
        scale -= .025
    compact_label(img, text, x, y, scale, color, 2 if scale >= .55 else 1)


def make_compact_background():
    p = np.full((SCREEN_HEIGHT, PANEL_WIDTH, 3), (18, 18, 24), np.uint8)
    cv2.rectangle(p, (5, 5), (PANEL_WIDTH-5, SCREEN_HEIGHT-5), (65, 65, 85), 1)
    m, w = 14, PANEL_WIDTH-28
    compact_label(p, 'QUALITY INSPECTION', m, 30, .68, (255,255,255), 2)
    cv2.line(p, (m, 42), (PANEL_WIDTH-m, 42), (75,75,95), 1)
    compact_label(p, 'SYSTEM STATUS', m, 62)
    compact_label(p, 'PRODUCTION', m, 158)
    third = (w-12)//3
    for i, label in enumerate(('TOTAL', 'PASSED', 'FAILED')):
        x = m+i*(third+6)
        cv2.rectangle(p, (x, 168), (x+third, 225), (42,42,58), -1)
        compact_label(p, label, x+6, 183, .36)
    compact_label(p, 'PASS RATE', m, 247)
    cv2.rectangle(p, (m, 255), (PANEL_WIDTH-m, 279), (42,32,52), -1)
    compact_label(p, 'INSPECTION', m, 308)
    half = (w-8)//2
    for i, title in enumerate(('CONTOUR AREA', 'CIRCULARITY', 'PREDICTION', 'CONFIDENCE')):
        row, col = divmod(i, 2)
        x, y = m+col*(half+8), 318+row*68
        cv2.rectangle(p, (x,y), (x+half,y+59), (42,42,58), -1)
        compact_label(p, title, x+6, y+15, .34)
    compact_label(p, 'SYSTEM PERFORMANCE', m, 472)
    return p


def draw_compact_panel(base, st, thumb, camera_fps, vision_fps, display_fps, active_count):
    p = base.copy()
    m, w = 14, PANEL_WIDTH-28
    color = st['color']
    green, red = color == (0,255,120), color == (0,60,255)
    bg = (20,60,35) if green else (30,30,75) if red else (35,35,48)
    cv2.rectangle(p, (m, 72), (PANEL_WIDTH-m, 138), bg, -1)
    cv2.rectangle(p, (m, 72), (PANEL_WIDTH-m, 138), color, 2)
    compact_value(p, st['status'], m+10, 110, w-20, .68, (255,255,255))
    third = (w-12)//3
    for i, (v, col) in enumerate(((st['total'], (240,240,250)),
                                  (st['passed'], (60,255,120)),
                                  (st['failed'], (100,100,255)))):
        compact_value(p, v, m+i*(third+6)+6, 211, third-12, .86, col)
    done = st['passed']+st['failed']
    ratio = st['passed']/done if done else 0
    if ratio:
        cv2.rectangle(p, (m,255), (m+round(w*ratio),279), (24,165,82), -1)
    compact_value(p, f'{ratio*100:.1f}% ({st["passed"]}/{done})', m+8, 273, w-16, .52)
    half = (w-8)//2
    vals = (f'{st["area"]:.0f}', f'{st["circularity"]:.2f}', st['prediction'],
            f'{st["confidence"]:.1f}%' if st['confidence'] else '--')
    for i, val in enumerate(vals):
        row, col = divmod(i, 2)
        compact_value(p, val, m+col*(half+8)+6, 318+row*68+47, half-12,
                      .65, (255,230,120) if row == 0 else (110,230,255))
    compact_value(p, f'AI {st["latency"]:.1f} ms   Tracks {active_count}', m, 497, w, .47)
    compact_value(p, f'Camera {camera_fps:.0f}   Vision {vision_fps:.0f}   UI {display_fps:.0f} FPS',
                  m, 519, w, .42)
    compact_value(p, f'Pending {st["pending"]}   Skipped {st["skipped"]}', m, 541, w, .46)
    # Small-screen dashboard deliberately omits the model thumbnail: no clipping.
    compact_label(p, 'Q: EXIT', m, SCREEN_HEIGHT-18, .43)
    return p


def make_panel_background():
    """One-time render of static panel borders, section titles and metric boxes."""
    if SCREEN_HEIGHT <= 850:
        return make_compact_background()
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
    if SCREEN_HEIGHT <= 850:
        return draw_compact_panel(base, st, thumb, camera_fps, vision_fps, display_fps, active_count)
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


def preview_text(canvas, lines, anchor='bottom'):
    """Compact readable explanation box, within either 1024x600 or 1080p."""
    h, w = canvas.shape[:2]
    font = .43 if w < 800 else .62
    line_h = 23 if w < 800 else 29
    top = (h - len(lines)*line_h - 17) if anchor == 'bottom' else 76
    top = max(5, top)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, top), (w-8, min(h-6,top+len(lines)*line_h+14)),
                  (12, 18, 26), -1)
    cv2.addWeighted(overlay, .78, canvas, .22, 0, canvas)
    for i, line in enumerate(lines):
        scale = font
        while scale > .27 and cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0] > w-30:
            scale -= .03
        cv2.putText(canvas,line,(18,top+19+i*line_h),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,(255,255,255) if i == 0 else (180,220,240),1,cv2.LINE_AA)


def render_camera(frame, roi_box, circles, display_fps, rim_marks=()):
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

    settings_preview = ui_state['tab'] == 'settings'
    chosen = ui_state['selected_setting']
    if settings_preview:
        # Recalculate the displayed ROI from the LIVE settings rather than from
        # the delayed vision frame. Feedback therefore appears immediately.
        side = min(ROI_SIZE, w, h)
        rx = max(0, min((w-side)//2 + ROI_OFFSET_X, w-side))
        ry = max(0, min((h-side)//2 + ROI_OFFSET_Y, h-side))
        roi_box = (rx,ry,side)
    if roi_box is not None:
        rx,ry,side = roi_box
        X = lambda value: round((value-ox)*sx)
        Y = lambda value: round((value-oy)*sy)
        roi_color = (60,200,255) if settings_preview and chosen.startswith('ROI_') else (0,255,120)
        cv2.rectangle(canvas,(X(rx),Y(ry)),(X(rx+side),Y(ry+side)),roi_color,3 if settings_preview else 2)
        line_y = Y(ry + side * COUNT_LINE_FRACTION)
        ai_line_y = Y(ry + side * AI_CAPTURE_FRACTION)
        line_color = (40,120,255) if settings_preview and chosen == 'COUNT_LINE_FRACTION' else (255,255,0)
        cv2.line(canvas,(X(rx),line_y),(X(rx+side),line_y),line_color,
                 4 if settings_preview and chosen == 'COUNT_LINE_FRACTION' else 2)
        # Magenta line = where the model image is taken; cyan = counting.
        ai_line_color = (255, 70, 230)
        cv2.line(canvas, (X(rx),ai_line_y),(X(rx+side),ai_line_y),ai_line_color,
                 4 if settings_preview and chosen == 'AI_CAPTURE_FRACTION' else 2)
        if settings_preview:
            cv2.putText(canvas,f'AI CAPTURE {AI_CAPTURE_FRACTION*100:.0f}%',
                        (max(8,X(rx)+6),max(22,min(SCREEN_HEIGHT-15,ai_line_y-9))),
                        cv2.FONT_HERSHEY_SIMPLEX,.44 if cam_w < 800 else .57,ai_line_color,2,cv2.LINE_AA)
            cv2.putText(canvas,f'COUNT {COUNT_LINE_FRACTION*100:.0f}%  UP',
                        (max(8,X(rx)+6),max(22,min(SCREEN_HEIGHT-15,line_y-9))),
                        cv2.FONT_HERSHEY_SIMPLEX,.48 if cam_w < 800 else .65,line_color,2,cv2.LINE_AA)
        # Show both detected contours instead of one misleading yellow circle:
        # blue = outer rim, yellow = inner/bottom rim.
        if rim_marks:
            for cx,cy,r,kind in rim_marks:
                center=(X(rx+cx),Y(ry+cy))
                cv2.circle(canvas,center,max(1,round(r*min(sx,sy))),
                           (255,210,60) if kind == 'outer' else (0,255,255),2)
        else:
            for cx,cy,r in circles:
                center=(X(rx+cx),Y(ry+cy))
                cv2.circle(canvas,center,max(1,round(r*min(sx,sy))),(0,255,255),2)
        if settings_preview:
            # Radius/area threshold rings are a visual scale reference. They do
            # not represent additional detected cans or change detection logic.
            cx_ref = X(rx+side*.50)
            cy_ref = Y(ry+side*.69)
            scale_ref = min(sx,sy)
            if chosen in ('MIN_CAN_RADIUS','MAX_CAN_RADIUS','MIN_CONTOUR_AREA','MAX_CONTOUR_AREA',
                          'OUTER_MIN_AREA','OUTER_MAX_AREA','OUTER_MIN_RADIUS','OUTER_MAX_RADIUS'):
                if chosen in ('MIN_CAN_RADIUS','MAX_CAN_RADIUS','OUTER_MIN_RADIUS','OUTER_MAX_RADIUS'):
                    r_min,r_max = ((OUTER_MIN_RADIUS,OUTER_MAX_RADIUS) if chosen.startswith('OUTER')
                                   else (MIN_CAN_RADIUS,MAX_CAN_RADIUS))
                    subtitle = f'Radius allowed: {r_min} - {r_max} camera px'
                else:
                    amin,amax = ((OUTER_MIN_AREA,OUTER_MAX_AREA) if chosen.startswith('OUTER')
                                 else (MIN_CONTOUR_AREA,MAX_CONTOUR_AREA))
                    r_min = np.sqrt(amin/np.pi)
                    r_max = np.sqrt(amax/np.pi)
                    subtitle = f'Area allowed: {amin:,} - {amax:,} px2'
                cv2.circle(canvas,(cx_ref,cy_ref),max(1,round(r_min*scale_ref)),(40,240,80),2)
                cv2.circle(canvas,(cx_ref,cy_ref),max(1,round(r_max*scale_ref)),(0,155,255),2)
                preview_text(canvas,['GREEN = minimum  |  ORANGE = maximum',subtitle])
            elif chosen in ('RIM_CENTER_ALIGNMENT_PX','INNER_ONLY_REVIEW_FRAMES'):
                preview_text(canvas,['Capture requires aligned INNER + OUTER rim centers',
                         f'Alignment tolerance: {RIM_CENTER_ALIGNMENT_PX}px; review after {INNER_ONLY_REVIEW_FRAMES} inner-only frames'])
            elif chosen == 'TRACK_MATCH_DISTANCE':
                # Show the search distance around a current tracked center, when
                # available, or a stable ROI reference point otherwise.
                center = (X(rx+circles[0][0]),Y(ry+circles[0][1])) if circles else (cx_ref,cy_ref)
                cv2.circle(canvas,center,round(TRACK_MATCH_DISTANCE*scale_ref),(255,100,255),2)
                preview_text(canvas,['PURPLE = matching distance',
                                      f'Track match limit: {TRACK_MATCH_DISTANCE} camera px'])
            elif chosen == 'AI_CAPTURE_FRACTION':
                preview_text(canvas,['MAGENTA = AI image capture position',
                                     'Changes model image, NOT can counting'])
            elif chosen == 'COUNT_LINE_FRACTION':
                preview_text(canvas,['ORANGE LINE = can counting position',
                                      'UPWARD crossing counts each tracked can once'])
            elif chosen.startswith('ROI_'):
                preview_text(canvas,['YELLOW BOX = detection area',
                                     f'ROI offset X: {ROI_OFFSET_X}  Y: {ROI_OFFSET_Y} px'])
            elif chosen == 'MIN_CIRCULARITY':
                preview_text(canvas,['Rim shape acceptance threshold',
                                      f'Below {MIN_CIRCULARITY:.2f} = defective rim'])
            elif chosen == 'TRACK_CONFIRM_FRAMES':
                preview_text(canvas,['Frames needed to confirm a can',
                                      f'Current: {TRACK_CONFIRM_FRAMES} vision frames'])
    label = f'FPS: {display_fps:.1f}'
    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,.8,2)
    cv2.rectangle(canvas,(25,25),(53+tw,49+th),(20,20,25),-1)
    cv2.rectangle(canvas,(25,25),(53+tw,49+th),(70,70,95),2)
    cv2.putText(canvas,label,(39,33+th),cv2.FONT_HERSHEY_DUPLEX,.8,(0,255,120),2,cv2.LINE_AA)
    return canvas


def run_station():
    """The original main(). Returns 'Normal Exit (Q)' when Q is pressed."""
    exit_status = None
    configure_display()
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
    cap = open_configured_camera()
    try:
        print('[INFO] Capture does not wait for vision; queue overflow means frames were lost.',flush=True)
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
            cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)
            if not RAW_PREVIEW:
                cv2.setWindowProperty(WINDOW_NAME,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        if SHOW_DASHBOARD and not RAW_PREVIEW:
            cv2.setMouseCallback(WINDOW_NAME,on_dashboard_mouse)
        base_panel = make_panel_background() if SHOW_DASHBOARD else None
        display_count, display_start, last_display = 0,time.monotonic(),0.
        display_fps = 0.
        interval = 1/max(DISPLAY_FPS,1)
        last_stats = 0.
        last_image_at = -1.
        newest_age_ms = 0.
        paint_ms = show_ms = key_ms = 0.
        actual_new_frames = 0
        print('[DIAG] Mode:', 'RAW' if RAW_PREVIEW else 'FULL', '| Vision:', 'OFF' if DISABLE_VISION else 'ON', '| Count direction:', 'UP' if COUNT_DIRECTION < 0 else 'DOWN', flush=True)
        # The main thread alone owns OpenCV GUI operations.
        last_metrics_publish = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now - last_metrics_publish >= 1.0:
                publish_shared_metrics()
                last_metrics_publish = now
            if SHOW_DASHBOARD and now-last_display >= interval:
                with shared_lock:
                    frame = shared['frame']
                    frame_at = shared['frame_at']
                    vision_frame_at = shared['vision_frame_at']
                    roi_box = shared['roi_box']
                    circles = shared['circles']
                    rim_marks = shared['rim_marks']
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
                        # Fit raw preview to the detected screen without changing capture resolution.
                        fh, fw = frame.shape[:2]
                        fit = min(SCREEN_WIDTH/fw, SCREEN_HEIGHT/fh)
                        canvas = cv2.resize(frame, (max(1,round(fw*fit)), max(1,round(fh*fit))),
                                            interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_LINEAR)
                        cv2.putText(canvas, f'RAW / age {newest_age_ms:.0f} ms / display {display_fps:.1f} FPS',
                                    (15,35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                    else:
                        overlay_age = max(0.0, frame_at - vision_frame_at)
                        if overlay_age > 0.25:
                            circles = ()
                            if ui_state['tab'] != 'settings':
                                roi_box = None
                        camera_canvas = render_camera(frame,roi_box,circles,display_fps,rim_marks)
                        panel = draw_hmi_panel(base_panel,st,thumb,capture_fps,vision_fps,display_fps,active)
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
                    exit_status = 'Normal Exit (Q)'
                    break
                if time.monotonic()-last_stats >= 1.:
                    with shared_lock:
                        cfps = shared['capture_fps']
                        vfps = shared['vision_fps']
                        capture_age_ms = max(0., (time.monotonic()-shared['frame_at'])*1000.) if shared['frame_at'] else -1.
                    print(f'[PERF] capture={cfps:.1f} vision={vfps:.1f} display={display_fps:.1f} '
                          f'fresh_preview_frames={actual_new_frames}/s frame_age={capture_age_ms:.0f}ms '
                          f'build={paint_ms:.1f}ms imshow={show_ms:.1f}ms waitKey={key_ms:.1f}ms '
                          f'vision_queue={frame_queue.qsize()}/{VISION_QUEUE_SIZE} '
                          f'overflow={frame_queue_stats["overflow"]}',flush=True)
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
    return exit_status


def main():
    """[DB] Wraps run_station() so every run writes one sessions row, like v2."""
    session_start = datetime.now().astimezone()
    run_info['session_start'] = session_start
    publish_shared_metrics()  # Reset stale values as soon as a session starts.
    exit_status, error_message = 'Unexpected Exit', None
    writer = threading.Thread(target=db_writer, daemon=True, name='sqlite') if DB_ENABLED else None
    if writer is not None:
        writer.start()
    image_writer = None
    if SAVE_MODEL_IMAGES:
        image_directory = (Path(__file__).resolve().parent / 'training_images' /
                           f'{session_start:%Y-%m-%d_%H-%M-%S_%f}')
        image_writer = threading.Thread(target=model_image_writer, args=(image_directory,),
                                        name='model_image_writer', daemon=False)
        image_writer.start()
        print(f'[IMAGES] Model inputs will be saved to {image_directory}', flush=True)
    try:
        exit_status = run_station() or 'Stopped'
    except KeyboardInterrupt:
        exit_status = 'Interrupted (Ctrl+C)'
        print('\n[INFO] Stopped')
    except Exception as exc:
        exit_status = 'Crash / Python Exception'
        error_message = f'{type(exc).__name__}: {exc}'
        raise  # Preserve the original traceback in the terminal.
    finally:
        stop_event.set()
        # Inference worker has already been joined by run_station. Finish all
        # enqueued images before closing the image dataset for this session.
        if image_writer is not None:
            image_save_queue.put(None)
            image_save_queue.join()
            image_writer.join()
        publish_shared_metrics()  # Ensure final counts reach Streamlit on exit.
        if run_info['fatal_error'] and error_message is None:
            exit_status = 'Crash / Hailo Error'
            error_message = run_info['fatal_error']
        # Snapshot final counters once for both text report and SQLite session.
        with shared_lock:
            final_state = dict(shared['state'])
        session_end = datetime.now().astimezone()
        try:
            save_session_log(session_start, session_end,
                             final_state['graded'] + run_info['prior_resets']['graded'],
                             final_state['passed'] + run_info['prior_resets']['passed'],
                             final_state['failed'] + run_info['prior_resets']['failed'],
                             exit_status, error_message)
        except Exception as log_error:
            # A file-system problem must not prevent the SQLite session report.
            print(f'[ERROR] Could not save session text log: {log_error}',
                  file=sys.stderr, flush=True)
        if writer is not None:
            # Workers are joined by now, so these are the final counters.
            st = final_state
            # [DB] save this session's report (image_count = cans that reached the trigger window;
            # cans still waiting on the Hailo at exit become no_result_count)
            db_queue.put(('session', (session_start, session_end,
                                      st['graded'] + run_info['prior_resets']['graded'],
                                      st['passed'] + run_info['prior_resets']['passed'],
                                      st['failed'] + run_info['prior_resets']['failed'],
                                      exit_status, error_message)))
            db_queue.put(None)
            writer.join(timeout=10)
            if writer.is_alive():
                print('[ERROR] SQLite writer did not finish; some rows may be missing', file=sys.stderr)


if __name__ == '__main__':
    try:
        if '--camera-test' in sys.argv:
            run_camera_test()
        else:
            main()
    except KeyboardInterrupt:
        print('\n[INFO] Stopped')
    finally:
        stop_event.set()