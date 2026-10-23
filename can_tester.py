import time
import sys
from pathlib import Path

print("[INFO] Initializing system, please wait...")
start_init = time.time()

# Import heavy libraries AFTER the status print
from ultralytics import YOLO
import cv2
import numpy as np

from can_vision import open_camera, find_can, crop_can, can_fully_in_frame

PROJECT_DIR = Path(__file__).resolve().parent

# 1. Camera - opened at an explicit resolution (see can_vision.py)
try:
    cap = cv2.VideoCapture(1)
except RuntimeError as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

# 2. Load YOLO Model
print("[INFO] Loading YOLO classification model...")
model = YOLO(str("can_grading_model.pt"))

print(f"[INFO] System ready in {time.time() - start_init:.2f} seconds. Starting feed...")

# Inspection settings
CENTER_TOLERANCE = 0.15       # can center must be within 15% of frame width of the middle
REARM_FRAMES = 10             # frames with no centered can before the next can is inspected
cooldown_time = 0.5           # minimum seconds between inspections (backup; REARM_FRAMES does the real work)
DISPLAY_SCALE = 0.5           # 1080p is shown at half size so the window fits the screen
GOOD_CLASS = "good_cans"

last_taken = 0
image_count = 0
pass_count = 0
fail_count = 0
armed = True                  # True = ready to inspect the next can
frames_without_can = 0
last_inspection_status = "Awaiting Can..."
last_color = (255, 255, 255)
last_yolo_text = "YOLO: Scanning..."
last_confidence = 0.0

# Camera Stats (kept from the original - not used by the detection logic)
CAMERA_DISTANCE = 18.5
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028
THEORETICAL_PPI = (FOCAL_LENGTH := 4) / ((CAMERA_DISTANCE * 25.4) * PIXEL_PITCH) * 25.4
CALIBRATION_FACTOR = 2.13 / 2.72
PIXELS_PER_INCH = ((4 / ((18.5 * 25.4) * 0.0028)) * 25.4) / CALIBRATION_FACTOR

print("Starting Live Can Inspection with UI. Press 'q' to quit.")

cv2.namedWindow("Live Can Inspection Dashboard", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera")
        break

    height, width = frame.shape[:2]

    # Inspection zone: a vertical band in the middle of the frame
    zone_half = int(width * CENTER_TOLERANCE)
    zone_x1, zone_x2 = width // 2 - zone_half, width // 2 + zone_half
    cv2.rectangle(frame, (zone_x1, 0), (zone_x2, height - 1), (0, 255, 0), 2)

    circle = find_can(frame)
    can_in_position = (
        circle is not None
        and zone_x1 <= circle[0] <= zone_x2
        and can_fully_in_frame(frame, circle)
    )

    if can_in_position:
        frames_without_can = 0
        current_time = time.time()

        # Inspect each can once: when it reaches the zone and we are armed
        if armed and current_time - last_taken > cooldown_time:
            armed = False
            last_taken = current_time
            image_count += 1

            can_img = crop_can(frame, circle)
            results = model(can_img, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "probs") and result.probs is not None:
                    top_class_idx = result.probs.top1
                    top_class_name = result.names[top_class_idx]
                    top_class_conf = float(result.probs.top1conf)

                    last_confidence = top_class_conf * 100
                    last_yolo_text = f"{top_class_name.upper()}"

                    print(f"[INSPECT #{image_count}] {top_class_name} ({last_confidence:.1f}%)")

                    if top_class_name == GOOD_CLASS:
                        last_inspection_status = "PASS - GOOD CAN"
                        last_color = (0, 255, 0)
                        pass_count += 1
                    else:
                        last_inspection_status = "FAIL - BAD CAN"
                        last_color = (0, 0, 255)
                        fail_count += 1
    else:
        frames_without_can += 1
        if frames_without_can >= REARM_FRAMES:
            armed = True

    if circle is not None:
        x, y, r = circle
        ring_color = last_color if can_in_position else (0, 255, 255)
        cv2.circle(frame, (int(x), int(y)), int(r), ring_color, 4)
        cv2.circle(frame, (int(x), int(y)), 6, ring_color, -1)

    # Scale the camera image down for display
    view = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    view_h = max(view.shape[0], 560)
    if view.shape[0] < view_h:
        view = cv2.copyMakeBorder(view, 0, view_h - view.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)

    # UI Dashboard Panel
    panel_width = 380
    dashboard = np.zeros((view_h, panel_width, 3), dtype=np.uint8)
    dashboard[:] = (35, 35, 35)

    cv2.putText(dashboard, "CAN INSPECTION UI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(dashboard, (20, 55), (panel_width - 20, 55), (100, 100, 100), 1)

    cv2.putText(dashboard, "INSPECTION STATUS", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    card_color_bgr = (
        (0, 100, 0) if last_color == (0, 255, 0) else
        ((0, 0, 150) if last_color == (0, 0, 255) else (60, 60, 60))
    )
    cv2.rectangle(dashboard, (20, 110), (panel_width - 20, 170), card_color_bgr, -1)
    cv2.rectangle(dashboard, (20, 110), (panel_width - 20, 170), (200, 200, 200), 1)
    cv2.putText(dashboard, last_inspection_status, (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if circle is None:
        detect_text, detect_color = "Camera: no can seen", (150, 150, 150)
    elif can_in_position:
        detect_text, detect_color = "Camera: can in position", (0, 255, 0)
    else:
        detect_text, detect_color = "Camera: can approaching", (0, 255, 255)
    cv2.putText(dashboard, detect_text, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detect_color, 1)

    cv2.putText(dashboard, "PRODUCTION METRICS", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    metrics_y = 270
    cv2.putText(dashboard, f"Total Inspected:", (20, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(dashboard, f"{image_count}", (panel_width - 80, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    metrics_y += 40
    cv2.putText(dashboard, f"Passed Cans:", (20, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(dashboard, f"{pass_count}", (panel_width - 80, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    metrics_y += 40
    cv2.putText(dashboard, f"Failed Cans:", (20, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(dashboard, f"{fail_count}", (panel_width - 80, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(dashboard, "YOLO AI CLASSIFIER", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(dashboard, f"Model: {last_yolo_text}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(dashboard, f"Confidence: {last_confidence:.1f}%", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.putText(dashboard, "Press 'q' to exit", (20, view_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    ui_display = cv2.hconcat([view, dashboard])
    cv2.imshow("Live Can Inspection Dashboard", ui_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
