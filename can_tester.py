import time
import sys

print("[INFO] Initializing system, please wait...")
start_init = time.time()

# Import heavy libraries AFTER the status print
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Faster camera initialization using DirectShow (Windows) or standard fallback
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 60)
time.sleep(1)  # Reduced sleep timer

# 2. Load YOLO Model
print("[INFO] Loading YOLO classification model...")
model = YOLO("can_grading_model.pt")

print(f"[INFO] System ready in {time.time() - start_init:.2f} seconds. Starting feed...")

square_size = 1000
cooldown_time = 2  
last_taken = 0
image_count = 0
pass_count = 0
fail_count = 0
last_inspection_status = "Awaiting Can..."
last_color = (255, 255, 255)
last_yolo_text = "YOLO: Scanning..."
last_confidence = 0.0

# Camera Stats
CAMERA_DISTANCE = 18.5
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028
THEORETICAL_PPI = (FOCAL_LENGTH := 4) / ((CAMERA_DISTANCE * 25.4) * PIXEL_PITCH) * 25.4
CALIBRATION_FACTOR = 2.13 / 2.72
PIXELS_PER_INCH = ((4 / ((18.5 * 25.4) * 0.0028)) * 25.4) / CALIBRATION_FACTOR
MIN_CIRCULARITY = 0.85

print("Starting Live Can Inspection with UI. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to load image")
        break

    og_frame = frame.copy()
    height, width, _ = frame.shape

    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (11, 11), 2)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 500:
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if circularity >= MIN_CIRCULARITY:
                current_time = time.time()

                if current_time - last_taken > cooldown_time:
                    last_taken = current_time
                    image_count += 1

                    results = model(og_frame)
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'probs') and result.probs is not None:
                            top_class_idx = result.probs.top1
                            top_class_name = result.names[top_class_idx]
                            top_class_conf = float(result.probs.top1conf)
                            
                            last_confidence = top_class_conf * 100
                            last_yolo_text = f"{top_class_name.upper()}"

                            if top_class_name == "good_cans":
                                last_inspection_status = "PASS - GOOD CAN"
                                last_color = (0, 255, 0)
                                pass_count += 1
                            else:
                                last_inspection_status = "FAIL - BAD CLASS"
                                last_color = (0, 0, 255)
                                fail_count += 1
            else:
                last_inspection_status = "FAIL - DEFECTIVE RIM"
                last_color = (0, 0, 255)
                current_time = time.time()
                if current_time - last_taken > cooldown_time:
                    last_taken = current_time
                    fail_count += 1
                    image_count += 1

            cv2.circle(frame, (int(x + x1), int(y + y1)), int(radius), last_color, 2)

    # UI Dashboard Panel
    panel_width = 380
    dashboard = np.zeros((height, panel_width, 3), dtype=np.uint8)
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

    cv2.putText(dashboard, "PRODUCTION METRICS", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    metrics_y = 260
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

    cv2.putText(dashboard, "Press 'q' to exit", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    ui_display = cv2.hconcat([frame, dashboard])
    cv2.imshow("Live Can Inspection Dashboard", ui_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()