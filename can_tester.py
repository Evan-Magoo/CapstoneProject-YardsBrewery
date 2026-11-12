import time
import sys
from inspection_metrics import MetricsRecorder
import cv2
import numpy as np
from ultralytics import YOLO

print("[INFO] Initializing system...")
start_init = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)
time.sleep(1)

model = YOLO("can_grading_model.pt")

# --- DISPLAY & SCREEN CONFIGURATION ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
TARGET_DISPLAY_HEIGHT = 1080 

window_name = "Live Can Inspection Dashboard"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print(f"[INFO] System ready. Starting feed...")

square_size = 600
image_count = 0
pass_count = 0
fail_count = 0

last_inspection_status = "Awaiting Can..."
last_color = (255, 255, 255)
last_yolo_text = "YOLO: Scanning..."
last_confidence = 0.0
last_circularity = 0.0
last_area = 0

# --- SPATIAL OBJECT TRACKING VARIABLES ---
tracked_objects = {}  # Format: {obj_id: {'centroid': (x, y), 'graded': bool, 'contour': c, 'area': area}}
next_object_id = 0
result_hold_until = 0
HOLD_DURATION = 2.0  # Keeps results frozen on UI for operator reading

has_captured = False
last_captured_thumb = None

metrics = MetricsRecorder()

while True:
    metrics.heartbeat()
    ret, frame = cap.read()
    if not ret:
        metrics.heartbeat("camera_error", force=True)
        print("Failed to load image")
        break

    height, width, _ = frame.shape
    scale_factor = TARGET_DISPLAY_HEIGHT / 1080.0

    x1 = ((width - square_size) // 2) - 50
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw region of interest box on camera feed
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (11, 11), 2)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_time = time.time()
    current_centroids = []

    if cnts:
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 50000:  # Valid can contour size threshold
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                current_centroids.append((int(x), int(y), c, area))
                cv2.circle(frame, (int(x + x1), int(y + y1)), int(radius), (0, 255, 255), 2)

    # --- SPEED-AGNOSTIC CENTROID TRACKING ---
    updated_tracked_objects = {}
    for cx, cy, c, area in current_centroids:
        matched_id = None
        min_dist = 99999
        
        # Match current contour to previously tracked cans across frames
        for obj_id, obj_data in tracked_objects.items():
            ox, oy = obj_data['centroid']
            dist = np.hypot(cx - ox, cy - oy)
            if dist < 85:  # Max pixel travel between consecutive frames
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
        
        if matched_id is not None:
            updated_tracked_objects[matched_id] = {
                'centroid': (cx, cy),
                'graded': tracked_objects[matched_id]['graded'],
                'contour': c,
                'area': area
            }
        else:
            # New can entered the ROI
            updated_tracked_objects[next_object_id] = {
                'centroid': (cx, cy),
                'graded': False,
                'contour': c,
                'area': area
            }
            next_object_id += 1

    tracked_objects = updated_tracked_objects

    # --- SPATIAL INSPECTION TRIGGER (SWEET SPOT ZONE) ---
    # The center of the 600x600 ROI box is Y = 300. 
    # We inspect cans as their centroids pass through the 250 to 350 window.
    INSPECTION_MIN_Y = 280
    INSPECTION_MAX_Y = 380

    for obj_id, obj_data in tracked_objects.items():
        cx, cy = obj_data['centroid']
        
        if not obj_data['graded'] and (INSPECTION_MIN_Y <= cy <= INSPECTION_MAX_Y):
            obj_data['graded'] = True  # Permanently lock this specific can object so it's never re-read
            grading_started = time.perf_counter()
            event_outcome, event_reason = "ERROR", "No classification returned"
            event_class, event_confidence = None, None
            image_count += 1
            result_hold_until = current_time + HOLD_DURATION

            last_captured_thumb = roi.copy()
            has_captured = True

            c = obj_data['contour']
            area = obj_data['area']
            last_area = area

            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            last_circularity = circularity

            if circularity >= 0.85:
                resized_roi = cv2.resize(roi, (320, 320))
                results = model(resized_roi, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'probs') and result.probs is not None:
                        top_class_idx = result.probs.top1
                        top_class_name = result.names[top_class_idx]
                        top_class_conf = float(result.probs.top1conf)
                        
                        event_class, event_confidence = top_class_name, top_class_conf
                        last_confidence = top_class_conf * 100
                        last_yolo_text = f"{top_class_name.upper()}"

                        if top_class_name == "good_cans":
                            last_inspection_status = "PASS - GOOD CAN"
                            last_color = (0, 255, 0)
                            pass_count += 1
                            event_outcome, event_reason = "PASS", "Good can"
                        else:
                            last_inspection_status = "FAIL - BAD CLASS"
                            last_color = (0, 0, 255)
                            fail_count += 1
                            event_outcome, event_reason = "FAIL", "Bad classification"
            else:
                last_inspection_status = "FAIL - DEFECTIVE RIM"
                last_color = (0, 0, 255)
                fail_count += 1
                last_confidence = 0.0
                event_outcome, event_reason = "FAIL", "Defective rim"

            metrics.record(obj_id, event_outcome, event_reason, event_class,
                           event_confidence, circularity,
                           (time.perf_counter() - grading_started) * 1000)

    # Clear status back to awaiting only after the UI hold timer expires and no cans are active
    if current_time > result_hold_until and not tracked_objects:
        last_inspection_status = "Awaiting Can..."
        last_color = (120, 120, 120)
        last_yolo_text = "YOLO: Scanning..."
        last_confidence = 0.0
        has_captured = False
        last_captured_thumb = None

    # ==========================================
    # MODERN HMI DASHBOARD PANEL
    # ==========================================
    panel_width = int(420 * scale_factor)
    dashboard = np.zeros((height, panel_width, 3), dtype=np.uint8)
    dashboard[:] = (22, 22, 26)

    margin = int(24 * scale_factor)
    line_spacing = int(20 * scale_factor)
    
    title_font = max(0.5, 0.75 * scale_factor)
    section_font = max(0.38, 0.48 * scale_factor)
    text_font = max(0.4, 0.52 * scale_factor)

    # 1. Header Title & Accent Line
    cv2.putText(dashboard, "QUALITY INSPECTION HMI", (margin, int(45 * scale_factor)), cv2.FONT_HERSHEY_DUPLEX, title_font, (245, 245, 245), 1)
    cv2.line(dashboard, (margin, int(60 * scale_factor)), (panel_width - margin, int(60 * scale_factor)), (60, 60, 70), 2)

    # 2. System Status Card
    cv2.putText(dashboard, "CURRENT STATUS", (margin, int(105 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, section_font, (140, 140, 160), 1)
    
    card_bg = (
        (18, 45, 18) if last_color == (0, 255, 0) else 
        ((18, 18, 55) if last_color == (0, 0, 255) else (40, 40, 50))
    )
    card_border = (
        (0, 200, 0) if last_color == (0, 255, 0) else 
        ((0, 0, 200) if last_color == (0, 0, 255) else (80, 80, 95))
    )
    
    card_y1 = int(120 * scale_factor)
    card_y2 = int(185 * scale_factor)
    cv2.rectangle(dashboard, (margin, card_y1), (panel_width - margin, card_y2), card_bg, -1)
    cv2.rectangle(dashboard, (margin, card_y1), (panel_width - margin, card_y2), card_border, 1)
    
    cv2.circle(dashboard, (margin + int(24 * scale_factor), int(152 * scale_factor)), int(7 * scale_factor), last_color, -1)
    cv2.putText(dashboard, last_inspection_status, (margin + int(42 * scale_factor), int(158 * scale_factor)), cv2.FONT_HERSHEY_DUPLEX, text_font, (255, 255, 255), 1)

    def draw_row(y_pos, label, val, val_color=(230, 230, 230)):
        cv2.putText(dashboard, label, (margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, text_font, (170, 170, 185), 1)
        cv2.putText(dashboard, str(val), (panel_width - margin - int(100 * scale_factor), y_pos), cv2.FONT_HERSHEY_DUPLEX, text_font, val_color, 1)

    # 3. Production Metrics Section
    metrics_start_y = int(235 * scale_factor)
    cv2.putText(dashboard, "PRODUCTION METRICS", (margin, metrics_start_y), cv2.FONT_HERSHEY_SIMPLEX, section_font, (140, 140, 160), 1)
    
    row_y = metrics_start_y + int(38 * scale_factor)
    draw_row(row_y, "Total Inspected:", image_count, (255, 255, 255))
    row_y += line_spacing
    draw_row(row_y, "Passed Cans:", pass_count, (60, 255, 60))
    row_y += line_spacing
    draw_row(row_y, "Failed Cans:", fail_count, (60, 60, 255))

    # 4. OpenCV Vision Measurements Section
    row_y += int(38 * scale_factor)
    cv2.putText(dashboard, "OPENCV MEASUREMENTS", (margin, row_y), cv2.FONT_HERSHEY_SIMPLEX, section_font, (140, 140, 160), 1)
    
    row_y += int(20 * scale_factor)
    draw_row(row_y, "Contour Area:", f"{last_area:.0f}", (255, 220, 100))
    row_y += line_spacing
    draw_row(row_y, "Circularity:", f"{last_circularity:.2f}", (255, 220, 100))

    # 5. YOLO AI Classifier Insights Section
    row_y += int(45 * scale_factor)
    cv2.putText(dashboard, "YOLO AI CLASSIFIER", (margin, row_y), cv2.FONT_HERSHEY_SIMPLEX, section_font, (140, 140, 160), 1)
    
    row_y += int(20 * scale_factor)
    draw_row(row_y, "Model Prediction:", last_yolo_text, (100, 220, 255))
    
    row_y += line_spacing
    confidence_display = f"{last_confidence:.1f}%" if last_confidence > 0 else "--"
    draw_row(row_y, "Confidence Score:", confidence_display, (100, 220, 255))

    # 6. Model Input Preview Thumbnail Card
    row_y += int(40 * scale_factor)
    cv2.putText(dashboard, "MODEL INPUT PREVIEW", (margin, row_y), cv2.FONT_HERSHEY_SIMPLEX, section_font, (140, 140, 160), 1)
    
    row_y += int(10 * scale_factor)
    thumb_size = int(120 * scale_factor)
    thumb_x1 = (panel_width - thumb_size) // 2
    thumb_x2 = thumb_x1 + thumb_size
    thumb_y1 = row_y
    thumb_y2 = thumb_y1 + thumb_size

    cv2.rectangle(dashboard, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (35, 35, 45), -1)
    cv2.rectangle(dashboard, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (80, 80, 95), 1)

    if has_captured and last_captured_thumb is not None:
        resized_thumb = cv2.resize(last_captured_thumb, (thumb_size, thumb_size))
        dashboard[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = resized_thumb
    else:
        cv2.putText(dashboard, "WAITING...", (thumb_x1 + int(22 * scale_factor), thumb_y1 + int(65 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.45 * scale_factor), (120, 120, 130), 1)

    cv2.putText(dashboard, "Press 'q' to exit system", (margin, height - int(25 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, max(0.32, 0.4 * scale_factor), (110, 110, 125), 1)

    ui_display = cv2.hconcat([frame, dashboard])
    ui_h, ui_w = ui_display.shape[:2]

    centered_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    x_offset = max(0, (SCREEN_WIDTH - ui_w) // 2)
    y_offset = max(0, (SCREEN_HEIGHT - ui_h) // 2)
    centered_canvas[y_offset:y_offset + ui_h, x_offset:x_offset + ui_w] = ui_display

    cv2.imshow(window_name, centered_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        metrics.heartbeat('stopped', force=True)
        break

cap.release()
cv2.destroyAllWindows()