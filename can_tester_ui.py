import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO

print("[INFO] Initializing system...")
start_init = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 200)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Turn OFF auto exposure (0.25 = Manual Mode) & set exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -4) 

time.sleep(1)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

model = YOLO("can_grading_model.pt")

# --- FULLSCREEN 1920x1080 16:9 SCREEN CONFIGURATION ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

PANEL_WIDTH = 560  # Slightly widened panel to fit larger text cleanly
CAM_REGION_WIDTH = SCREEN_WIDTH - PANEL_WIDTH  # 1360 pixels
CAM_REGION_HEIGHT = SCREEN_HEIGHT            # 1080 pixels
target_cam_aspect = CAM_REGION_WIDTH / CAM_REGION_HEIGHT

window_name = "Live Can Inspection Dashboard"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print(f"[INFO] System ready. Starting feed...")

square_size = 900
image_count = 0
pass_count = 0
fail_count = 0

last_inspection_status = "Awaiting Can..."
last_color = (255, 255, 255)
last_yolo_text = "SCANNING..."
last_confidence = 0.0
last_circularity = 0.0
last_area = 0

# --- PERFORMANCE & TRACKING VARIABLES ---
fps_prev_time = time.time()
fps = 0.0
last_latency_ms = 0.0

# --- SPATIAL OBJECT TRACKING VARIABLES ---
tracked_objects = {}  
next_object_id = 0
result_hold_until = 0
HOLD_DURATION = 5.0  

has_captured = False
last_captured_thumb = None

while True:
    loop_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to load image")
        break

    # Calculate real-time FPS
    current_time = time.time()
    time_diff = current_time - fps_prev_time
    if time_diff > 0:
        instant_fps = 1.0 / time_diff
        fps = (fps * 0.9) + (instant_fps * 0.1)
    fps_prev_time = current_time

    raw_h, raw_w, _ = frame.shape
    raw_aspect = raw_w / raw_h

    # --- DYNAMIC CENTER CROP TO FIT EXACT VIEWPORT (NO STRETCHING) ---
    if raw_aspect > target_cam_aspect:
        new_w = int(raw_h * target_cam_aspect)
        x_start = (raw_w - new_w) // 2
        cropped_frame = frame[:, x_start:x_start + new_w]
        crop_offset_x = x_start
        crop_offset_y = 0
    else:
        new_h = int(raw_w / target_cam_aspect)
        y_start = (raw_h - new_h) // 2
        cropped_frame = frame[y_start:y_start + new_h, :]
        crop_offset_x = 0
        crop_offset_y = y_start

    # Resize cropped view to fill camera region completely
    cam_canvas = cv2.resize(cropped_frame, (CAM_REGION_WIDTH, CAM_REGION_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Map ROI box coordinates dynamically onto the cropped viewport
    raw_x1 = ((raw_w - square_size) // 2) - 70
    raw_y1 = (raw_h - square_size) // 2
    
    x1 = int((raw_x1 - crop_offset_x) * (CAM_REGION_WIDTH / cropped_frame.shape[1]))
    y1 = int((raw_y1 - crop_offset_y) * (CAM_REGION_HEIGHT / cropped_frame.shape[0]))
    box_scaled_size = int(square_size * (CAM_REGION_WIDTH / cropped_frame.shape[1]))
    x2 = x1 + box_scaled_size
    y2 = y1 + box_scaled_size

    # Draw region of interest box on camera viewport
    cv2.rectangle(cam_canvas, (x1, y1), (x2, y2), (0, 255, 120), 4)
    roi = cam_canvas[y1:y2, x1:x2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (11, 11), 2)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centroids = []

    if cnts:
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 40000:  
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                current_centroids.append((int(x), int(y), c, area))
                cv2.circle(cam_canvas, (int(x + x1), int(y + y1)), int(radius), (0, 255, 255), 2)

    # --- CENTROID TRACKING ---
    updated_tracked_objects = {}
    for cx, cy, c, area in current_centroids:
        matched_id = None
        min_dist = 99999
        
        for obj_id, obj_data in tracked_objects.items():
            ox, oy = obj_data['centroid']
            dist = np.hypot(cx - ox, cy - oy)
            if dist < 90:
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
            updated_tracked_objects[next_object_id] = {
                'centroid': (cx, cy),
                'graded': False,
                'contour': c,
                'area': area
            }
            next_object_id += 1

    tracked_objects = updated_tracked_objects

    # --- SPATIAL INSPECTION TRIGGER ---
    INSPECTION_MIN_Y = y1 + int(box_scaled_size * 0.35)
    INSPECTION_MAX_Y = y1 + int(box_scaled_size * 0.65)

    for obj_id, obj_data in tracked_objects.items():
        cx, cy = obj_data['centroid']
        
        if not obj_data['graded'] and (INSPECTION_MIN_Y <= cy <= INSPECTION_MAX_Y):
            obj_data['graded'] = True  
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
                        
                        last_confidence = top_class_conf * 100
                        last_yolo_text = f"{top_class_name.upper()}"

                        if top_class_name == "good_cans":
                            last_inspection_status = "PASS - GOOD CAN"
                            last_color = (0, 255, 120)
                            pass_count += 1
                        else:
                            last_inspection_status = "FAIL - BAD CLASS"
                            last_color = (0, 60, 255)
                            fail_count += 1
            else:
                last_inspection_status = "FAIL - DEFECTIVE RIM"
                last_color = (0, 60, 255)
                fail_count += 1
                last_confidence = 0.0

    if current_time > result_hold_until and not tracked_objects:
        last_inspection_status = "Awaiting Can..."
        last_color = (140, 140, 160)
        last_yolo_text = "SCANNING..."
        last_confidence = 0.0
        has_captured = False
        last_captured_thumb = None

    # --- DRAW FPS COUNTER ON UPPER LEFT OF CAMERA VIEWPORT ---
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    badge_x1, badge_y1 = 25, 25
    badge_x2, badge_y2 = badge_x1 + tw + 28, badge_y1 + th + 24
    
    cv2.rectangle(cam_canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 25), -1)
    cv2.rectangle(cam_canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (70, 70, 95), 2)
    cv2.putText(cam_canvas, fps_text, (badge_x1 + 14, badge_y1 + th + 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 120), 2)

    # ==========================================
    # HIGH-READABILITY 1080p HMI DASHBOARD PANEL
    # ==========================================
    dashboard = np.zeros((SCREEN_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    dashboard[:] = (18, 18, 24)  # Rich slate-charcoal background

    margin = 22
    total_w = PANEL_WIDTH - (2 * margin)
    
    # Outer Border Box framing the entire right panel
    cv2.rectangle(dashboard, (10, 10), (PANEL_WIDTH - 10, SCREEN_HEIGHT - 10), (60, 60, 80), 2)

    # 1. Header Title & Accent Line
    cv2.putText(dashboard, "QUALITY INSPECTION HMI", (margin, 44), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2)
    cv2.line(dashboard, (margin, 58), (PANEL_WIDTH - margin, 58), (70, 70, 90), 2)

    # 2. Current Status Card
    cv2.putText(dashboard, "SYSTEM STATUS", (margin, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)
    
    card_bg = (
        (20, 60, 35) if last_color == (0, 255, 120) else 
        ((30, 30, 75) if last_color == (0, 60, 255) else (35, 35, 48))
    )
    card_border = (
        (0, 230, 110) if last_color == (0, 255, 120) else 
        ((0, 60, 230) if last_color == (0, 60, 255) else (80, 80, 105))
    )
    
    card_y1, card_y2 = 100, 172
    cv2.rectangle(dashboard, (margin, card_y1), (PANEL_WIDTH - margin, card_y2), card_bg, -1)
    cv2.rectangle(dashboard, (margin, card_y1), (PANEL_WIDTH - margin, card_y2), card_border, 2)
    
    cv2.circle(dashboard, (margin + 30, 136), 10, last_color, -1)
    cv2.putText(dashboard, last_inspection_status, (margin + 56, 145), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)

    # 3. Production Metrics Row (Total, Passed, Failed)
    metric_start_y = 196
    cv2.putText(dashboard, "PRODUCTION METRICS", (margin, metric_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)

    grid_y1 = metric_start_y + 10
    grid_y2 = grid_y1 + 70
    box_w = (total_w - 16) // 3

    metrics_data = [
        ("TOTAL", str(image_count), (240, 240, 250), (42, 42, 58)),
        ("PASSED", str(pass_count), (60, 255, 120), (22, 52, 35)),
        ("FAILED", str(fail_count), (100, 100, 255), (42, 32, 52))
    ]

    for i, (label, val, val_color, bg_col) in enumerate(metrics_data):
        bx1 = margin + i * (box_w + 8)
        bx2 = bx1 + box_w
        cv2.rectangle(dashboard, (bx1, grid_y1), (bx2, grid_y2), bg_col, -1)
        cv2.rectangle(dashboard, (bx1, grid_y1), (bx2, grid_y2), (75, 75, 95), 1)
        
        cv2.putText(dashboard, label, (bx1 + 8, grid_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 195), 1)
        cv2.putText(dashboard, val, (bx1 + 8, grid_y1 + 52), cv2.FONT_HERSHEY_DUPLEX, 0.72, val_color, 2)

    # --- PASS/FAIL RATIO PERCENTAGE BAR ---
    ratio_title_y = grid_y2 + 20
    cv2.putText(dashboard, "PASS RATIO DISTRIBUTION", (margin, ratio_title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)
    
    bar_y1 = ratio_title_y + 8
    bar_y2 = bar_y1 + 30

    total_inspected = pass_count + fail_count
    pass_ratio = (pass_count / total_inspected) if total_inspected > 0 else 0.0
    pass_percentage = pass_ratio * 100

    bar_x1 = margin
    bar_x2 = PANEL_WIDTH - margin
    bar_w = bar_x2 - bar_x1

    cv2.rectangle(dashboard, (bar_x1, bar_y1), (bar_x2, bar_y2), (42, 32, 52), -1)
    if total_inspected > 0 and pass_count > 0:
        pass_fill_w = int(bar_w * pass_ratio)
        cv2.rectangle(dashboard, (bar_x1, bar_y1), (bar_x1 + pass_fill_w, bar_y2), (24, 165, 82), -1)

    cv2.rectangle(dashboard, (bar_x1, bar_y1), (bar_x2, bar_y2), (75, 75, 95), 1)

    ratio_text = f"Pass Rate: {pass_percentage:.1f}%  ({pass_count} / {total_inspected})" if total_inspected > 0 else "Pass Rate: --% (0 / 0)"
    (rt_w, rt_h), _ = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    rt_x = bar_x1 + (bar_w - rt_w) // 2
    rt_y = bar_y1 + (bar_y2 - bar_y1 + rt_h) // 2 - 2
    cv2.putText(dashboard, ratio_text, (rt_x, rt_y), cv2.FONT_HERSHEY_DUPLEX, 0.48, (250, 250, 250), 1)

    # 4. OpenCV Vision Measurements Section
    opencv_start_y = bar_y2 + 20
    cv2.putText(dashboard, "OPENCV MEASUREMENTS", (margin, opencv_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)

    oc_y1 = opencv_start_y + 10
    oc_y2 = oc_y1 + 65
    oc_box_w = (total_w - 10) // 2

    opencv_data = [
        ("CONTOUR AREA", f"{last_area:.0f}", (255, 230, 120), (42, 42, 58)),
        ("CIRCULARITY", f"{last_circularity:.2f}", (255, 230, 120), (42, 42, 58))
    ]

    for i, (label, val, val_col, bg_col) in enumerate(opencv_data):
        bx1 = margin + i * (oc_box_w + 10)
        bx2 = bx1 + oc_box_w
        cv2.rectangle(dashboard, (bx1, oc_y1), (bx2, oc_y2), bg_col, -1)
        cv2.rectangle(dashboard, (bx1, oc_y1), (bx2, oc_y2), (75, 75, 95), 1)

        cv2.putText(dashboard, label, (bx1 + 10, oc_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 195), 1)
        cv2.putText(dashboard, val, (bx1 + 10, oc_y1 + 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, val_col, 2)

    # 5. YOLO AI Classifier Section
    yolo_start_y = oc_y2 + 20
    cv2.putText(dashboard, "YOLO AI CLASSIFIER", (margin, yolo_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)

    y_y1 = yolo_start_y + 10
    y_y2 = y_y1 + 65
    
    confidence_display = f"{last_confidence:.1f}%" if last_confidence > 0 else "--"
    yolo_data = [
        ("PREDICTION", last_yolo_text, (110, 230, 255), (42, 42, 58)),
        ("CONFIDENCE", confidence_display, (110, 230, 255), (42, 42, 58))
    ]

    for i, (label, val, val_col, bg_col) in enumerate(yolo_data):
        bx1 = margin + i * (oc_box_w + 10)
        bx2 = bx1 + oc_box_w
        cv2.rectangle(dashboard, (bx1, y_y1), (bx2, y_y2), bg_col, -1)
        cv2.rectangle(dashboard, (bx1, y_y1), (bx2, y_y2), (75, 75, 95), 1)

        cv2.putText(dashboard, label, (bx1 + 10, y_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 195), 1)
        cv2.putText(dashboard, val, (bx1 + 10, y_y1 + 48), cv2.FONT_HERSHEY_DUPLEX, 0.62, val_col, 2)

    # 6. SYSTEM PERFORMANCE SECTION
    perf_start_y = y_y2 + 20
    cv2.putText(dashboard, "SYSTEM PERFORMANCE", (margin, perf_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)

    p_y1 = perf_start_y + 10
    p_y2 = p_y1 + 65

    current_latency = (time.time() - loop_start_time) * 1000
    last_latency_ms = (last_latency_ms * 0.8) + (current_latency * 0.2)

    perf_data = [
        ("LOOP LATENCY", f"{last_latency_ms:.1f} ms", (255, 160, 110), (42, 42, 58)),
        ("ACTIVE TRACKERS", str(len(tracked_objects)), (255, 160, 110), (42, 42, 58))
    ]

    for i, (label, val, val_col, bg_col) in enumerate(perf_data):
        bx1 = margin + i * (oc_box_w + 10)
        bx2 = bx1 + oc_box_w
        cv2.rectangle(dashboard, (bx1, p_y1), (bx2, p_y2), bg_col, -1)
        cv2.rectangle(dashboard, (bx1, p_y1), (bx2, p_y2), (75, 75, 95), 1)

        cv2.putText(dashboard, label, (bx1 + 10, p_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 195), 1)
        cv2.putText(dashboard, val, (bx1 + 10, p_y1 + 48), cv2.FONT_HERSHEY_DUPLEX, 0.62, val_col, 2)

    # 7. MODEL INPUT PREVIEW SQUARE THUMBNAIL (Pinned centrally at bottom of HMI panel)
    thumb_title_y = p_y2 + 20
    cv2.putText(dashboard, "MODEL INPUT PREVIEW", (margin, thumb_title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 190), 1)
    
    square_side = min(total_w, SCREEN_HEIGHT - (thumb_title_y + 45))
    thumb_x1 = margin + (total_w - square_side) // 2
    thumb_x2 = thumb_x1 + square_side
    thumb_y1 = thumb_title_y + 8
    thumb_y2 = thumb_y1 + square_side

    cv2.rectangle(dashboard, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (25, 25, 35), -1)
    cv2.rectangle(dashboard, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (75, 75, 95), 2)

    if has_captured and last_captured_thumb is not None:
        resized_thumb = cv2.resize(last_captured_thumb, (square_side, square_side))
        dashboard[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = resized_thumb
        cv2.rectangle(dashboard, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), last_color, 3)
    else:
        cv2.putText(dashboard, "WAITING FOR CAN...", (thumb_x1 + int(square_side * 0.02), thumb_y1 + int(square_side * 0.98)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 130, 150), 2)

    # Footer Instructions & Exit Note
    cv2.putText(dashboard, "Press 'q' to exit system", (margin, SCREEN_HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 165), 1)

    # Combine camera viewport and dashboard side-by-side to fill 1920x1080
    ui_display = cv2.hconcat([cam_canvas, dashboard])

    cv2.imshow(window_name, ui_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()