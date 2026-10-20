from ultralytics import YOLO
import cv2
import time
import numpy as np

# 1. Initialize Camera and YOLO Model once at startup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
time.sleep(1)

model = YOLO("can_grading_model.pt")

square_size = 1000
cooldown_time = 2  # Time in seconds between inspections of new cans
last_taken = 0
image_count = 0
last_inspection_status = "Awaiting Can..."
last_color = (255, 255, 255)
last_yolo_text = "YOLO: Scanning..."

# Camera & Grading Stats (for diameter calculation if needed)
CAMERA_DISTANCE = 18.5
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028
THEORETICAL_PPI = (FOCAL_LENGTH := 4) / ((CAMERA_DISTANCE * 25.4) * PIXEL_PITCH) * 25.4 # simplified
CALIBRATION_FACTOR = 2.13 / 2.72
PIXELS_PER_INCH = ((4 / ((18.5 * 25.4) * 0.0028)) * 25.4) / CALIBRATION_FACTOR
MIN_CIRCULARITY = 0.85

print("Starting Live Can Inspection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    og_frame = frame.copy()
    height, width, _ = frame.shape

    # Define Region of Interest (ROI)
    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw tracking box on display frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for contour and circularity check
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (11, 11), 2)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use persistent variables for status display
    inspection_status = last_inspection_status
    color = last_color
    yolo_text = last_yolo_text

    # FIXED: Changed 'aif' to 'if'
    if cnts:
        print("Contours")

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        print(area)

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

                            last_yolo_text = f"YOLO: {top_class_name} ({top_class_conf*100:.1f}%)"

                            if top_class_name == "good_cans":
                                last_inspection_status = "PASS - GOOD CAN"
                                last_color = (0, 255, 0)
                            else:
                                last_inspection_status = "FAIL - BAD CLASSIFICATION"
                                last_color = (0, 0, 255)
                else:
                    # During cooldown, maintain status
                    pass 
            else:
                last_inspection_status = "FAIL - DEFECTIVE RIM"
                last_color = (0, 0, 255)

            # Draw using the persistent values
            cv2.circle(frame, (int(x + x1), int(y + y1)), int(radius), last_color, 2)
            cv2.putText(frame, last_inspection_status, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
    else:
        print("No Contours")

    # Draw text on frame BEFORE hconcat so it shows up in the final display window
    cv2.putText(frame, last_yolo_text, (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Prepare side-by-side edge mask view
    h1, w1 = frame.shape[:2]
    new_w1 = int(edged.shape[1] * h1 / edged.shape[0])
    mask_resized = cv2.resize(edged, (new_w1, h1), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    result = cv2.hconcat([frame, mask_resized])

    cv2.imshow("Live Can Inspection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()