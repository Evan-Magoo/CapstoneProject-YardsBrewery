import cv2
import time
import numpy as np
from pathlib import Path

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
time.sleep(1)

PROJECT_DIR = Path(__file__).resolve().parent
subfolder = PROJECT_DIR / "test_images"
subfolder.mkdir(parents=True, exist_ok=True)

square_size = 1000
cooldown_time = 3
last_taken = 0
image_count = 0

print("Press 'q' to quit.")

#object_detector = cv2.createBackgroundSubtractorMOG2(history=25, varThreshold=100, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
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
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    mask = cv2.Canny(blurred, 50, 200)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        # Find the largest contour directly once
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        print(area)
        if area > 400:
            perimeter = cv2.arcLength(c, True)
            
            # Calculate the circularity
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            print(f"Circularity: {circularity:.2f}, Area: {area}")

            if circularity > 0.85:
                current_time = time.time()
                image_count += 1
                print(f"{image_count}. Circle Seen")

                if current_time - last_taken > cooldown_time:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                    filename = f"can_{int(current_time)}.jpg"
                    filepath = subfolder / filename
                    
                    # Convert Path to string for safe cross-version compatibility
                    cv2.imwrite(str(filepath), og_frame)
                    print(f"Picture saved as {filename}")

                    last_taken = current_time

    # Resize mask for side-by-side display
    h1, w1 = frame.shape[:2]
    new_w1 = int(mask.shape[1] * h1 / mask.shape[0])
    mask_resized = cv2.resize(mask, (new_w1, h1), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    result = cv2.hconcat([frame, mask_resized])

    result_w = int(result.shape[1] * 0.5)
    result_h = int(result.shape[0] * 0.5)

    resized_result = cv2.resize(result, (result_w, result_h), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Can Detector", resized_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()