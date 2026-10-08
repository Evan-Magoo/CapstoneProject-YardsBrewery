import cv2
import time
from pathlib import Path

cap = cv2.VideoCapture(0)
time.sleep(1)

PROJECT_DIR = Path(__file__).resolve().parent
subfolder = PROJECT_DIR / "test_images"
subfolder.mkdir(parents=True, exist_ok=True)

square_size = 300
cooldown_time = 3
last_taken = 0
image = 0

print("Press 'q' to quit.")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    edged = cv2.Canny(roi, 50, 100)
    mask = object_detector.apply(edged)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=100, minRadius=10, maxRadius=1000)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            print(area)

        if area > 500 and circles is not None:
            current_time = time.time()

            image += 1
            print(f"{image}. Circle Seen")

            if current_time - last_taken > cooldown_time:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                filename = f"can_{int(current_time)}.jpg"
                filepath = subfolder / filename
                cv2.imwrite(filepath, frame)
                print(f"Picture saved as {filename}")

                last_taken = current_time
            break

    cv2.imshow("Can Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.contourAreadestroyAllWindows()