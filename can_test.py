import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(1)

square_size = 200
cooldown_time = 3
last_taken = 0

print("Press 'q' to quit.")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

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

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1500:
            current_time = time.time()

            if current_time - last_taken > cooldown_time:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                filename = f"can_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Picture saved as {filename}")

                last_taken = current_time
            break

    cv2.imshow("Can Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.contourAreadestroyAllWindows()