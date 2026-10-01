from ultralytics import YOLO
import cv2
import os
import time

# -----------------------------
# SETTINGS
# -----------------------------
CAMERA_INDEX = 1
NUMBER_OF_PICTURES = 2

# Create folder for pictures
os.makedirs("captured_images", exist_ok=True)

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

# Try to increase camera FPS
cap.set(cv2.CAP_PROP_FPS, 60)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened!")
print("Press 's' to capture multiple pictures.")
print("Press 'q' to quit.")


while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Show live camera
    cv2.imshow("High Speed Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press S to capture burst
    if key == ord("s"):

        print(f"Capturing {NUMBER_OF_PICTURES} pictures...")

        start_time = time.time()

        for i in range(NUMBER_OF_PICTURES):

            success, frame = cap.read()

            if success:

                filename = f"captured_images/photo_{i + 1}.jpg"

                cv2.imwrite(filename, frame)

                print(f"Saved: {filename}")

        end_time = time.time()

        total_time = end_time - start_time

        print("Capture complete!")
        print(f"Time: {total_time:.2f} seconds")

        if total_time > 0:
            print(
                f"Capture speed: "
                f"{NUMBER_OF_PICTURES / total_time:.2f} pictures/second"
            )

    # Press Q to quit
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()