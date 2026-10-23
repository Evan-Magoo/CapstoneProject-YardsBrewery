"""
Capture tool for building training data.

[SPACE] saves two images:
  test_images/          - the can crop, exactly what the classifier sees live
  test_images/full/     - the full grayscale frame (for re-cropping later)
Sort the crops into training_data/train|val/good_cans|bad_cans, or put the full
frames there and run prepare_dataset.py.
"""

import sys
import time
from pathlib import Path

import cv2

from can_vision import open_camera, find_can, crop_can, to_gray

DISPLAY_SCALE = 0.5   # 1080p is shown at half size so the window fits the screen

try:
    cap = cv2.VideoCapture(1)
except RuntimeError as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

# Setup test_images directory
PROJECT_DIR = Path(__file__).resolve().parent
subfolder = PROJECT_DIR / "test_images"
full_folder = subfolder / "full"
full_folder.mkdir(parents=True, exist_ok=True)

image_count = 0
print("--- CAMERA CAPTURE ACTIVE ---")
print("Press [SPACEBAR] to take a picture.")
print("Press [q] to quit.")

window = "Capture Tool - [SPACE] to Save | [q] to Quit"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break

    gray = to_gray(frame)
    circle = find_can(frame)

    # Live preview: grayscale frame with the detected can outlined
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if circle is not None:
        x, y, r = circle
        cv2.circle(preview, (int(x), int(y)), int(r), (0, 255, 0), 4)
        status = "CAN DETECTED"
    else:
        status = "NO CAN DETECTED"
    preview = cv2.resize(preview, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    cv2.putText(preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if circle is not None else (0, 0, 255), 2)
    cv2.imshow(window, preview)

    key = cv2.waitKey(1) & 0xFF

    # Check if SPACEBAR is pressed (ASCII value 32)
    if key == ord(' '):
        image_count += 1
        timestamp = int(time.time() * 1000)
        filename = f"can_{timestamp}.jpg"

        cv2.imwrite(str(full_folder / filename), gray)
        if circle is not None:
            cv2.imwrite(str(subfolder / filename), crop_can(frame, circle))
            print(f"[{image_count}] Saved can crop + full frame -> {filename}")
        else:
            print(f"[{image_count}] No can detected - saved full frame only -> full/{filename}")

    # Check if 'q' is pressed to quit
    elif key == ord('q'):
        print("Closing capture tool...")
        break

cap.release()
cv2.destroyAllWindows()
