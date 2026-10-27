import cv2
import time
from pathlib import Path

square_size = 900

# Initialize camera (using MJPEG encoding for ELP stability)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 200)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Turn OFF auto exposure (0.25 = Manual Mode)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# Now you can manually set your exposure value 
# Note: Values vary by camera (e.g., -4, -6, or absolute numbers like 100)
cap.set(cv2.CAP_PROP_EXPOSURE, -4) 

# Wait for camera to load
time.sleep(1)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Setup test_images directory
PROJECT_DIR = Path(__file__).resolve().parent
subfolder = PROJECT_DIR / "test_images"
subfolder.mkdir(parents=True, exist_ok=True)

image_count = 0
print("--- CAMERA CAPTURE ACTIVE ---")
print("Press [SPACEBAR] to take a picture.")
print("Press [q] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break

    height, width, _ = frame.shape
    
    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Show live feed
    cv2.imshow("Capture Tool - [SPACE] to Save | [q] to Quit", gray_roi)

    key = cv2.waitKey(1) & 0xFF

    # Check if SPACEBAR is pressed (ASCII value 32)
    if key == ord(' '):
        image_count += 1
        timestamp = int(time.time())
        filename = f"can_{timestamp}.jpg"
        filepath = subfolder / filename
        
        # Save frame to test_images folder
        cv2.imwrite(str(filepath), gray_roi)
        print(f"[{image_count}] Saved snapshot -> {filename}")

    # Check if 'q' is pressed to quit
    elif key == ord('q'):
        print("Closing capture tool...")
        break

cap.release()
cv2.destroyAllWindows()