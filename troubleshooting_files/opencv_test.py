import cv2
import time
import numpy as np
import keyboard
from matplotlib import pyplot as plt


# --------------------------------------------------
# 1. Camera setup
# --------------------------------------------------

CAMERA_INDEX = 0

print(f"Opening camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX)

time.sleep(1)

# Set the MJPEG codec first (crucial for high FPS/high resolutions)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Frame width and height (resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set target frames per second (FPS)
cap.set(cv2.CAP_PROP_FPS, 200.0)

# Verify the applied settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")


if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("Camera opened successfully!")
print("Press q to quit.")

canny_min = 50
canny_max = 150

# --------------------------------------------------
# 2. Live camera + can top detection
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success or frame is None:
        print("Could not read frame.")
        break


    # --------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(frame, canny_min, canny_max)

    # --------------------------------------------------
    # Blur image to reduce noise
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    edges_blurred = cv2.Canny(blurred, canny_min, canny_max)


    # --------------------------------------------------
    # Detect circular shapes
    # --------------------------------------------------

    circles = cv2.HoughCircles(
        edges_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=100,
        minRadius=40,
        maxRadius=1000
    )


    # --------------------------------------------------
    # Draw detected circles
    # --------------------------------------------------

    if circles is not None:

        circles = np.round(circles[0,:]).astype("int")

        for circle in circles:

            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])


            # Draw detected circle
            cv2.circle(
                frame,
                (x, y),
                radius,
                (0, 255, 0),
                3
            )


            # Draw center
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )


            # Add text
            cv2.putText(
                frame,
                f"Can top detected - Radius: {radius}px",
                (x - 150, y - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------
    # Show camera
    # --------------------------------------------------

    # Text Info
    cv2.putText(edges, "Color - Edges", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(edges, f"Canny Max: {canny_max}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(edges, f"Canny Min: {canny_min}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(edges_blurred, "Blurred - Edges", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(edges_blurred, f"Canny Max: {canny_max}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(edges_blurred, f"Canny Min: {canny_min}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Color", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(blurred, "Blurred", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #cv2.imshow("Can Defect Detection", frame)
    #cv2.imshow('Blurred Image', edges)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_blurred_bgr = cv2.cvtColor(edges_blurred, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    combined_frame_top = cv2.hconcat([frame, edges_bgr])
    combined_frame_bottom = cv2.hconcat([blurred_bgr, edges_blurred_bgr])
    combined_frame = cv2.vconcat([combined_frame_top, combined_frame_bottom])
    
    cv2.imshow('Side by Side with Spacing', combined_frame)

    # Press q to quit
    key = cv2.waitKey(1) & 0xFF

    match key:
        case 27:
            break
        case 113:  # ord("q")
            break
        case 116:  # ord("t")
            canny_max += 10
        case 103:  # ord("g")
            canny_max -= 10
        case 121:  # ord("y")
            canny_min += 10
        case 104:  # ord("h")
            canny_min -= 10



# --------------------------------------------------
# 3. Clean up
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Camera closed.")