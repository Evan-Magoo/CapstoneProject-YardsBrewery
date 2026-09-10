import cv2
import time


# --------------------------------------------------
# 1. Camera setup
# --------------------------------------------------

CAMERA_INDEX = 0

print(f"Opening camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

time.sleep(1)

print("Camera opened successfully!")
print("Press q to quit.")


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


    # --------------------------------------------------
    # Blur image to reduce noise
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )


    # --------------------------------------------------
    # Detect circular shapes
    # --------------------------------------------------

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=40,
        minRadius=30,
        maxRadius=250
    )


    # --------------------------------------------------
    # Draw detected circles
    # --------------------------------------------------

    if circles is not None:

        circles = circles[0]

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

    cv2.imshow(
        "Can Defect Detection",
        frame
    )


    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 3. Clean up
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Camera closed.")