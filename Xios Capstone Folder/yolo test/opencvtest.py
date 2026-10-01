import cv2
import numpy as np
import time

# ---------------------------------
# CAMERA SETTINGS
# ---------------------------------
CAMERA_INDEX = 0

# Open Mac camera
cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

# Give camera time to initialize
time.sleep(1)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("Press Q to quit.")

# ---------------------------------
# MAIN LOOP
# ---------------------------------
while True:

    success, frame = cap.read()

    if not success or frame is None:
        print("ERROR: Could not read frame.")
        break

    # Resize camera image
    frame = cv2.resize(frame, (640, 480))

    # ---------------------------------
    # IMAGE PROCESSING
    # ---------------------------------

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Blur image to reduce noise
    blur = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    # Detect edges
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Default result
    result = "NO CAN DETECTED"
    result_color = (255, 255, 255)

    best_contour = None
    largest_area = 0

    # ---------------------------------
    # FIND CAN LIP
    # ---------------------------------
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small objects
        if area > 3000:

            if area > largest_area:
                largest_area = area
                best_contour = contour

    # ---------------------------------
    # CHECK SHAPE
    # ---------------------------------
    if best_contour is not None:

        perimeter = cv2.arcLength(
            best_contour,
            True
        )

        if perimeter > 0:

            circularity = (
                4 * np.pi * largest_area
            ) / (perimeter * perimeter)

            print(
                f"Circularity: {circularity:.3f}"
            )

            # Draw can outline
            cv2.drawContours(
                frame,
                [best_contour],
                -1,
                (255, 0, 0),
                3
            )

            # ---------------------------------
            # PASS / FAIL
            # ---------------------------------

            # Adjust this value after testing
            if circularity >= 0.80:

                result = "PASS"
                result_color = (0, 255, 0)

            else:

                result = "FAIL - LIP DEFECT"
                result_color = (0, 0, 255)

            # Show circularity
            cv2.putText(
                frame,
                f"Circularity: {circularity:.2f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

    # ---------------------------------
    # DISPLAY RESULT
    # ---------------------------------
    cv2.putText(
        frame,
        result,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        result_color,
        3
    )

    # Show camera
    cv2.imshow(
        "Can Lip Defect Detection",
        frame
    )

    # Show edge detection
    cv2.imshow(
        "Edges",
        edges
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
cv2.destroyAllWindows()

print("Camera closed.")
