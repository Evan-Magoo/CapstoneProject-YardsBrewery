from scipy.spatial import distance as dist
from imutils import perspective
import cv2
import os
import imutils
import numpy as np

def find_pixels_per_inch(pixel_pitch, focal_length, distance_from_camera):
    distance_mm = distance_from_camera * 25.4
    pixels_per_mm = focal_length / (distance_mm * pixel_pitch)
    pixels_per_inch = pixels_per_mm * 25.4
    return pixels_per_inch

def calibrate_reference():
    return None

focal_length = 0
distance_camera_to_surface = 0
pixel_pitch = 0

CAMERA_INDEX = 0

CAN_HEIGHT = 4.83
CAMERA_DISTANCE = 18.5
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028

THEORETICAL_PPI = find_pixels_per_inch(PIXEL_PITCH, CAMERA_FOCAL_LENGTH, CAMERA_DISTANCE)

CALIBRATION_FACTOR = 2.13 / 2.72
PIXELS_PER_INCH = THEORETICAL_PPI / CALIBRATION_FACTOR
TARGET_HEIGHT = 4.83
TARGET_DIAMETER = 2.13
DIAMETER_TOLERANCE = 0.1
MIN_CIRCULARITY = 0.85

images = []
if os.path.exists("test_images"):
    for filename in os.listdir("test_images"):
        img = cv2.imread(os.path.join("test_images",filename))
        if img is not None:
            images.append(img)

count = 0
for img in images:
    count += 1
    print(f"Processing Image: {count}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    height, width, channels = 500, 500, 3

    # Find contours directly from the physical edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Create the black image (uint8 is standard for 8-bit images)
    black_image = np.zeros_like(edged)
    inspection_status = "PASS - GOOD RIM"
    color = (0, 255, 0)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)
        if area > 1000:
            perimeter = cv2.arcLength(c, True)

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            measured_diameter = (radius * 2) / PIXELS_PER_INCH

            diameter_ok = abs(measured_diameter - TARGET_DIAMETER) <= DIAMETER_TOLERANCE
            circularity_ok = circularity >= MIN_CIRCULARITY

            if not diameter_ok:
                inspection_status = "FAIL - BAD DIAMETER"
                color = (0, 0, 255)
            elif not circularity_ok:
                inspection_status = "FAIL - DEFECTIVE/DENTED RIM"
                color = (0, 0, 255)

            cv2.drawContours(img, [c], -1, color, 2)
            cv2.circle(img, (int(x), int(y)), int(radius), color, 2)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.drawContours(black_image, [c], -1, (255, 255, 255), 2)

            cv2.putText(img, f"Diameter: {measured_diameter:.2f}in", (int(x) - 40, int(y) - int(radius) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"Circularity: {circularity:.2f}in", (int(x) - 40, int(y) - int(radius) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img, inspection_status, (int(x) - 100, int(y) - int(radius) - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    edged_bgr = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)      

    # Add text labels to distinguish the views
    cv2.putText(img, "Original Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(edged_bgr, "Processed Inspection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Combine both frames horizontally side-by-side
    combined_img = np.hstack([img, edged_bgr])

    # Show side-by-side view
    cv2.imshow("Can Quality Inspection - Original vs Processed", combined_img)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()