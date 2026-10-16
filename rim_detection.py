from scipy.spatial import distance as dist
from imutils import perspective
from ultralytics import YOLO
import cv2
import os
import imutils
import numpy as np

# Helper function for determining the pixel per inch of an image
# It uses the pixel pitch & focal length which is a stat for a camera that
# will need to be researched to get correct results.
def find_pixels_per_inch(pixel_pitch, focal_length, distance_from_camera):
    distance_mm = distance_from_camera * 25.4
    pixels_per_mm = focal_length / (distance_mm * pixel_pitch)
    pixels_per_inch = pixels_per_mm * 25.4
    return pixels_per_inch

# Placeholder for fixing wrong calibration if needed down line
# The diameter measuring may be dropped in favor of solely using
# YOLO if performance testing shows it to not need dimension
# measurements.
def calibrate_reference():
    return None

# Loads the generated caning model
# Made using images of good & bad cans using yolo_training.py
model = YOLO("can_grading_model.pt")

# Camera Stats - MAYS PHASE THIS OUT
CAMERA_DISTANCE = 18.5
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028

# For finding how many pixels with in a recieved image are equal to an inch in that image
# Helps to try and work out perspective as measurements differ with distance from the camera
THEORETICAL_PPI = find_pixels_per_inch(PIXEL_PITCH, CAMERA_FOCAL_LENGTH, CAMERA_DISTANCE)
CALIBRATION_FACTOR = 2.13 / 2.72
PIXELS_PER_INCH = THEORETICAL_PPI / CALIBRATION_FACTOR

# Grading Stats
TARGET_HEIGHT = 4.83
TARGET_DIAMETER = 2.13
DIAMETER_TOLERANCE = 0.1
MIN_CIRCULARITY = 0.85

# Load all images from the test_images folder which is made by can_test.py
# These are images that were taken when that python file saw what it would assume to be a can.
images = []
if os.path.exists("test_images"):
    for filename in os.listdir("test_images"):
        img = cv2.imread(os.path.join("test_images",filename))
        if img is not None:
            images.append(img)

# Loop can analyze each image that was loaded
count = 0
for img in images:
    count += 1
    print(f"Processing Image: {count}")

    # Preprocess image for contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Get image stats for creating blank frame
    height, width, channels = 500, 500, 3

    # Find contours directly from the physical edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Create the black image to place found contours onto
    # This will help prevent outside noise from interfering.
    black_image = np.zeros_like(edged)
    inspection_status = "PASS - GOOD RIM"
    color = (0, 255, 0)

    # Get results found by model
    results = model(img)
    yolo_text = "YOLO: No Prediction"

    # If model found results, anaylze those results
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'probs') and result.probs is not None:
            top_class_idx = result.probs.top1
            top_class_name = result.names[top_class_idx]
            top_class_conf = float(result.probs.top1conf)
            yolo_text = f"YOLO: {top_class_name} ({top_class_conf*100:.1f}%)"
            model_ok = top_class_name == "good_can"
            # Check that can is a good can
            if not model_ok:
                inspection_status = "FAIL - BAD CLASSIFICATION"
                color = (0, 0, 255)

    # If contours are found, analyze them on the blank frame
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Make sure contour area is larger than 1000 to help prevent noise results
        if area > 1000:
            perimeter = cv2.arcLength(c, True)

            # Calculate the circularity of the contours.
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0

            # Find inner circle diameter if any
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            measured_diameter = (radius * 2) / PIXELS_PER_INCH

            # Verify that both the diamensions and circularity
            # are within the manufacturers specifications
            diameter_ok = abs(measured_diameter - TARGET_DIAMETER) <= DIAMETER_TOLERANCE
            circularity_ok = circularity >= MIN_CIRCULARITY

            # Check if the verifying functions were bad 
            # or not and display the approite text.
            if not diameter_ok:
                inspection_status = "FAIL - BAD DIAMETER"
                color = (0, 0, 255)
            elif not circularity_ok:
                inspection_status = "FAIL - DEFECTIVE/DENTED RIM"
                color = (0, 0, 255)

            # Draw contours onto the img map
            cv2.drawContours(img, [c], -1, color, 2)
            cv2.circle(img, (int(x), int(y)), int(radius), color, 2)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw contouirs onto the blank frame
            cv2.drawContours(black_image, [c], -1, (255, 255, 255), 2)

            # Test placing functions
            cv2.putText(img, f"Diameter: {measured_diameter:.2f}in", (int(x) - 40, int(y) - int(radius) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"Circularity: {circularity:.2f}in", (int(x) - 40, int(y) - int(radius) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img, inspection_status, (int(x) - 100, int(y) - int(radius) - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert grayscale image to color to 
    # be displayed with other unprocessed frame.
    edged_bgr = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)      

    # Add text labels to distinguish the views
    cv2.putText(img, "Original Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, yolo_text, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(edged_bgr, "Processed Inspection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine both frames horizontally side-by-side
    combined_img = np.hstack([img, edged_bgr])

    # Show side-by-side view
    cv2.imshow("Can Quality Inspection - Original vs Processed", combined_img)

    # Allow for each image to pause to press q
    # Removing this breaks it, making all images flash at once
    # This works for now
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Destorys windows after every image is processed
cv2.destroyAllWindows()