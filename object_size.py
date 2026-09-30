# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import time
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def find_pixels_per_inch(pixel_pitch, focal_length, distance_from_camera):
    distance_mm = distance_from_camera * 25.4
    pixels_per_mm = focal_length / (distance_mm * pixel_pitch)
    pixels_per_inch = pixels_per_mm * 25.4
    return pixels_per_inch

focal_length = 0
distance_camera_to_surface = 0
pixel_pitch = 0

CAMERA_INDEX = 0

CAMERA_DISTANCE = 18.45 - 4.83
CAMERA_FOCAL_LENGTH = 4
PIXEL_PITCH = 0.0028

PIXELS_PER_INCH = find_pixels_per_inch(PIXEL_PITCH, CAMERA_FOCAL_LENGTH, CAMERA_DISTANCE)

TARGET_HEIGHT = 4.83
TARGET_DIAMETER = 2.13
TOLERANCE = 0.2
# ---------------------------------

print(f"Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

time.sleep(1)
print("Camera opened successfully!")
print("Press q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to get frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 1000:
                continue

            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            measured_dimA = dA / PIXELS_PER_INCH
            measured_dimB = dB / PIXELS_PER_INCH

            #can_height = max(measured_dimA, measured_dimB)
            can_diameter = min(measured_dimA, measured_dimB)

            #height_ok = abs(can_height - TARGET_HEIGHT) <= TOLERANCE
            diameter_ok = abs(can_diameter - TARGET_DIAMETER) <= TOLERANCE

            if diameter_ok:
                status = "PASS - GOOD TO FILL"
                color = (0, 255, 0)
            else:
                status = "FAIL - REJECT"
                color = (0, 0, 255)

            cv2.drawContours(frame, [box.astype("int")], -1, color, 2)

            #text_height = f"H: {can_height:.2f}in"
            text_diameter = f"D: {can_diameter:.2f}in"

            #cv2.putText(frame, text_height, (int(tltrX - 20), int(tltrY - 10)), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, text_diameter, (int(trbrX + 10), int(trbrY)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display overall pass/fail status box at the top of the can
            cv2.putText(frame, status, (int(tl[0]), int(tl[1] - 30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    edged_bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    # Add text labels to distinguish the views
    cv2.putText(frame, "Original Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(edged_bgr, "Processed Inspection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Combine both frames horizontally side-by-side
    combined_frame = np.hstack([frame, edged_bgr])

    # Show side-by-side view
    cv2.imshow("Can Quality Inspection - Original vs Processed", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()