# Import the OpenCV library for image processing
import cv2

# Import NumPy for working with arrays and numerical calculations
import numpy as np

# ---------------------------------------------------------
# STEP 1: Load the image of the aluminum can
# ---------------------------------------------------------

# Read the image from the current directory.
# Change "can_top.jpg" to the name of your image file.
img = cv2.imread("can_top.jpg")

# Make sure the image was successfully loaded.
if img is None:
    print("Error: Could not open the image.")
    exit()

# ---------------------------------------------------------
# STEP 2: Convert the image to grayscale
# ---------------------------------------------------------

# OpenCV normally loads an image in BGR color format.
# Circle detection does not require color information,
# so we convert the image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------
# STEP 3: Reduce noise in the image
# ---------------------------------------------------------

# Apply a Gaussian blur to smooth the image.
# This helps remove small details and noise that could
# interfere with detecting the circular rim of the can.
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# ---------------------------------------------------------
# STEP 4: Detect circles
# ---------------------------------------------------------

# HoughCircles searches the image for circular objects.
# In this exercise, we are looking for the outside
# circular rim of the aluminum can.
circles = cv2.HoughCircles(
    gray,                       # Grayscale image
    cv2.HOUGH_GRADIENT,         # Circle detection method
    dp=1.2,                     # Resolution of circle search
    minDist=100,                # Minimum distance between circles
    param1=100,                 # Edge detection threshold
    param2=40,                  # Circle detection sensitivity
    minRadius=50,               # Smallest expected circle radius
    maxRadius=300               # Largest expected circle radius
)

# ---------------------------------------------------------
# STEP 5: Process the detected circles
# ---------------------------------------------------------

# Check whether OpenCV found at least one circle.
if circles is not None:

    # Convert the detected circle values to integers.
    circles = np.round(circles[0, :]).astype("int")

    # Loop through every detected circle.
    for x, y, radius in circles:

        # Calculate the diameter in pixels.
        # Diameter = 2 × radius
        diameter_pixels = radius * 2

        # Display the measured diameter.
        print("Diameter in pixels:", diameter_pixels)


        # -------------------------------------------------
        # STEP 6: Convert pixels to millimeters
        # -------------------------------------------------

        # Example calibration:
        # Suppose a known 50 mm object measured 400 pixels.
        #
        # pixels_per_mm = 400 / 50
        #               = 8 pixels per millimeter

        pixels_per_mm = 8.0

        # Convert the detected can diameter from pixels
        # into millimeters.
        diameter_mm = diameter_pixels / pixels_per_mm

        # Display the calculated diameter.
        print("Diameter in millimeters:", diameter_mm)

        # -------------------------------------------------
        # STEP 7: Draw the detected circle
        # -------------------------------------------------

        # Draw a green circle around the detected can rim.
        # (0, 255, 0) represents green in BGR format.
        # The final value, 2, is the line thickness.
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)

        # Draw a small red dot at the center of the circle.
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # -------------------------------------------------
        # STEP 8: Put the measurement on the image
        # -------------------------------------------------

        # Create text containing the measured diameter.
        measurement_text = f"Diameter: {diameter_mm:.2f} mm"

        # Display the measurement above the detected circle.
        cv2.putText(
            img,
            measurement_text,
            (x - radius, y - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

else:
    # This message appears if no circles were detected.
    print("No circles detected.")

# ---------------------------------------------------------
# STEP 9: Display the final image
# ---------------------------------------------------------

# Open a window containing the analyzed image.
cv2.imshow("Aluminum Can Measurement", img)

# Wait until the user presses a key.
cv2.waitKey(0)

# Close all OpenCV windows.
cv2.destroyAllWindows()