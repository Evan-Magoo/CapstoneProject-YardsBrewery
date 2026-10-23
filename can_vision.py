"""
Shared camera + can-detection code used by can_tester.py, get_test_images.py
and prepare_dataset.py, so the live system, the image-capture tool and the
training data all see the can exactly the same way.

FIXES vs. the original scripts
  1. The camera resolution is now requested explicitly (1920x1080). Before,
     OpenCV fell back to 640x480, the 900x900 "center square" went negative,
     and Python's negative slicing silently cropped the BOTTOM-RIGHT corner of
     the frame instead of the center - so a can under the camera was never seen.
  2. Can detection uses a Hough circle search on the whole frame instead of
     "largest Canny contour + circularity". Canny edges are usually broken
     lines, which almost never scored as circular, so real cans were missed or
     reported as "defective rim".
  3. The model input is a square crop centered on the detected can, converted
     to grayscale and back to 3 channels. The same crop is used for training
     (prepare_dataset.py) so the model sees the same kind of image it was
     trained on.
"""

import platform
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Camera settings - change these to match your setup
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAMERA_FPS = 30          # ELP cameras usually run 1080p MJPG at 30 fps
MANUAL_EXPOSURE = True   # set False to let the camera auto-expose
EXPOSURE_VALUE = -4      # Windows/DirectShow scale (-1 bright ... -13 dark).
                         # On Linux (V4L2) use an absolute value, e.g. 150.

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
DETECT_WIDTH = 480            # frame is shrunk to this short side for speed
MIN_RADIUS_FRAC = 0.18        # can radius limits, as a fraction of the
MAX_RADIUS_FRAC = 0.48        # frame's short side
HOUGH_EDGE_THRESHOLD = 120    # param1: Canny high threshold inside Hough
HOUGH_VOTE_THRESHOLD = 40     # param2: lower = more sensitive (more false hits)
CROP_MARGIN = 1.10            # crop = circle diameter * this


def open_camera(index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                fps=CAMERA_FPS, manual_exposure=MANUAL_EXPOSURE,
                exposure=EXPOSURE_VALUE):
    """Open the USB camera at a known resolution and return it.

    Raises RuntimeError if the camera cannot be opened.
    """
    is_windows = platform.system() == "Windows"

    # DirectShow is much faster to open and supports MJPG 1080p on Windows.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(index)
    if is_windows and not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}. "
                           "Check the USB cable or try CAMERA_INDEX = 1.")

    # Order matters: codec first, then resolution, then fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if manual_exposure:
        # "Manual" is 0.25 on DirectShow/MSMF and 1 on Linux V4L2.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if is_windows else 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if is_windows else 3)

    # Grab one frame so we report the resolution the camera actually gave us.
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but returned no frame.")
    actual_h, actual_w = frame.shape[:2]
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h} "
          f"(requested {width}x{height})")
    if (actual_w, actual_h) != (width, height):
        print("[WARN] Camera did not accept the requested resolution. "
              "Detection still works, but check FRAME_WIDTH/FRAME_HEIGHT.")
    return cap


def to_gray(image):
    """Return a single-channel grayscale copy of any BGR or gray image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_can(image):
    """Find the can rim in a frame.

    Returns (x, y, r) in full-frame pixel coordinates, or None if no can.
    """
    gray = to_gray(image)
    h, w = gray.shape[:2]
    scale = DETECT_WIDTH / float(min(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    short_side = min(small.shape[:2])

    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=short_side,
        param1=HOUGH_EDGE_THRESHOLD, param2=HOUGH_VOTE_THRESHOLD,
        minRadius=int(short_side * MIN_RADIUS_FRAC),
        maxRadius=int(short_side * MAX_RADIUS_FRAC),
    )
    if circles is None:
        return None
    x, y, r = circles[0][0]
    return float(x / scale), float(y / scale), float(r / scale)


def crop_can(image, circle, margin=CROP_MARGIN):
    """Square crop centered on the can, padded with black if it hits an edge.

    Returns a 3-channel grayscale image (what the classifier expects).
    """
    gray = to_gray(image)
    x, y, r = circle
    half = int(round(r * margin))
    cx, cy = int(round(x)), int(round(y))
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    crop = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)


def can_fully_in_frame(image, circle):
    """True if the whole can rim is inside the frame."""
    h, w = image.shape[:2]
    x, y, r = circle
    return r <= x <= w - r and r <= y <= h - r
