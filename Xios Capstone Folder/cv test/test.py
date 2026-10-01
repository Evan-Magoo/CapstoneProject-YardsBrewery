import cv2
import numpy as np
import math
import time


# ============================================================
# CAN LIP SPECS
# ============================================================

OUTER_LIP_MM = 56.90

INNER_OPENING_MM = 52.40
INNER_TOLERANCE_MM = 0.25

INNER_MIN_MM = INNER_OPENING_MM - INNER_TOLERANCE_MM
INNER_MAX_MM = INNER_OPENING_MM + INNER_TOLERANCE_MM

FLANGE_NOMINAL_MM = 2.10
FLANGE_TOLERANCE_MM = 0.20

FLANGE_MIN_MM = FLANGE_NOMINAL_MM - FLANGE_TOLERANCE_MM
FLANGE_MAX_MM = FLANGE_NOMINAL_MM + FLANGE_TOLERANCE_MM

EXPECTED_RATIO = INNER_OPENING_MM / OUTER_LIP_MM


# ============================================================
# DEFECT SETTINGS
# ============================================================

CAMERA_INDEX = 0

MIN_AREA = 1800

# How close the detected inner/outer diameter ratio
# must be to the expected 52.4 / 56.9 ratio
MAX_RATIO_ERROR = 0.08

# Smooth the rings from frame to frame
SMOOTHING = 0.35

# Local dent detection
RIM_DEVIATION_LIMIT_MM = 0.65

# Number of strongly deviating contour points allowed
BAD_RIM_POINT_LIMIT = 20


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

time.sleep(2)

camera_ready = False

for attempt in range(30):

    ret, frame = cap.read()

    if ret and frame is not None:
        camera_ready = True
        break

    time.sleep(0.1)

if not camera_ready:
    print("ERROR: Camera could not provide frames.")
    cap.release()
    exit()

print("Camera started.")
print("LIP-ONLY inspection.")
print("Press Q to quit.")


# ============================================================
# SMOOTHING STORAGE
# ============================================================

smooth_outer = None
smooth_inner = None


# ============================================================
# SMOOTH ELLIPSE
# ============================================================

def smooth_ellipse(previous, current):

    if previous is None:
        return current

    old_center, old_axes, old_angle = previous
    new_center, new_axes, new_angle = current

    a = SMOOTHING

    center = (
        old_center[0] * (1 - a) + new_center[0] * a,
        old_center[1] * (1 - a) + new_center[1] * a
    )

    axes = (
        old_axes[0] * (1 - a) + new_axes[0] * a,
        old_axes[1] * (1 - a) + new_axes[1] * a
    )

    angle_difference = new_angle - old_angle

    if angle_difference > 90:
        new_angle -= 180

    elif angle_difference < -90:
        new_angle += 180

    angle = (
        old_angle * (1 - a)
        +
        new_angle * a
    )

    return (
        center,
        axes,
        angle
    )


# ============================================================
# LOCAL RIM DEVIATION
# ============================================================

def rim_deviation(
    contour,
    ellipse,
    pixels_per_mm
):

    center, axes, angle = ellipse

    cx, cy = center

    a = axes[0] / 2.0
    b = axes[1] / 2.0

    if a <= 0 or b <= 0 or pixels_per_mm <= 0:
        return 0, 0.0, []

    angle_rad = math.radians(angle)

    bad_points = []
    max_deviation_mm = 0.0

    for point in contour:

        x, y = point[0]

        dx = x - cx
        dy = y - cy

        xr = (
            dx * math.cos(angle_rad)
            +
            dy * math.sin(angle_rad)
        )

        yr = (
            -dx * math.sin(angle_rad)
            +
            dy * math.cos(angle_rad)
        )

        normalized_radius = math.sqrt(
            (xr * xr) / (a * a)
            +
            (yr * yr) / (b * b)
        )

        average_radius_px = (
            a + b
        ) / 2.0

        deviation_px = (
            abs(normalized_radius - 1.0)
            *
            average_radius_px
        )

        deviation_mm = (
            deviation_px
            /
            pixels_per_mm
        )

        max_deviation_mm = max(
            max_deviation_mm,
            deviation_mm
        )

        if deviation_mm > RIM_DEVIATION_LIMIT_MM:

            bad_points.append(
                (int(x), int(y))
            )

    return (
        len(bad_points),
        max_deviation_mm,
        bad_points
    )


# ============================================================
# FIND OUTER + INNER LIP
# ============================================================

def detect_lip(frame):

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        45,
        130
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    image_center = np.array(
        [
            width / 2,
            height / 2
        ]
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if len(contour) < 20:
            continue

        ellipse = cv2.fitEllipse(
            contour
        )

        center, axes, angle = ellipse

        major = max(axes)
        minor = min(axes)

        if minor <= 0:
            continue

        # Allow perspective tilt
        if major / minor > 1.8:
            continue

        distance_from_center = np.linalg.norm(
            np.array(center)
            -
            image_center
        )

        if distance_from_center > width * 0.32:
            continue

        diameter = (
            major + minor
        ) / 2.0

        candidates.append(
            {
                "contour": contour,
                "ellipse": ellipse,
                "center": np.array(center),
                "diameter": diameter
            }
        )

    if len(candidates) < 2:

        return (
            None,
            None,
            None,
            None,
            edges
        )


    # ========================================================
    # FIND PAIR MATCHING EXPECTED LIP RATIO
    # ========================================================

    best_pair = None
    best_score = float("inf")

    for i in range(len(candidates)):

        for j in range(
            i + 1,
            len(candidates)
        ):

            c1 = candidates[i]
            c2 = candidates[j]

            if c1["diameter"] > c2["diameter"]:

                outer = c1
                inner = c2

            else:

                outer = c2
                inner = c1

            ratio = (
                inner["diameter"]
                /
                outer["diameter"]
            )

            ratio_error = abs(
                ratio
                -
                EXPECTED_RATIO
            )

            if ratio_error > MAX_RATIO_ERROR:
                continue

            center_difference = np.linalg.norm(
                outer["center"]
                -
                inner["center"]
            )

            if (
                center_difference
                >
                outer["diameter"] * 0.07
            ):
                continue

            score = (
                ratio_error * 500
                +
                center_difference
            )

            if score < best_score:

                best_score = score

                best_pair = (
                    outer,
                    inner
                )


    if best_pair is None:

        return (
            None,
            None,
            None,
            None,
            edges
        )


    outer, inner = best_pair


    return (
        outer["contour"],
        outer["ellipse"],
        inner["contour"],
        inner["ellipse"],
        edges
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        continue


    frame = cv2.resize(
        frame,
        (800, 600)
    )

    display = frame.copy()


    (
        outer_contour,
        detected_outer,
        inner_contour,
        detected_inner,
        edges
    ) = detect_lip(frame)


    status = "NO LIP DETECTED"
    status_color = (255, 255, 255)


    # ========================================================
    # LIP DETECTED
    # ========================================================

    if (
        detected_outer is not None
        and
        detected_inner is not None
    ):

        # Smooth the two rings
        smooth_outer = smooth_ellipse(
            smooth_outer,
            detected_outer
        )

        smooth_inner = smooth_ellipse(
            smooth_inner,
            detected_inner
        )


        # ====================================================
        # SELF CALIBRATE EACH CAN
        # ====================================================

        outer_axes = smooth_outer[1]

        outer_effective_px = math.sqrt(
            outer_axes[0]
            *
            outer_axes[1]
        )

        pixels_per_mm = (
            outer_effective_px
            /
            OUTER_LIP_MM
        )


        # ====================================================
        # INNER OPENING SIZE
        # ====================================================

        inner_axes = smooth_inner[1]

        inner_effective_px = math.sqrt(
            inner_axes[0]
            *
            inner_axes[1]
        )

        inner_mm = (
            inner_effective_px
            /
            pixels_per_mm
        )


        inner_pass = (
            INNER_MIN_MM
            <= inner_mm
            <= INNER_MAX_MM
        )


        # ====================================================
        # FLANGE WIDTH
        # ====================================================

        flange_mm = (
            OUTER_LIP_MM
            -
            inner_mm
        ) / 2.0


        flange_pass = (
            FLANGE_MIN_MM
            <= flange_mm
            <= FLANGE_MAX_MM
        )


        # ====================================================
        # LOCAL RIM DAMAGE
        # ====================================================

        (
            bad_point_count,
            max_deviation_mm,
            bad_points
        ) = rim_deviation(
            inner_contour,
            detected_inner,
            pixels_per_mm
        )


        rim_shape_pass = (
            bad_point_count
            <= BAD_RIM_POINT_LIMIT
        )


        # ====================================================
        # FINAL LIP RESULT
        # ====================================================

        lip_good = (
            inner_pass
            and
            flange_pass
            and
            rim_shape_pass
        )


        if lip_good:

            status = "PASS - LIP GOOD"
            status_color = (0, 255, 0)

        else:

            status = "FAIL - LIP DAMAGE"
            status_color = (0, 0, 255)


        # ====================================================
        # DRAW OUTER + INNER RINGS
        # ====================================================

        cv2.ellipse(
            display,
            smooth_outer,
            (255, 0, 0),
            3
        )

        cv2.ellipse(
            display,
            smooth_inner,
            (0, 255, 255),
            3
        )


        # ====================================================
        # DRAW LOCAL DEFECT POINTS
        # ====================================================

        for point in bad_points:

            cv2.circle(
                display,
                point,
                4,
                (0, 0, 255),
                -1
            )


        # ====================================================
        # RESULT TEXT
        # ====================================================

        def test_color(passed):

            if passed:
                return (0, 255, 0)

            return (0, 0, 255)


        cv2.putText(
            display,
            f"Inner opening: {inner_mm:.2f} mm",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            test_color(inner_pass),
            2
        )


        cv2.putText(
            display,
            f"Flange width: {flange_mm:.2f} mm",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            test_color(flange_pass),
            2
        )


        cv2.putText(
            display,
            f"Rim shape: "
            f"{'GOOD' if rim_shape_pass else 'DAMAGED'}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            test_color(rim_shape_pass),
            2
        )


        cv2.putText(
            display,
            f"Max deformation: {max_deviation_mm:.2f} mm",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            test_color(rim_shape_pass),
            2
        )


        cv2.putText(
            display,
            f"Bad rim points: {bad_point_count}",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            test_color(rim_shape_pass),
            2
        )


    else:

        # Reset smoothing when the can disappears
        smooth_outer = None
        smooth_inner = None


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        display,
        status,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )


    cv2.imshow(
        "Can Lip Inspection",
        display
    )


    cv2.imshow(
        "Lip Edges",
        edges
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Inspection stopped.")