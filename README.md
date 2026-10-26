# Can Rim Inspection (`can_tester.py`)

## About the project

This is part of a capstone project for **Yards Brewery** (repository `CapstoneProject-YardsBrewery`). The goal is to automatically inspect the rims and interiors of beverage cans so that defective cans be caught before they move on down the line.

## Overview

Live inspection of can rims from a camera feed. The script checks how round the rim is with OpenCV, then classifies the can with a YOLO model (`can_grading_model.pt`). Results are shown in a live window with an inspection dashboard beside the camera feed.

## Requirements

- Python 3
- A camera (the script opens camera index `0`)
- The model file `can_grading_model.pt` in the folder you run the script from
- These Python libraries:

```bash
pip install ultralytics opencv-python numpy
```

## Running

```bash
python can_tester.py
```

Startup prints its progress:

```text
[INFO] Initializing system, please wait...
[INFO] Loading YOLO classification model...
[INFO] System ready in X.XX seconds. Starting feed...
Starting Live Can Inspection with UI. Press 'q' to quit.
```

A window titled **Live Can Inspection Dashboard** opens. Place a can rim inside the green box. Press **q** with the window focused to quit. If a frame can't be read from the camera, the script prints `Failed to load image` and stops.

## How it works

Each frame goes through two stages.

**1. Rim shape check (OpenCV).** A 900 x 900 px square in the center of the frame is the inspection region (the green box). The region is converted to grayscale, blurred (11 x 11 Gaussian), and run through Canny edge detection (thresholds 50 / 150). The largest outer contour is taken as the rim. If its area is over 500 px, the script computes its circularity:

```
circularity = 4 * pi * area / perimeter^2
```

A perfect circle scores 1.0, and a dented or out-of-round rim scores lower.

**2. Classification (YOLO).** If circularity is at least `MIN_CIRCULARITY` (0.85), the grayscale region is passed to the YOLO classification model. The top class decides the result:

| Outcome | Condition | Dashboard status |
| --- | --- | --- |
| Pass | Round rim and YOLO class is `good_cans` | `PASS - GOOD CAN` (green) |
| Fail | Round rim but YOLO picks any other class | `FAIL - BAD CLASS` (red) |
| Fail | Circularity below 0.85 | `FAIL - DEFECTIVE RIM` (red) |

A rim that fails the shape check is rejected without running YOLO. A circle (from `cv2.minEnclosingCircle`) is drawn around the detected rim in the result color.

**Cooldown.** After a can is counted, the script waits `cooldown_time` (2 s) before counting another, so a can sitting in view is not counted on every frame. The feed and overlay keep updating during the cooldown.

## Dashboard

The right-hand panel (380 px wide) shows:

- **Inspection status:** the latest result on a green, red, or gray card (gray means "Awaiting Can...").
- **Production metrics:** total inspected, passed, and failed counts.
- **YOLO AI classifier:** the last predicted class and its confidence in percent.

Counts reset each time the script starts. Nothing is saved to disk.

## Configuration

These values are set near the top of `can_tester.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `square_size` | `900` | Side of the square inspection region, in pixels |
| `cooldown_time` | `2` | Seconds between counted inspections |
| `MIN_CIRCULARITY` | `0.85` | Lowest circularity that counts as a good rim |
| `CAMERA_DISTANCE` | `18.5` | Camera to rim distance, in inches |
| `CAMERA_FOCAL_LENGTH` | `4` | Lens focal length, in mm |
| `PIXEL_PITCH` | `0.0028` | Sensor pixel size, in mm |
| `CALIBRATION_FACTOR` | `2.13 / 2.72` | Correction applied to the theoretical pixels per inch |
| `PIXELS_PER_INCH` | computed | Calibrated pixels per inch at the rim |

Camera settings applied at startup: camera index `0`, MJPG format, 60 FPS requested, and `CAP_PROP_AUTO_EXPOSURE` set to `0.25`. To use a different camera, change the index in `cv2.VideoCapture(0)`.

The "good" class name is hard-coded as `good_cans`. If the model uses different class names, update that string.

## Notes

- **Frame size.** The inspection region is 900 px, but the script does not set the camera resolution. If the camera's frames are smaller than 900 x 900, the region won't fit in the frame. Set the resolution with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` and `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ...)`, or lower `square_size`.
- **Unused values.** `PIXELS_PER_INCH` and `THEORETICAL_PPI` are computed but not used in the inspection yet.