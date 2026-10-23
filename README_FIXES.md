# Fixed_CapstoneProject_YardsBrewery: what changed

## Why the camera wasn't seeing the can
Neither script told the camera what resolution to use, so OpenCV used its
default of 640x480. The code then cut a 900x900 square out of the middle:

    x1 = (640 - 900) // 2 = -130     y1 = (480 - 900) // 2 = -210

Python treats negative slice indexes as "count from the end", so
`frame[-210:690, -130:770]` was actually a small patch in the **bottom-right
corner**. A can in the center of the belt was never in the area being checked.
The green box was drawn off-screen too, which is why it never showed up.

## Changes
| File | Change |
|---|---|
| `can_vision.py` (new) | Shared camera + detection code. Opens the camera at **1920x1080 MJPG** (DirectShow on Windows), sets manual exposure, prints the real resolution, finds the can with a Hough circle search, and makes a can-centered crop. |
| `can_tester.py` | Uses the shared code. Inspects each can **once** when it enters the green zone in the middle of the frame, then waits for it to leave. Sends the classifier a 3-channel can crop (it got a 1-channel image before). Shows camera status on the dashboard and logs each inspection. The display is scaled to half size so 1080p fits on screen. |
| `get_test_images.py` | Same camera setup. The live preview outlines the detected can. SPACE saves the can crop (what the model sees) to `test_images/` and the full frame to `test_images/full/`. |
| `prepare_dataset.py` (new) | Crops every image in `training_data/` the same way the live system does → `training_data_cropped/`. Also builds a labeled detection set → `training_data_detect/data.yaml`. |
| `yolo_training.py` | Trains on `training_data_cropped`, `lr0=0.001` (AdamW was unstable at the default 0.01), copies `best.pt` to `can_grading_model.pt` automatically. |
| `yolo_training_cls.py` | Fixed the `optimzer` typo, uses the cropped data, batch 8 instead of 1. |
| `yolo_training_obj.py` | Fixed the typo. Points at `training_data_detect/data.yaml`, because a detection model needs box labels and folders alone don't have them. |
| `can_grading_model.pt` | Retrained on the cropped images with the fixed `yolo_training.py`: 100% train (76/76) and 100% val (18/18). The old model scored 73/76 and 17/18 on the same crops; it's kept as `can_grading_model_OLD_backup.pt`. |

The old rule "contour not circular → FAIL - DEFECTIVE RIM" was removed. It
was failing good cans because Canny edges are broken lines. When I measured
rim roundness properly, good and bad cans in the training set both scored
about 0.99, so it can't tell them apart. The classifier, trained on
`bad_cans`, now makes the pass/fail call.

## Workflow
1. `python get_test_images.py`: capture cans, then sort the crops into
   `training_data/train|val/good_cans|bad_cans`.
2. `python prepare_dataset.py`
3. `python yolo_training.py` (updates `can_grading_model.pt`)
4. `python can_tester.py`

## Settings to check on your hardware (top of `can_vision.py`)
- `EXPOSURE_VALUE`: -4 on Windows. If the image is too dark or too bright,
  try -3 to -7. Set `MANUAL_EXPOSURE = False` to use auto exposure.
- If the console says the camera didn't accept 1920x1080, set
  `FRAME_WIDTH/FRAME_HEIGHT` to a resolution your camera supports.
- `CAMERA_INDEX`: try 1 if a laptop webcam opens instead of the USB camera.
- `CENTER_TOLERANCE` in `can_tester.py`: width of the inspection zone.

## Tested
- The new detector found the can in all 94 training images (3 setups, 3 resolutions).
- A simulated 1080p conveyor feed (good, bad, good) through `can_tester.py`
  inspected each can exactly once and classified all three correctly.
- `yolo_training.py` ran end-to-end. `yolo_training_obj.py` was smoke-tested for 1 epoch.
- **Not tested on your physical camera.** On first run, check the console
  line `Camera resolution: 1920x1080` and adjust `EXPOSURE_VALUE` if needed.

Note: 18 validation images is a small set. Capture more cans on the real
conveyor with `get_test_images.py` and retrain before trusting the accuracy numbers.
