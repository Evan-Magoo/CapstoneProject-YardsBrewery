# This file is for training the YOLO26 Object Detection Model
# This model finds objects with their roi & classifys it
# All objects on screen
#
# A detection model needs bounding-box labels, not just class folders.
# prepare_dataset.py makes them automatically from the detected can circle
# and writes training_data_detect/data.yaml - run it first.

from pathlib import Path

import os
from ultralytics import YOLO

EPOCHS = int(os.environ.get("EPOCHS", 100))

PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "training_data_detect" / "data.yaml"
if not DATA_YAML.exists():
    raise SystemExit("training_data_detect/data.yaml not found - run prepare_dataset.py first")

# Point data.yaml at this folder, in case the project was moved/copied to another PC
lines = DATA_YAML.read_text().splitlines()
lines = [f"path: {DATA_YAML.parent.as_posix()}" if l.startswith("path:") else l for l in lines]
DATA_YAML.write_text("\n".join(lines) + "\n")

model = YOLO(str(PROJECT_DIR / "yolo26n.pt"))

results = model.train(
    data=str(DATA_YAML),  # was "training_data" (no labels)
    epochs=EPOCHS,
    imgsz=640,
    batch=8,
    optimizer='AdamW',     # was misspelled "optimzer"
    lr0=0.001,             # AdamW is unstable at Ultralytics' default lr0=0.01
    augment=True,
    project=str(PROJECT_DIR / "runs"),
    name="can_detect",
)
