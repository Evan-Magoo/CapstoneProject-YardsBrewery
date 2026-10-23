# This file is for training the YOLO26 Classification Model
# This model will read the entire image that is input and
# classifies it into one category.
#
# Run prepare_dataset.py first: it crops every training image around the can
# the same way can_tester.py does live.

from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

model = YOLO(str(PROJECT_DIR / "yolo26n-cls.pt"))

results = model.train(
    data=str(PROJECT_DIR / "training_data_cropped"),
    epochs=100,
    imgsz=640,
    batch=8,               # batch=1 is very slow and unstable; lower this if you run out of memory
    optimizer='AdamW',     # was misspelled "optimzer", which Ultralytics rejects
    lr0=0.001,             # AdamW is unstable at Ultralytics' default lr0=0.01
    augment=True,
    project=str(PROJECT_DIR / "runs"),
    name="can_cls_640",
)
