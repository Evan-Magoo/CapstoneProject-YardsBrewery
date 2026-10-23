# Trains can_grading_model.pt (the model can_tester.py uses).
# Run prepare_dataset.py first so the training images are cropped the same
# way the live camera crops them.

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

model = YOLO(str(PROJECT_DIR / "yolo26n-cls.pt"))

results = model.train(
    data=str(PROJECT_DIR / "training_data_cropped"),
    epochs=50,
    imgsz=224,
    optimizer='AdamW',
    lr0=0.001,            # AdamW is unstable at Ultralytics' default lr0=0.01
    device='cpu',
    project=str(PROJECT_DIR / "runs"),
    name="can_grading",
    exist_ok=True,
)

# Copy the best weights to the file can_tester.py loads
best = Path(results.save_dir) / "weights" / "best.pt"
shutil.copy(best, PROJECT_DIR / "can_grading_model.pt")
print(f"Saved new model -> {PROJECT_DIR / 'can_grading_model.pt'}")
