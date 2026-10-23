"""
Builds the training sets from training_data/ using the SAME can detection and
crop the live system uses, so training images match what the model sees.

  training_data/              (your raw images: train|val / good_cans|bad_cans)
      -> training_data_cropped/  classification set (used by yolo_training.py
                                 and yolo_training_cls.py)
      -> training_data_detect/   detection set with auto-made bounding boxes
                                 + data.yaml (used by yolo_training_obj.py)

Run it again whenever you add images:  python prepare_dataset.py
"""

import shutil
from pathlib import Path

import cv2

from can_vision import find_can, crop_can, to_gray

PROJECT_DIR = Path(__file__).resolve().parent
SRC = PROJECT_DIR / "training_data"
CLS_OUT = PROJECT_DIR / "training_data_cropped"
DET_OUT = PROJECT_DIR / "training_data_detect"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if not SRC.exists():
        raise SystemExit(f"Could not find {SRC}")

    classes = sorted({p.name for split in SRC.iterdir() if split.is_dir()
                      for p in split.iterdir() if p.is_dir()})
    print(f"Classes: {classes}")

    for out in (CLS_OUT, DET_OUT):
        if out.exists():
            shutil.rmtree(out)

    kept = skipped = 0
    for split_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        split = split_dir.name
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            (CLS_OUT / split / cls).mkdir(parents=True, exist_ok=True)
            (DET_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (DET_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [skip] unreadable: {img_path.name}")
                    skipped += 1
                    continue
                circle = find_can(img)
                if circle is None:
                    print(f"  [skip] no can found: {split}/{cls}/{img_path.name}")
                    skipped += 1
                    continue

                # Classification: can crop
                cv2.imwrite(str(CLS_OUT / split / cls / (img_path.stem + ".jpg")),
                            crop_can(img, circle))

                # Detection: grayscale frame + YOLO box label from the circle
                gray = to_gray(img)
                h, w = gray.shape[:2]
                x, y, r = circle
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(w, x + r), min(h, y + r)
                name = f"{cls}_{img_path.stem}"
                cv2.imwrite(str(DET_OUT / "images" / split / (name + ".jpg")), gray)
                with open(DET_OUT / "labels" / split / (name + ".txt"), "w") as f:
                    f.write(f"{classes.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
                kept += 1

    with open(DET_OUT / "data.yaml", "w") as f:
        f.write(f"path: {DET_OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"Done. {kept} images prepared, {skipped} skipped.")
    print(f"  Classification set: {CLS_OUT}")
    print(f"  Detection set:      {DET_OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
