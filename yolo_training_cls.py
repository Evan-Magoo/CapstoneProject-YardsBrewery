# This file is for training the YOLO26 Classification Model
# This model will read the entire image that is input and
# classifies it into one category.

from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

results = model.train(
    data="training_data",
    epochs=100,
    imgsz=640,
    batch=1,
    optimzer='AdamW',
    augment=True,
)