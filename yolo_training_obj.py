# This file is for training the YOLO26 Object Detection Model
# This model finds objects with their roi & classifys it
# All objects on screen

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="training_data",
    epochs=100,
    imgsz=640,
    batch=1,
    optimzer='AdamW',
    augment=True,
)