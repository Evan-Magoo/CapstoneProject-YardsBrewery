# This file is for training the YOLO26 Object Detection Model
# This model finds objects with their roi & classifys it
# All objects on screen

from ultralytics import YOLO

model = YOLO("runs/classify/hailo8/weights/best.pt")

results = model.export(
    format="hailo",
    name="hailo8",
    data="training_data"
)