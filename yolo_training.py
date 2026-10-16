from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

results = model.train(
    data="training_data",
    epochs=50,
    imgsz=224,
    device='cpu'
)