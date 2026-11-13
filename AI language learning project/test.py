from ultralytics import YOLO

# model load
model = YOLO("yolov8n.pt")

# image test
results = model("https://ultralytics.com/images/bus.jpg")

# show result
results[0].show()