from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.pt')

# Train the model
results = model.train(data='coco8-pose.yaml', epochs=1000, imgsz=640, device=[0], batch=128)