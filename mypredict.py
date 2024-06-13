from ultralytics import YOLO


# Load a pretrained YOLOv8n model

model = YOLO('runs/detect/Improved YOLOv8s/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('bus.jpg', save=True, imgsz=640, conf=0.5)


