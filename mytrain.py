from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model


# 跑 yolov5 模型
# Create a new YOLO model from scratch
model = YOLO('ultralytics/cfg/models/v8/yolov8s_GAM_C2fDCN_WIOU_Dyhead.yaml')
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(data='VisDrone.yaml')





# #跑 rt-detr 模型
# # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR('rtdetr-l.pt')
#
# # Display model information (optional)
# model.info()
#
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
#
# # Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model('path/to/bus.jpg')





