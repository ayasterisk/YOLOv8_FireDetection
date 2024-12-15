from ultralytics import YOLO

model = YOLO('D:/GitHub/YOLOv8_FireDetection/runs/detect/train/weights/best.pt')

result = model.predict(source='D:/GitHub/YOLOv8_FireDetection/fire_detection_test.mp4',
                       show=True, save=False)