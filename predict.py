from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

result = model.predict(source='fire_detection_test.mp4',
                       show=True, save=False)