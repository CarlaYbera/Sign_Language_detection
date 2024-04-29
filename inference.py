from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('best.pt')

results = model.predict (source="0",save=True, imgsz = 640, conf=0.5, show=True,stream=True)
for r in results:
    boxes=r.boxes
    print (boxes.xyxy) 