# train.py: Füge den Code für das Training ein.

from ultralytics import YOLO

# Lade das YOLOv8-Modell (z.B. 'yolov8n' für das kleine Modell)
model = YOLO("yolov8n.pt")  # Du kannst auch 'yolov8s.pt' oder 'yolov8m.pt' verwenden

# Starte das Training
model.train(data="C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\UI-Control-C#-1\\config\\data.yaml", epochs=50, batch=16, imgsz=640)


