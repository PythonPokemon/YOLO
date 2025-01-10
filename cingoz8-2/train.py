# train.py: Füge den Code für das Training ein.

from ultralytics import YOLO

# Lade das YOLOv8-Modell (z.B. 'yolov8n' für das kleine Modell)
model = YOLO("yolov8n.pt")  # Du kannst auch 'yolov8s.pt' oder 'yolov8m.pt' verwenden

# Starte das Training | neue Parameter
model.train( data=r"C:\Users\jakob.derzapf\source\repos\PythonProjekte\YOLO\UI-Control-C#-1\config\data.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
    lr0=0.005,
    optimizer="AdamW",
    warmup_epochs=5,
    patience=20,
    save=True,
    save_period=2,
    workers=8,
    verbose=True,
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    perspective=0.0,
    hsv_h=0.01,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    degrees=15.0,
    translate=0.1,
    shear=5.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    copy_paste=0.0,
    copy_paste_mode="flip",
    auto_augment="randaugment",
    erasing=0.4,
    crop_fraction=1.0,
)