# train.py: Füge den Code für das Training ein.

from ultralytics import YOLO

# Lade das YOLOv8-Modell (z.B. 'yolov8n' für das kleine Modell)
model = YOLO("yolov8n.pt")  # Du kannst auch 'yolov8s.pt' oder 'yolov8m.pt' verwenden

# Starte das Training | neue Parameter
model.train(data="C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\UI-Control-C#-1\\config\\data.yaml", 
            epochs=120,               # Erhöhte Epochenanzahl für längeres Training
            batch=32,                 # Mittelgroße Batch-Größe für GPU-Speicher und Stabilität
            imgsz=640,                # Standardbildgröße bleibt 640x640
            lr0=0.003,                # Reduzierte Start-Lernrate für stabilere Konvergenz
            optimizer="AdamW",        # Spezifischer AdamW-Optimierer (gut für größere Datenmengen)
            warmup_epochs=5,          # Verlängerte Warm-up-Phase für besseren Start
            patience=20,              # Geduld erhöhen, um mehr Zeit für Verbesserungen zu lassen | wohl eher wenn es keine verbesserungen mehr gibt!
            save=True,                # Modelle nach jeder Epoche speichern
            save_period=2,            # Speicherintervall auf jede zweite Epoche setzen
            workers=8,                # Maximale Parallelisierung der Datenaufbereitung
            verbose=True,             # Detaillierte Trainingsinformationen anzeigen
            augment=True,             # Datenaugmentation aktivieren
            mosaic=1.0,               # Mosaic-Augmentierung für vielfältigere Trainingsdaten
            mixup=0.2,                # MixUp-Datenaugmentation aktivieren
            perspective=0.0,          # Perspektivische Verzerrung deaktivieren (UI-spezifisch)
            hsv_h=0.01,               # Farbtonänderung minimieren (relevanter für UI-Bilder)
            hsv_s=0.7,                # Sättigungsänderung für Varianz
            hsv_v=0.4,                 # Helligkeitsänderung für unterschiedliche Lichtbedingungen
            scale=0.5,                # Skaliere kleine Objekte stärker
            )




