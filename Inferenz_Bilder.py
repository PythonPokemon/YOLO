from ultralytics import YOLO
import cv2

# Lade das trainierte Modell
model = YOLO("runs/detect/train3/weights/best.pt")

# Lade ein Bild und führe die Inferenz durch
image_path = "c:\\Users\\jakob.derzapf\\Downloads\\6.PNG"  # Pfad zum Testbild
results = model.predict(source=image_path, save=True, save_txt=True)

# Ergebnisse anzeigen
for result in results:
    print(result.boxes)  # Begrenzungsrahmen der Objekte
