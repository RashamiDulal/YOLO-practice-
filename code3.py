#Python Code (With YOLO11 + Voice Alert)
# for voice alert download pyttsx3 in cmd
#step to downloads:
#C:\Users\ACER\OneDrive\Desktop\yoloP1>C:.venv\Scripts\python.exe code3.py
#C:\Users\ACER\OneDrive\Desktop\yoloP1>C:\Users\ACER\OneDrive\Desktop\yoloP1\.venv\Scripts\python.exe code3.py

import cv2
from ultralytics import YOLO
import pyttsx3

# Load YOLO Model
model = YOLO('yolo11n.pt')

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speed
engine.setProperty('volume', 1.0)

# Set objects for voice alert
alert_classes = ['person', 'dog', 'cat', 'car']

# Keep track so it does not repeat too much
last_alert = ""

# Start Camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    plot_frame = results[0].plot()

    # Get detected classes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in alert_classes and class_name != last_alert:
            engine.say(f"{class_name} detected")
            engine.runAndWait()
            last_alert = class_name

    cv2.imshow("YOLO Detection + Voice Alert", plot_frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

camera.release()
cv2.destroyAllWindows()
