#Python Code (With YOLO11 + Voice Alert with 0.7s Delay)
#Voice now repeats while the object is visible, but only every 0.7 seconds.

import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

# Load YOLO model
model = YOLO('yolo11n.pt')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

# Objects for voice alerts
alert_classes = ['person', 'dog', 'cat', 'car']

# Track last alert time for each class
last_alert_time = {cls: 0 for cls in alert_classes}
alert_delay = 0.7  # seconds between repeated alerts

# Function to speak in a separate thread
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, verbose=False)
    plot_frame = results[0].plot()

    current_time = time.time()

    # Voice alert for each detected object
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in alert_classes:
            # Only speak if delay has passed since last alert
            if current_time - last_alert_time[class_name] > alert_delay:
                speak(f"{class_name} detected")
                last_alert_time[class_name] = current_time

    # Show the frame
    cv2.imshow("YOLO Detection + Voice Alert", plot_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
