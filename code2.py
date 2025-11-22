import cv2
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO('yolo11n.pt')

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Run YOLO on each frame
    results = model(frame, verbose=False)

    # Draw predictions on the frame
    plot_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO11 Live Detection", plot_frame)

    # Press 'e' to exit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release camera and close window OUTSIDE loop
camera.release()
cv2.destroyAllWindows()
