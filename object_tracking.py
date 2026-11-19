
import cv2
from ultralytics import YOLO

print("IBVAP - OBJECT TRACKING")
print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera connection failed!")
    exit()

print("Camera connected!")
print("Object tracking started.")
print("Press Q to stop.")

target_classes = [0, 2, 3, 5, 7]

while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read camera frame.")
        break

    # YOLO tracking with ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=target_classes,
        conf=0.35,
        verbose=False
    )

    # Draw tracking boxes and IDs
    output = results[0].plot()

    cv2.imshow(
        "IBVAP - Intelligent Object Tracking",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("IBVAP tracking stopped.")
