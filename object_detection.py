
import cv2
from ultralytics import YOLO

print("================================")
print("IBVAP AI SURVEILLANCE SYSTEM")
print("================================")

# Load YOLO11 pretrained model
print("Loading AI model...")
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera not accessible.")
    exit()

print("Camera connected!")
print("AI detection started.")
print("Press Q to exit.")

# Person, car, motorcycle, bus, truck
target_classes = [0, 2, 3, 5, 7]

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read video.")
        break

    # AI object detection
    results = model.predict(
        frame,
        classes=target_classes,
        conf=0.35,
        verbose=False
    )

    # Draw detection boxes
    output = results[0].plot()

    # Display live detection
    cv2.imshow(
        "IBVAP - Intelligent Border Surveillance",
        output
    )

    # Print detected object information
    for detection in results[0].boxes:
        class_id = int(detection.cls[0])
        confidence = float(detection.conf[0])

        object_name = model.names[class_id]

        print(
            f"Detected: {object_name} | "
            f"Confidence: {confidence:.2f}"
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("IBVAP surveillance stopped.")
