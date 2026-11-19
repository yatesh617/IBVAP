import time
import cv2
import numpy as np
from ultralytics import YOLO
from evidence_capture import save_evidence

print("IBVAP - INTRUSION DETECTION SYSTEM")

# Load YOLO11 model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera not accessible.")
    exit()

print("Camera connected!")
print("Restricted zone monitoring started.")
print("Press Q to exit.")

# Store IDs that have triggered alerts & cooldown tracker
alerted_ids = set()
last_saved_time = {}
SAVE_COOLDOWN = 10  # seconds before capturing the same ID again

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    height, width = frame.shape[:2]

    # Define restricted zone
    zone = np.array([
        [int(width * 0.55), int(height * 0.20)],
        [int(width * 0.95), int(height * 0.20)],
        [int(width * 0.95), int(height * 0.95)],
        [int(width * 0.55), int(height * 0.95)]
    ], dtype=np.int32)

    # Run YOLO tracking for people only
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.35,
        verbose=False
    )

    output = results[0].plot()

    # Draw restricted zone
    cv2.polylines(
        output,
        [zone],
        True,
        (0, 0, 255),
        3
    )

    cv2.putText(
        output,
        "RESTRICTED ZONE",
        (int(width * 0.57), int(height * 0.17)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    boxes = results[0].boxes

    # Check tracked people
    if boxes is not None and boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = (
            boxes.conf.cpu().tolist()
            if boxes.conf is not None
            else [None] * len(track_ids)
        )

        for track_id, box, conf in zip(track_ids, coordinates, confidences):

            x1, y1, x2, y2 = map(int, box)

            # Bottom-centre point of person
            center_x = (x1 + x2) // 2
            bottom_y = y2

            # Check whether person is inside zone
            inside = cv2.pointPolygonTest(
                zone,
                (float(center_x), float(bottom_y)),
                False
            )

            if inside >= 0:

                cv2.putText(
                    output,
                    "INTRUSION ALERT!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                if track_id not in alerted_ids:
                    print(f"INTRUSION ALERT: Person ID {track_id}")
                    alerted_ids.add(track_id)

                cv2.putText(
                    output,
                    f"INTRUDER ID: {track_id}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # Save evidence image + log with cooldown
                current_time = time.time()
                if track_id not in last_saved_time or (current_time - last_saved_time[track_id]) > SAVE_COOLDOWN:
                    conf_val = round(float(conf), 2) if conf is not None else None
                    saved_path = save_evidence(output, track_id, confidence=conf_val)
                    if saved_path:
                        print(f"Evidence saved: {saved_path}")
                    last_saved_time[track_id] = current_time

    # Display monitoring window
    cv2.imshow(
        "IBVAP - Restricted Zone Monitoring",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("IBVAP intrusion monitoring stopped.")