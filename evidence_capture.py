import csv
from datetime import datetime
from pathlib import Path
import cv2
from blockchain_ledger import add_block

BASE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = EVIDENCE_DIR / "incident_log.csv"


def save_evidence(frame, person_id, confidence=None, camera_id="CAM-01"):
    now = datetime.now()
    file_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    readable_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    image_name = f"intrusion_{person_id}_{file_ts}.jpg"
    image_path = EVIDENCE_DIR / image_name

    success = cv2.imwrite(str(image_path), frame)
    if not success:
        print("ERROR: Could not save evidence image.")
        return None

    # Anchor evidence SHA-256 hash onto the blockchain ledger
    block = add_block(
        image_path=image_path,
        person_id=person_id,
        camera_id=camera_id,
        timestamp=readable_ts,
        confidence=confidence,
    )

    # Append to incident_log.csv
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "person_id",
                "confidence",
                "camera_id",
                "image_filename",
                "evidence_hash",
                "block_index",
                "tx_id",
            ])
        writer.writerow([
            readable_ts,
            person_id,
            confidence if confidence is not None else "N/A",
            camera_id,
            image_name,
            block["evidence_hash"],
            block["index"],
            block["tx_id"],
        ])

    return str(image_path)