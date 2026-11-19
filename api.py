import json
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from blockchain_ledger import get_ledger, verify_evidence
from evidence_capture import save_evidence

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "cameras.json"
EVIDENCE_DIR = BASE_DIR / "evidence"
AUDIT_FILE = EVIDENCE_DIR / "security_audit.json"
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
MODEL_PATH = BASE_DIR / "yolo11n.pt"

app = FastAPI(title="IBVAP Border Surveillance API")

# Load YOLO model once
model = YOLO(str(MODEL_PATH))

# Role-Based Access Control (Cybersecurity Demo Accounts)
USERS = {
    "commander": {"password": "mha@2026", "role": "Commander (Admin)"},
    "operator": {"password": "watch@2026", "role": "Surveillance Operator"},
}
ACTIVE_TOKENS = {}

# Shared state
state_lock = threading.Lock()
night_mode = False
active_camera_id = "cam_01"
latest_alert = None
last_saved_time = {}
SAVE_COOLDOWN = 8  # seconds


def load_camera_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {
        "active_camera": "cam_01",
        "cameras": {
            "cam_01": {
                "name": "Sector-01 Live Webcam",
                "source": 0,
                "restricted_zone_ratio": [[0.55, 0.20], [0.95, 0.20], [0.95, 0.95], [0.55, 0.95]],
            }
        },
    }


cfg = load_camera_config()
active_camera_id = cfg.get("active_camera", "cam_01")


def log_audit(user: str, role: str, action: str):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    if AUDIT_FILE.exists():
        try:
            logs = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        except Exception:
            logs = []
    logs.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "role": role,
        "action": action,
    })
    AUDIT_FILE.write_text(json.dumps(logs[:50], indent=2), encoding="utf-8")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def serve_dashboard():
    return FileResponse(str(FRONTEND_FILE))


@app.post("/api/login")
def login(req: LoginRequest):
    user_info = USERS.get(req.username.strip().lower())
    if not user_info or user_info["password"] != req.password:
        log_audit(req.username, "UNKNOWN", "FAILED LOGIN ATTEMPT")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_hex(16)
    ACTIVE_TOKENS[token] = {"username": req.username.lower(), "role": user_info["role"]}
    log_audit(req.username.lower(), user_info["role"], "Logged into Command Center")
    return {"token": token, "username": req.username.lower(), "role": user_info["role"]}


@app.get("/api/status")
def get_status():
    config_data = load_camera_config()
    ledger = get_ledger()
    incidents = [b for b in ledger if b["index"] > 0]
    return {
        "active_camera": active_camera_id,
        "cameras": config_data["cameras"],
        "night_mode": night_mode,
        "latest_alert": latest_alert,
        "total_incidents": len(incidents),
        "total_blocks": len(ledger),
    }


@app.get("/api/incidents")
def get_incidents():
    ledger = get_ledger()
    incidents = [b for b in reversed(ledger) if b["index"] > 0]
    return {"incidents": incidents}


@app.get("/api/blockchain")
def get_blockchain():
    return {"chain": list(reversed(get_ledger()))}


@app.get("/api/audit_log")
def get_audit_log():
    if AUDIT_FILE.exists():
        try:
            return {"logs": json.loads(AUDIT_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return {"logs": []}


@app.get("/api/evidence/{filename}")
def get_evidence_image(filename: str):
    safe_path = (EVIDENCE_DIR / Path(filename).name).resolve()
    if not safe_path.exists() or safe_path.parent != EVIDENCE_DIR.resolve():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return FileResponse(str(safe_path))


@app.post("/api/verify/{filename}")
def verify_evidence_api(filename: str):
    clean_name = Path(filename).name
    result = verify_evidence(clean_name)
    log_audit("system", "Blockchain Verifier", f"Integrity check on {clean_name}: {result['status']}")
    return result


@app.post("/api/switch_camera/{camera_id}")
def switch_camera(camera_id: str):
    global active_camera_id
    config_data = load_camera_config()
    if camera_id not in config_data["cameras"]:
        raise HTTPException(status_code=404, detail="Camera ID not found")
    with state_lock:
        active_camera_id = camera_id
    log_audit("operator", "Session", f"Switched surveillance feed to {camera_id}")
    return {"active_camera": active_camera_id}


@app.post("/api/toggle_night_mode")
def toggle_night_mode():
    global night_mode
    with state_lock:
        night_mode = not night_mode
        current = night_mode
    log_audit("operator", "Session", f"Night Vision Enhancement {'ENABLED' if current else 'DISABLED'}")
    return {"night_mode": current}


def _make_placeholder_frame(message: str):
    img = np.zeros((480, 720, 3), dtype=np.uint8)
    cv2.putText(img, "IBVAP SURVEILLANCE FEED", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2)
    cv2.putText(img, message, (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def _apply_night_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    cv2.putText(enhanced, "NIGHT VISION ENHANCED", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)
    return enhanced


def _open_capture(cam_info: dict):
    src = cam_info["source"]
    if isinstance(src, int):
        return cv2.VideoCapture(src, cv2.CAP_DSHOW), True
    src_str = str(src)
    if src_str.startswith("rtsp://") or src_str.startswith("http://") or src_str.startswith("https://"):
        return cv2.VideoCapture(src_str), False
    video_path = BASE_DIR / src_str
    if not video_path.exists():
        return None, False
    return cv2.VideoCapture(str(video_path)), False


def generate_frames():
    global latest_alert
    loaded_cam_id = None
    cap = None
    is_webcam = True

    try:
        while True:
            config_data = load_camera_config()
            with state_lock:
                target_cam_id = active_camera_id
                use_night = night_mode

            cam_info = config_data["cameras"].get(target_cam_id)

            # Re-open capture if camera changed
            if target_cam_id != loaded_cam_id:
                if cap is not None:
                    cap.release()
                cap, is_webcam = _open_capture(cam_info)
                loaded_cam_id = target_cam_id

            if cap is None or not cap.isOpened():
                placeholder = _make_placeholder_frame(f"Source unavailable: {cam_info['source']}")
                _, buffer = cv2.imencode(".jpg", placeholder)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                time.sleep(0.5)
                continue

            success, frame = cap.read()
            if not success:
                if not is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue

            if use_night:
                frame = _apply_night_vision(frame)

            height, width = frame.shape[:2]
            ratios = cam_info["restricted_zone_ratio"]
            zone = np.array([[int(width * r[0]), int(height * r[1])] for r in ratios], dtype=np.int32)

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.35, verbose=False)
            output = results[0].plot()

            # Draw restricted zone & HUD
            cv2.polylines(output, [zone], True, (0, 0, 255), 3)
            cv2.putText(output, "RESTRICTED ZONE", (zone[0][0], max(25, zone[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(output, f"{target_cam_id.upper()} | {datetime.now().strftime('%H:%M:%S')}",
                        (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(track_ids)

                for track_id, box, conf in zip(track_ids, coordinates, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    bottom_y = y2
                    inside = cv2.pointPolygonTest(zone, (float(center_x), float(bottom_y)), False)

                    if inside >= 0:
                        cv2.putText(output, "INTRUSION ALERT!", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        cv2.putText(output, f"INTRUDER ID: {track_id}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        now_ts = time.time()
                        if track_id not in last_saved_time or (now_ts - last_saved_time[track_id]) > SAVE_COOLDOWN:
                            conf_val = round(float(conf), 2) if conf is not None else None
                            saved = save_evidence(output, track_id, confidence=conf_val, camera_id=target_cam_id.upper())
                            if saved:
                                latest_alert = {
                                    "person_id": track_id,
                                    "camera_id": target_cam_id.upper(),
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "image_filename": Path(saved).name,
                                }
                            last_saved_time[track_id] = now_ts

            _, buffer = cv2.imencode(".jpg", output)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        if cap is not None:
            cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("Starting IBVAP Command Center at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)