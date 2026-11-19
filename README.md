# IBVAP — AI-Based Intelligent Video Analytics Platform for Border Surveillance

**Smart India Hackathon (SIH26187)**  
**Ministry of Home Affairs | Theme: Blockchain & Cybersecurity**

## Overview
IBVAP upgrades existing CCTV infrastructure with real-time AI intrusion detection, tamper-proof blockchain evidence sealing, and a role-based Command Center dashboard.

## Key Features
- **Real-Time AI Surveillance:** Person detection & tracking using **YOLO11 + ByteTrack** with configurable restricted perimeter zones.
- **Existing CCTV Compatible:** Supports live cameras, sample border footage, and RTSP IP streams via `cameras.json`, plus **Night Vision (CLAHE)** enhancement.
- **Blockchain Tamper-Proof Evidence:** Every intrusion snapshot is hashed (**SHA-256**) and sealed in an immutable hash-chained ledger (`blockchain_ledger.json`) with instant **VERIFIED / TAMPERED** checks.
- **Cybersecurity & Command Center:** Role-Based Access Control (`Commander` / `Operator`), security audit logging, live alert banner, hourly intrusion analytics, and Evidence Locker UI.

## How to Run Locally
```powershell
pip install -r requirements.txt
python backend/api.py
