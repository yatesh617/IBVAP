import hashlib
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = BASE_DIR / "evidence"
LEDGER_FILE = EVIDENCE_DIR / "blockchain_ledger.json"


def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _calculate_block_hash(block: dict) -> str:
    block_copy = {
        "index": block["index"],
        "timestamp": block["timestamp"],
        "camera_id": block["camera_id"],
        "person_id": block["person_id"],
        "image_filename": block["image_filename"],
        "evidence_hash": block["evidence_hash"],
        "previous_hash": block["previous_hash"],
    }
    encoded = json.dumps(block_copy, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _init_ledger() -> list:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if LEDGER_FILE.exists():
        try:
            data = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            pass

    genesis = {
        "index": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": "GENESIS",
        "person_id": 0,
        "confidence": None,
        "image_filename": "GENESIS_BLOCK",
        "evidence_hash": "0" * 64,
        "previous_hash": "0" * 64,
    }
    genesis["block_hash"] = _calculate_block_hash(genesis)
    genesis["tx_id"] = "0x" + genesis["block_hash"][:16]
    LEDGER_FILE.write_text(json.dumps([genesis], indent=2), encoding="utf-8")
    return [genesis]


def get_ledger() -> list:
    return _init_ledger()


def add_block(image_path: Path, person_id: int, camera_id: str, timestamp: str, confidence=None) -> dict:
    chain = _init_ledger()
    last_block = chain[-1]
    evidence_hash = compute_file_hash(image_path)

    new_block = {
        "index": last_block["index"] + 1,
        "timestamp": timestamp,
        "camera_id": camera_id,
        "person_id": int(person_id),
        "confidence": confidence,
        "image_filename": image_path.name,
        "evidence_hash": evidence_hash,
        "previous_hash": last_block["block_hash"],
    }
    new_block["block_hash"] = _calculate_block_hash(new_block)
    new_block["tx_id"] = "0x" + new_block["block_hash"][:16]

    chain.append(new_block)
    LEDGER_FILE.write_text(json.dumps(chain, indent=2), encoding="utf-8")
    return new_block


def verify_evidence(image_filename: str) -> dict:
    chain = _init_ledger()

    # 1. Verify full blockchain integrity
    for i in range(1, len(chain)):
        curr = chain[i]
        prev = chain[i - 1]
        if curr["previous_hash"] != prev["block_hash"] or curr["block_hash"] != _calculate_block_hash(curr):
            return {
                "verified": False,
                "status": "CHAIN_TAMPERED",
                "message": f"Blockchain ledger broken at Block #{curr['index']}!",
            }

    # 2. Find matching evidence block
    target_block = next((b for b in reversed(chain) if b["image_filename"] == image_filename), None)
    if not target_block:
        return {
            "verified": False,
            "status": "NOT_FOUND",
            "message": "No blockchain record found for this evidence.",
        }

    image_path = EVIDENCE_DIR / image_filename
    if not image_path.exists():
        return {
            "verified": False,
            "status": "FILE_MISSING",
            "message": "Evidence image deleted or missing from disk!",
            "block_index": target_block["index"],
            "tx_id": target_block["tx_id"],
        }

    current_hash = compute_file_hash(image_path)
    if current_hash == target_block["evidence_hash"]:
        return {
            "verified": True,
            "status": "VERIFIED",
            "message": "Image hash matches immutable blockchain record.",
            "block_index": target_block["index"],
            "tx_id": target_block["tx_id"],
            "stored_hash": target_block["evidence_hash"],
            "current_hash": current_hash,
        }
    else:
        return {
            "verified": False,
            "status": "TAMPERED",
            "message": "HASH MISMATCH! Evidence image was modified after capture.",
            "block_index": target_block["index"],
            "tx_id": target_block["tx_id"],
            "stored_hash": target_block["evidence_hash"],
            "current_hash": current_hash,
        }