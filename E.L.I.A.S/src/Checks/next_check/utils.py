import os, json, time, numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(BASE_DIR, "data", "face")
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def save_embedding(path, arr):
    np.save(path, np.asarray(arr, dtype=float))

def load_embedding(path):
    return np.load(path)

def list_face_users():
    files = [f for f in os.listdir(FACE_DIR) if f.endswith(".npy")]
    return [os.path.splitext(f)[0] for f in files]

def log_event(kind, msg, extra=None):
    fn = os.path.join(LOG_DIR, "mfa_log.txt")
    entry = {"ts": now_iso(), "kind": kind, "msg": msg}
    if extra: entry["extra"] = extra
    with open(fn, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
