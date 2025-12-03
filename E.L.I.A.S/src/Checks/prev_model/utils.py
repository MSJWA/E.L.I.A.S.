# src/utils.py
import os
import json
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACE_DIR = os.path.join(DATA_DIR, "face")
VOICE_DIR = os.path.join(DATA_DIR, "voice")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def user_face_path(user_id: str):
    return os.path.join(FACE_DIR, f"{user_id}.json")


def user_face_image_path(user_id: str):
    return os.path.join(FACE_DIR, f"{user_id}.jpg")


def user_voice_path(user_id: str):
    return os.path.join(VOICE_DIR, f"{user_id}.json")


def log_event(component: str, message: str):
    log_file = os.path.join(LOGS_DIR, "auth_log.txt")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{component}] {message}\n"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass
