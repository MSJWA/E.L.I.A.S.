# src/vision_auth.py
import os
import json
import cv2
from camera_utils import open_camera
from face_recognition_core import load_known_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_FILE = os.path.join(BASE_DIR, "data", "auth", "user_face.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)

def register_admin():
    print("Registering admin face (single-shot). Please face camera and keep steady.")
    if not os.path.exists(CASCADE_PATH):
        print("Cascade missing:", CASCADE_PATH)
        return False

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap, backend = open_camera(preferred=("MSMF","ANY","DSHOW"), set_resolution=(640,480), warmup_frames=3, exposure_tweak=True)
    if cap is None:
        print("Camera unavailable.")
        return False

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Frame failed.")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
    if len(faces) == 0:
        print("No face found. Improve lighting and try again.")
        return False

    x,y,w,h = faces[0]
    # simple normalized "landmarks" (placeholder)
    embedding = [
        [0.3, 0.35],
        [0.7, 0.35],
        [0.5, 0.55],
        [0.35, 0.75],
        [0.65, 0.75]
    ]
    data = {"user_id": "admin", "embedding": embedding}
    try:
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"✅ ADMIN REGISTERED at {AUTH_FILE}")
        return True
    except Exception as e:
        print("Error saving:", e)
        return False

def run_auth_check():
    import vision
    return vision.detect_and_authenticate()

if __name__ == "__main__":
    c = input("Type 'r' to register admin, 't' to test auth: ").strip().lower()
    if c == 'r':
        register_admin()
    elif c == 't':
        print(run_auth_check())
    else:
        print("invalid")
