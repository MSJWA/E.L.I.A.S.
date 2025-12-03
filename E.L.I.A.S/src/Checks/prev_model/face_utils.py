# src/face_utils.py
import cv2
import os
import numpy as np
import json
from utils import user_face_path, user_face_image_path, log_event

# Path to Haar cascade: put your haarcascade_frontalface_default.xml inside models/ or modify path if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

def _compute_simple_embedding_from_bbox(bbox):
    # bbox = (x, y, w, h)
    x, y, w, h = bbox
    # normalized pseudo-landmarks relative to bbox
    emb = [
        (0.3, 0.35),
        (0.7, 0.35),
        (0.5, 0.55),
        (0.35, 0.75),
        (0.65, 0.75)
    ]
    # Return as list of [x_norm, y_norm]
    return emb


def register_face(user_id: str, device_index=0, require_multiple_frames=5):
    """
    Registers a face for `user_id`. Captures several frames and saves the best snapshot + embedding.
    Run this interactively from console. Silent capture (no GUI).
    """
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError("Cascade missing; ensure models/haarcascade_frontalface_default.xml exists")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)  # fallback
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

    captured = 0
    best_frame = None
    best_bbox = None
    best_area = 0

    # capture up to 3 seconds or until enough good frames
    import time
    start = time.time()
    while time.time() - start < 6 and captured < require_multiple_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) >= 1:
            # choose largest face
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces[0]
            area = w*h
            if area > best_area:
                best_area = area
                best_frame = frame.copy()
                best_bbox = (x, y, w, h)
            captured += 1

    cap.release()

    if best_frame is None or best_bbox is None:
        log_event("face_utils", f"Register failed for {user_id}: no face captured")
        return False, "No face captured"

    # Save image (cropped)
    x, y, w, h = best_bbox
    crop = cv2.cvtColor(best_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    img_path = user_face_image_path(user_id)
    cv2.imwrite(img_path, crop)

    # Create embedding (simple normalized landmarks relative to bbox)
    emb = _compute_simple_embedding_from_bbox(best_bbox)
    data = {"user_id": user_id, "embedding": emb}
    try:
        with open(user_face_path(user_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
        log_event("face_utils", f"Registered face for {user_id}")
        return True, f"Registered {user_id}"
    except Exception as e:
        log_event("face_utils", f"Error saving face for {user_id}: {e}")
        return False, str(e)
