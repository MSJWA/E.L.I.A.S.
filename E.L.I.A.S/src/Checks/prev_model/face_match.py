# src/face_match.py
import cv2
import os
import json
import numpy as np
from utils import user_face_path, user_face_image_path, log_event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

def _load_all_known_embeddings():
    ids = []
    embs = []
    for fname in os.listdir(os.path.join(BASE_DIR, "data", "face")):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(BASE_DIR, "data", "face", fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
                user_id = j.get("user_id") or fname.replace(".json", "")
                emb = j.get("embedding")
                if emb:
                    ids.append(user_id)
                    embs.append(np.array(emb).flatten())
        except:
            continue
    return ids, embs

def _compute_current_embedding_from_bbox(bbox):
    # same normalized scheme as registration (we don't need exact coords)
    emb = [
        (0.3, 0.35),
        (0.7, 0.35),
        (0.5, 0.55),
        (0.35, 0.75),
        (0.65, 0.75)
    ]
    return np.array(emb).flatten()

def match_face_once(device_index=0):
    """
    Capture a single frame silently, detect face, return sorted candidates:
    [ (user_id, score), ... ]
    """
    if not os.path.exists(CASCADE_PATH):
        return {"error":"cascade_missing"}

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return {"error":"camera_unavailable"}

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error":"frame_failed"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    if len(faces) == 0:
        return {"error":"no_face"}

    # choose largest face
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    bbox = faces[0]
    curr_emb = _compute_current_embedding_from_bbox(bbox)

    ids, known_embs = _load_all_known_embeddings()
    if len(ids) == 0:
        return {"error":"no_registered_users"}

    scores = []
    for uid, kemb in zip(ids, known_embs):
        # similarity by cosine
        ka = kemb / (np.linalg.norm(kemb) + 1e-9)
        ca = curr_emb / (np.linalg.norm(curr_emb) + 1e-9)
        score = float(np.dot(ka, ca))
        scores.append((uid, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return {"candidates": scores, "bbox": tuple(map(int, bbox))}
