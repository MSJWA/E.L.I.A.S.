import cv2
import mediapipe as mp
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(BASE, "data", "face")

mp_face = mp.solutions.face_mesh

def identify_face():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, 0.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face.FaceMesh(static_image_mode=True) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        return None, 0.0

    lm = res.multi_face_landmarks[0]

    vec = []
    for p in lm.landmark:
        vec.append([p.x, p.y, p.z])
    emb = np.array(vec).flatten()

    best_id = None
    best_score = -1

    for f in os.listdir(FACE_DIR):
        if not f.endswith(".npy"):
            continue

        uid = f[:-4]
        ref = np.load(os.path.join(FACE_DIR, f))

        score = np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref))

        if score > best_score:
            best_score = score
            best_id = uid

    return best_id, best_score
