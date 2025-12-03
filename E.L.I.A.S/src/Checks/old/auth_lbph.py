# src/auth_lbph.py
import os, json
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "face")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lbph.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# load images labeled by folder name
faces = []
labels = []
label_map = {}
label_id = 0
face_cascade = cv2.CascadeClassifier(os.path.join(BASE, "models", "haarcascade_frontalface_default.xml"))

for user in sorted(os.listdir(DATA_DIR)):
    user_path = os.path.join(DATA_DIR, user)
    if not os.path.isdir(user_path): continue
    label_map[str(label_id)] = user
    for fname in os.listdir(user_path):
        if not fname.lower().endswith((".jpg",".png")): continue
        img = cv2.imread(os.path.join(user_path, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50,50))
        if len(faces_rect)==0: continue
        x,y,w,h = faces_rect[0]
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop, (200,200))
        faces.append(crop)
        labels.append(label_id)
    label_id += 1

if len(faces)==0:
    raise SystemExit("No faces found to train")

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(label_map, f)
print("LBPH model trained:", MODEL_PATH)
