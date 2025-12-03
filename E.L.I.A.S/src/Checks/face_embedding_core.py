# face_embedding_core.py
# Mediapipe detection + facenet-pytorch embedding extraction + small helpers

import os, time, json
import numpy as np
import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from typing import Tuple

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FACE_DIR = os.path.join(BASE, "data", "face")
os.makedirs(DATA_FACE_DIR, exist_ok=True)

# mediapipe face detection + face mesh for alignment
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

# load FaceNet model (pretrained)
_resnet = None
def get_facenet_model(device="cpu"):
    global _resnet
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _resnet

def detect_and_align(frame) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Return aligned face (RGB, 160x160) and bbox (x,y,w,h) or (None, None)
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        res = fd.process(img_rgb)
    if not res.detections:
        return None, None
    det = res.detections[0]
    # bbox in relative coords
    loc = det.location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x = int(loc.xmin * w)
    y = int(loc.ymin * h)
    bw = int(loc.width * w)
    bh = int(loc.height * h)
    # expand a bit
    pad = int(max(bw, bh) * 0.2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    # align/crop to square and resize to 160x160
    h2, w2 = crop.shape[:2]
    side = max(h2, w2)
    sq = np.zeros((side, side, 3), dtype=np.uint8)
    sx = (side - w2) // 2
    sy = (side - h2) // 2
    sq[sy:sy+h2, sx:sx+w2] = crop
    aligned = cv2.resize(sq, (160,160))
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned_rgb, (x0, y0, x1 - x0, y1 - y0)

def image_to_embedding(rgb_face, device="cpu"):
    """
    Input: rgb_face as numpy uint8 shape (160,160,3)
    Output: 512-d or 512/128-d embedding normalized (l2)
    """
    model = get_facenet_model(device=device)
    # facenet-pytorch expects PIL or tensor; we convert
    import torch
    face_t = torch.tensor(rgb_face, dtype=torch.float32).permute(2,0,1).unsqueeze(0)  # 1,3,160,160
    # normalize to [0,1] then to model expectation (-1..1)
    face_t = (face_t / 255.0 - 0.5) / 0.5
    face_t = face_t.to(device)
    with torch.no_grad():
        emb = model(face_t)  # e.g. 512-d (InceptionResnetV1 default gives 512)
    emb = emb.cpu().numpy().reshape(-1)
    # L2 normalize
    emb = emb / np.linalg.norm(emb + 1e-10)
    return emb

def save_embedding(user_id: str, embedding: np.ndarray):
    out = os.path.join(DATA_FACE_DIR, f"{user_id}.npy")
    np.save(out, embedding)
    return out

def load_embedding(user_id: str):
    p = os.path.join(DATA_FACE_DIR, f"{user_id}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a)+1e-10)
    b = b / (np.linalg.norm(b)+1e-10)
    return float(np.dot(a, b))

# small debug
if __name__ == "__main__":
    print("face_embedding_core quick test")
