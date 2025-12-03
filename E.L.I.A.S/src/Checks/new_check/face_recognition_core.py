# src/face_recognition_core.py
import os
import json
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_FILE = os.path.join(BASE_DIR, "data", "auth", "user_face.json")

def load_known_face():
    if not os.path.exists(AUTH_FILE):
        return None
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("embedding")
    except Exception as e:
        print("Error loading face data:", e)
        return None

def compare_faces(known_embedding, current_embedding, threshold=0.75):
    """
    Compare two normalized landmark lists ([(x,y),...]).
    Returns (is_match, score) where score in (0..1] (higher = more similar)
    """
    if not known_embedding or not current_embedding:
        return False, 0.0

    total_diff = 0.0
    limit = min(len(known_embedding), len(current_embedding))
    for i in range(limit):
        p1 = known_embedding[i]
        p2 = current_embedding[i]
        try:
            dx = float(p2[0]) - float(p1[0])
            dy = float(p2[1]) - float(p1[1])
            dist = math.sqrt(dx*dx + dy*dy)
        except Exception:
            dist = 1.0
        total_diff += dist

    score = 1.0 / (1.0 + total_diff)  # 0..1
    is_match = score > threshold
    return is_match, float(score)
