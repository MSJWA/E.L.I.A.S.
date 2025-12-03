# src/face_recognition_core.py
"""
Face recognition core (lightweight, demo-ready).

Responsibilities:
- Load / save canonical face embedding (JSON).
- Provide efficient comparison between stored embedding and current embedding.
- Small memory footprint and simple Euclidean-based similarity mapping.
- Minimal external dependencies (only builtins + json + os + math).

File format:
data/auth/user_face.json  -> {"user_id": "...", "method":"landmarks-v1", "embedding":[[x,y],...], "created_at": "...", "metadata": {...}}

Notes on thresholds:
- Default compare threshold is 0.65 (tunable).
- Score is normalized in (0..1) where larger is more similar.
"""

from __future__ import annotations
import os
import json
import math
import time
from typing import List, Optional, Tuple, Dict

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_DIR = os.path.join(BASE_DIR, "data", "auth")
AUTH_FILE = os.path.join(AUTH_DIR, "user_face.json")

# Performance: small constants, no heavy objects kept in memory.
# We'll keep only simple lists and primitive types.


def _ensure_auth_dir():
    if not os.path.exists(AUTH_DIR):
        try:
            os.makedirs(AUTH_DIR, exist_ok=True)
        except Exception:
            # best-effort; failure will surface when writing file
            pass


def save_known_face(embedding: List[List[float]],
                    user_id: str = "admin",
                    method: str = "landmarks-v1",
                    metadata: Optional[Dict] = None) -> bool:
    """
    Save canonical face embedding to disk (JSON).
    embedding: list of [x,y] normalized coordinates (0..1).
    Returns True on success.
    """
    _ensure_auth_dir()
    payload = {
        "user_id": user_id,
        "method": method,
        "embedding": embedding,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata or {}
    }
    try:
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return True
    except Exception:
        return False


def load_known_face() -> Optional[List[List[float]]]:
    """
    Load stored embedding. Returns list of [x,y] floats or None if missing/corrupted.
    Time: O(file size) ~ small. Space: small.
    """
    if not os.path.exists(AUTH_FILE):
        return None
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        emb = data.get("embedding")
        # Basic validation: ensure it's a list of pairs
        if not emb or not isinstance(emb, list):
            return None
        # normalize types and length-safe conversion
        cleaned = []
        for p in emb:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return None
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                return None
            cleaned.append([x, y])
        return cleaned
    except Exception:
        return None


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # micro-optimized distance
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def compare_faces(known_embedding: Optional[List[List[float]]],
                  current_embedding: Optional[List[List[float]]],
                  threshold: float = 0.65) -> Tuple[bool, float]:
    """
    Compare two embeddings and return (is_match, score).
    - known_embedding and current_embedding are lists of [x,y] normalized coordinates.
    - score is mapped to 0..1 (higher -> more similar).
    - threshold default 0.65 (tunable).
    Complexity:
      - Time: O(k) where k = min(len(known), len(current))
      - Space: O(1) extra
    """
    if not known_embedding or not current_embedding:
        return False, 0.0

    # Use the minimum length for safety
    limit = min(len(known_embedding), len(current_embedding))
    if limit == 0:
        return False, 0.0

    # Sum of Euclidean distances
    total_diff = 0.0
    # loop using local variables for speed
    ke = known_embedding
    ce = current_embedding
    for i in range(limit):
        p1 = ke[i]
        p2 = ce[i]
        total_diff += _euclidean((p1[0], p1[1]), (p2[0], p2[1]))

    # normalize to a similarity score between 0 and 1
    # Observation: total_diff ∈ [0, ~k*sqrt(2)] for normalized coords. We use a smooth mapping:
    # score = 1 / (1 + alpha * total_diff)
    # Choose alpha = 1.0 for conservative mapping. If you need more permissive scoring, reduce alpha.
    alpha = 1.0
    raw = total_diff
    score = 1.0 / (1.0 + alpha * raw)

    is_match = score >= threshold
    # clamp
    try:
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
    except Exception:
        score = float(score)

    return bool(is_match), float(score)


# ----------------- CLI quick tests (optional) -----------------
if __name__ == "__main__":
    # Small self-test: save sample, load it back, compare
    sample = [[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]]
    ok = save_known_face(sample)
    print("Saved:", ok)
    loaded = load_known_face()
    print("Loaded:", loaded)
    match, score = compare_faces(loaded, sample)
    print("Match:", match, "Score:", score)
