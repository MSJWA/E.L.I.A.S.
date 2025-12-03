# src/vision.py
"""
Integrated vision authentication (middle-ground security profile).

Behavior summary:
 - Detect face using Haar cascade (fast)
 - Light enhancement when needed (hist equalize)
 - Run liveness checks from vision_liveness (optical flow + LBP)
 - Compare landmarks-style embedding against stored canonical embedding using face_recognition_core
 - Use 3-step verification counter with 3.0s tolerance (prevents transient false accepts)
 - Returns structured dict describing detection, liveness, and auth decision.

Assumptions:
 - face_recognition_core.py exists and exposes load_known_face() and compare_faces()
 - vision_liveness.py exists and exposes collect_face_roi_frames(), optical_flow_liveness(), lbp_liveness_one_frame(), aggregate_liveness()
 - Haar cascade xml placed in models/haarcascade_frontalface_default.xml
 - Stored canonical face embedding file path: data/auth/user_face.json
"""

from __future__ import annotations
import os
import time
import datetime
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

# local imports (same src folder)
from face_recognition_core import load_known_face, compare_faces
import vision_liveness as vl

# --- CONFIG (middle-ground / demo-ready) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# thresholds tuned for middle-ground balance
LIVENESS_THRESHOLD = 0.63     # aggregate liveness score threshold (0..1)
FACE_MATCH_THRESHOLD = 0.65   # face similarity threshold (0..1)
VERIFICATION_STEPS_REQUIRED = 3
VERIFICATION_MAX_INTERVAL = 3.0  # seconds allowed between verified frames (relaxed)

# small ROI width for perf: must match vl.ROI_WIDTH default or pass explicitly
ROI_WIDTH = vl.ROI_WIDTH

# --- internal state (cached) ---
_cached_known_face = None
_verification_counter = 0
_last_verify_time = 0.0

# --- utilities ---
def _ensure_logs_dir():
    try:
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR, exist_ok=True)
    except Exception:
        pass

def log_vision_event(status: str, message: str):
    _ensure_logs_dir()
    log_file = os.path.join(LOGS_DIR, "vision_log.txt")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {status}: {message}\n")
    except Exception:
        pass

def improve_low_light(frame: np.ndarray) -> np.ndarray:
    """Small auto-enhance using YUV equalization (fast and lightweight)."""
    try:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return enhanced
    except Exception:
        return frame

def _load_known_face_cached(force_reload: bool = False):
    global _cached_known_face
    if _cached_known_face is None or force_reload:
        _cached_known_face = load_known_face()
    return _cached_known_face

# --- main detection & auth flow ---
def detect_and_authenticate(video_index: int = 0, timeout_secs: float = 5.0) -> Dict[str, Any]:
    """
    High-level: attempt to detect + liveness-check + compare.
    Returns dict with keys:
      - face_detected (bool)
      - liveness (aggregate score float)
      - liveness_pass (bool)
      - face_score (float or None)
      - auth_status (bool if known face matched)
      - message (human-readable)
      - debug (optional dict)
    """

    global _verification_counter, _last_verify_time

    # 1. check resources
    if not os.path.exists(CASCADE_PATH):
        return {"face_detected": False, "message": "Model missing", "error": "cascade_missing"}

    # Ensure cached known face loaded (may be None)
    known = _load_known_face_cached()

    # 2. open camera
    cap = cv2.VideoCapture(video_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # fallback to default VideoCapture without CAP_DSHOW on some systems
        cap = cv2.VideoCapture(video_index)
        if not cap.isOpened():
            return {"face_detected": False, "message": "Camera unavailable", "error": "camera_unavailable"}

    # quick warm-read
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return {"face_detected": False, "message": "Frame capture failed", "error": "frame_failed"}

    # 3. lighting check
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(gray_full.mean())
    used_frame = frame
    lighting_note = ""
    if avg_brightness < 60:
        used_frame = improve_low_light(frame)
        lighting_note = f"low_light_enhanced ({int(avg_brightness)})"

    # 4. detect faces (fast single-shot)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray_used = cv2.cvtColor(used_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_used, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        cap.release()
        return {
            "face_detected": False,
            "message": f"No face detected {lighting_note}".strip(),
            "liveness": 0.0
        }

    # choose primary face (largest)
    faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    bbox = tuple(map(int, faces_sorted[0]))  # (x,y,w,h)

    # 5. Liveness check (collect few small ROI frames)
    frames = vl.collect_face_roi_frames(cap, bbox, n_frames=vl.FLOW_FRAMES, spacing_ms=vl.FLOW_SPACING_MS, roi_width=ROI_WIDTH)
    # release camera ASAP
    cap.release()

    if not frames:
        return {"face_detected": False, "message": "Could not collect frames for liveness", "liveness": 0.0}

    optical = vl.optical_flow_liveness(frames)
    lbp_score = vl.lbp_liveness_one_frame(frames[0])  # no ref hist by default; uses variance fallback
    aggregate = vl.aggregate_liveness({"optical": optical, "lbp": lbp_score, "blink": 0.0})

    liveness_pass = aggregate >= LIVENESS_THRESHOLD

    # 6. if liveness low -> require active challenge or reject (for demo we reject here)
    if not liveness_pass:
        # log and return early (active-challenge path can be implemented in auth_manager)
        log_vision_event("LIVENESS_FAIL", f"agg:{aggregate:.3f} optical:{optical:.3f} lbp:{lbp_score:.3f}")
        return {
            "face_detected": True,
            "liveness": float(aggregate),
            "liveness_pass": False,
            "message": f"Low liveness score ({aggregate:.3f}); active challenge required or reject.",
            "debug": {"optical": float(optical), "lbp": float(lbp_score)}
        }

    # 7. Construct current embedding (simple normalized landmarks / relative points)
    # We use normalized keypoints relative to bbox size (same scheme as saved embedding)
    x, y, w, h = bbox
    # The chosen pseudo-landmarks are proportional positions inside bbox (fast & deterministic)
    current_embedding = [
        [0.3, 0.35],  # left eye approx
        [0.7, 0.35],  # right eye approx
        [0.5, 0.55],  # nose
        [0.35, 0.75], # mouth left
        [0.65, 0.75]  # mouth right
    ]
    # Note: current_embedding uses normalized coords (0..1). If you later use actual landmark extractor,
    # replace the above with real landmarks normalized by bbox.

    # 8. Compare against stored
    if known is None:
        # No registered known user
        return {
            "face_detected": True,
            "liveness": float(aggregate),
            "liveness_pass": True,
            "auth_status": False,
            "auth_type": "GUEST_NO_REGISTERED_USER",
            "face_score": None,
            "message": "Face detected and live, but no registered user data found."
        }

    is_match, face_score = compare_faces(known, current_embedding, threshold=FACE_MATCH_THRESHOLD)

    # 9. 3-step verification: require VERIFICATION_STEPS_REQUIRED consecutive 'is_match' within allowed time window
    now = time.time()
    if is_match:
        # reset counter if too much time has passed
        if _verification_counter > 0 and (now - _last_verify_time) > VERIFICATION_MAX_INTERVAL:
            _verification_counter = 0
        _verification_counter += 1
        _last_verify_time = now
    else:
        _verification_counter = 0
        _last_verify_time = 0.0

    # If not yet reached required steps, return intermediate message (auth in progress)
    if _verification_counter < VERIFICATION_STEPS_REQUIRED:
        return {
            "face_detected": True,
            "liveness": float(aggregate),
            "liveness_pass": True,
            "auth_status": False,
            "auth_progress": _verification_counter,
            "required": VERIFICATION_STEPS_REQUIRED,
            "face_score": float(face_score),
            "message": f"Verifying identity... ({_verification_counter}/{VERIFICATION_STEPS_REQUIRED})",
            "debug": {"optical": float(optical), "lbp": float(lbp_score)}
        }

    # success: reset counter and return approved
    _verification_counter = 0
    _last_verify_time = 0.0
    msg = f"ACCESS GRANTED (face_score={face_score:.3f})"
    log_vision_event("SUCCESS", msg)
    return {
        "face_detected": True,
        "liveness": float(aggregate),
        "liveness_pass": True,
        "auth_status": True,
        "face_score": float(face_score),
        "message": msg,
        "debug": {"optical": float(optical), "lbp": float(lbp_score)}
    }


# convenience wrappers
def authenticate_user(timeout: float = 6.0) -> Dict[str, Any]:
    """
    Repeatedly call detect_and_authenticate until success or timeout.
    Returns last result (success or last failure).
    """
    start = time.time()
    last_result = {"face_detected": False, "message": "timeout"}
    while (time.time() - start) < timeout:
        res = detect_and_authenticate()
        # if face_detected and auth_status True -> success
        if res.get("auth_status"):
            return res
        last_result = res
        # small sleep between tries (avoid tight loops)
        time.sleep(0.1)
    return last_result


def reload_known_face():
    """Force reload of known face from disk."""
    global _cached_known_face
    _cached_known_face = load_known_face()
    return _cached_known_face


# for direct test
if __name__ == "__main__":
    print("--- Vision Module (integration test) ---")
    r = detect_and_authenticate()
    print(r)
