# src/vision.py
import os
import time
import datetime
import cv2
from typing import Dict, Any
from camera_utils import open_camera
from face_recognition_core import load_known_face, compare_faces
import vision_liveness as vl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

LIVENESS_THRESHOLD = 0.62
FACE_MATCH_THRESHOLD = 0.70
VERIFICATION_STEPS_REQUIRED = 3
VERIFICATION_MAX_INTERVAL = 3.0

_cached_known_face = None
_verification_counter = 0
_last_verify_time = 0.0

def _ensure_logs_dir():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)

def log_vision_event(status: str, message: str):
    _ensure_logs_dir()
    log_file = os.path.join(LOGS_DIR, "vision_log.txt")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {status}: {message}\n")
    except Exception:
        pass

def improve_low_light(frame):
    try:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    except Exception:
        return frame

def _load_known_face_cached(force=False):
    global _cached_known_face
    if _cached_known_face is None or force:
        _cached_known_face = load_known_face()
    return _cached_known_face

def detect_and_authenticate(video_index:int=0) -> Dict[str, Any]:
    global _verification_counter, _last_verify_time
    if not os.path.exists(CASCADE_PATH):
        return {"face_detected": False, "message": "Model missing", "error": "cascade_missing"}

    known = _load_known_face_cached()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap, backend = open_camera(index=video_index, preferred=("MSMF","ANY","DSHOW"), set_resolution=(640,480), exposure_tweak=True)
    if cap is None:
        return {"face_detected": False, "message": "Camera unavailable", "error": "camera_unavailable"}

    ret, frame = cap.read()
    if (not ret) or frame is None:
        cap.release()
        return {"face_detected": False, "message": "Frame capture failed", "error": "frame_failed"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(gray.mean())
    lighting_note = ""
    if avg_brightness < 60:
        frame = improve_low_light(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lighting_note = f"(low:{int(avg_brightness)})"

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
    if len(faces) == 0:
        cap.release()
        return {"face_detected": False, "message": f"No face detected {lighting_note}".strip(), "liveness": 0.0}

    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    bbox = tuple(map(int, faces_sorted[0]))

    frames = vl.collect_face_roi_frames(cap, bbox)
    cap.release()

    if not frames:
        return {"face_detected": True, "liveness": 0.0, "message": "Could not collect frames for liveness"}

    optical = vl.optical_flow_liveness(frames)
    lbp = vl.lbp_liveness_one_frame(frames[0])
    aggregate = vl.aggregate_liveness({"optical": optical, "lbp": lbp, "blink": 0.0})
    liveness_pass = aggregate >= LIVENESS_THRESHOLD

    if not liveness_pass:
        log_vision_event("LIVENESS_FAIL", f"agg:{aggregate:.3f}")
        return {"face_detected": True, "liveness": float(aggregate), "liveness_pass": False, "message": f"Low liveness score ({aggregate:.3f})"}

    x,y,w,h = bbox
    # lightweight placeholder embedding: normalized keypoints relative to bbox
    current_embedding = [
        [0.3, 0.35],
        [0.7, 0.35],
        [0.5, 0.55],
        [0.35, 0.75],
        [0.65, 0.75]
    ]

    if known is None:
        return {"face_detected": True, "liveness": float(aggregate), "liveness_pass": True, "auth_status": False, "auth_type": "GUEST_NO_REGISTERED_USER", "message": "Live face but no registered user."}

    is_match, face_score = compare_faces(known, current_embedding, threshold=FACE_MATCH_THRESHOLD)
    now = time.time()
    if is_match:
        if _verification_counter > 0 and (now - _last_verify_time) > VERIFICATION_MAX_INTERVAL:
            _verification_counter = 0
        _verification_counter += 1
        _last_verify_time = now
    else:
        _verification_counter = 0
        _last_verify_time = 0.0

    if _verification_counter < VERIFICATION_STEPS_REQUIRED:
        return {"face_detected": True, "liveness": float(aggregate), "liveness_pass": True, "auth_status": False, "auth_progress": _verification_counter, "required": VERIFICATION_STEPS_REQUIRED, "face_score": float(face_score), "message": f"Verifying... ({_verification_counter}/{VERIFICATION_STEPS_REQUIRED})"}

    _verification_counter = 0
    _last_verify_time = 0.0
    msg = f"ACCESS GRANTED (score={face_score:.3f})"
    log_vision_event("SUCCESS", msg)
    return {"face_detected": True, "liveness": float(aggregate), "liveness_pass": True, "auth_status": True, "face_score": float(face_score), "message": msg}
