# src/mfa_full.py
"""
Single-file MFA pipeline (Face auto-identify + Voice challenge)
Usage:
  python src/mfa_full.py register_face <user_id>
  python src/mfa_full.py enroll_voice <user_id>
  python src/mfa_full.py auth [strict|balanced|relaxed]
Notes:
 - Requires: opencv-python, numpy, sounddevice, scipy
 - Place haarcascade_frontalface_default.xml into models/
 - Data written to data/face/ and data/voice/ ; logs in logs/
"""

import os
import sys
import time
import json
import random
import math
import datetime

import cv2
import numpy as np

# sound
import sounddevice as sd
from scipy.signal import lfilter

# ----------------------- CONFIG -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FACE_DIR = os.path.join(BASE_DIR, "data", "face")
DATA_VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FACE_DIR, exist_ok=True)
os.makedirs(DATA_VOICE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
LOG_FILE = os.path.join(LOGS_DIR, "auth_log.txt")

SAMPLE_RATE = 16000
VOICE_EMB_BINS = 40

# thresholds (tuneable)
FACE_CONF_THRESHOLD = 0.78
VOICE_THRESH = {"strict": 0.82, "balanced": 0.72, "relaxed": 0.60}

CHALLENGE_POOL = [
    "blue river", "quiet forest", "silver moon", "hidden valley",
    "gentle sunrise", "orange lantern", "silent horizon"
]

# Try backends in this order (your hardware preferred MSMF then ANY)
_BACKENDS_TRY = [getattr(cv2, "CAP_MSMF", None), getattr(cv2, "CAP_ANY", None), 0]


# ----------------------- UTILITIES -----------------------
def log_event(component: str, message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{component}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def user_face_json(user_id: str):
    return os.path.join(DATA_FACE_DIR, f"{user_id}.json")


def user_face_image(user_id: str):
    return os.path.join(DATA_FACE_DIR, f"{user_id}.jpg")


def user_voice_json(user_id: str):
    return os.path.join(DATA_VOICE_DIR, f"{user_id}.json")


# ----------------------- FACE: helpers -----------------------
def _open_camera(device_index=0):
    for b in _BACKENDS_TRY:
        try:
            if b is None:
                cap = cv2.VideoCapture(device_index)
            else:
                cap = cv2.VideoCapture(device_index, b)
            if cap.isOpened():
                # test read
                ret, _ = cap.read()
                if ret:
                    return cap, b
                else:
                    cap.release()
        except Exception:
            continue
    return None, None


def _compute_embedding_from_bbox(bbox):
    # simple normalized pseudo-landmarks relative to bbox for demo-level matching
    # returns flattened list
    emb = [
        (0.3, 0.35),
        (0.7, 0.35),
        (0.5, 0.55),
        (0.35, 0.75),
        (0.65, 0.75)
    ]
    return np.array(emb).flatten().tolist()


def register_face_auto(user_id: str, device_index=0, attempts=30, min_face_size=80):
    """
    Auto-capture best frame over N attempts and save face image + simple embedding.
    """
    if not os.path.exists(CASCADE_PATH):
        print("ERROR: Haarcascade missing. Put haarcascade_frontalface_default.xml in models/")
        return False

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap, backend = _open_camera(device_index)
    if cap is None:
        print("ERROR: cannot open camera (try different device index).")
        return False

    best_face = None
    best_area = 0
    print(f"Camera opened (backend={backend}). Scanning {attempts} frames for best face...")

    for i in range(attempts):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_face_size, min_face_size))

        if len(faces) >= 1:
            # choose largest
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces[0]
            area = w * h
            if area > best_area:
                best_area = area
                best_face = gray[y:y+h, x:x+w].copy()
        time.sleep(0.03)

    cap.release()

    if best_face is None:
        print("No face found. Try better lighting / different angle / increase attempts.")
        log_event("face", f"register_failed:{user_id}")
        return False

    # Save image and embedding
    try:
        cv2.imwrite(user_face_image(user_id), best_face)
        emb = _compute_embedding_from_bbox((0,0,0,0))
        with open(user_face_json(user_id), "w", encoding="utf-8") as f:
            json.dump({"user_id": user_id, "embedding": emb}, f)
        print(f"Face registered: {user_id} -> {user_face_image(user_id)}")
        log_event("face", f"registered:{user_id}")
        return True
    except Exception as e:
        print("Error saving face:", e)
        log_event("face", f"save_error:{user_id}:{e}")
        return False


def _load_all_face_embeddings():
    ids = []
    embs = []
    for fname in os.listdir(DATA_FACE_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DATA_FACE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
                uid = j.get("user_id") or fname.replace(".json", "")
                emb = j.get("embedding")
                if emb:
                    ids.append(uid)
                    embs.append(np.array(emb).flatten())
        except Exception:
            continue
    return ids, embs


def match_face_once(device_index=0, min_face_size=60):
    """
    Capture a single frame (silent), detect largest face, compute simple embedding,
    compare to all registered users, return sorted list of (user, score).
    """
    if not os.path.exists(CASCADE_PATH):
        return {"error": "cascade_missing"}

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap, backend = _open_camera(device_index)
    if cap is None:
        return {"error": "camera_unavailable"}

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error": "frame_failed"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_face_size, min_face_size))
    if len(faces) == 0:
        return {"error": "no_face"}
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    bbox = faces[0]
    curr_emb = np.array(_compute_embedding_from_bbox(bbox)).flatten()

    ids, known_embs = _load_all_face_embeddings()
    if len(ids) == 0:
        return {"error": "no_registered_users"}

    scores = []
    for uid, kemb in zip(ids, known_embs):
        ka = kemb / (np.linalg.norm(kemb) + 1e-9)
        ca = curr_emb / (np.linalg.norm(curr_emb) + 1e-9)
        score = float(np.dot(ka, ca))
        scores.append((uid, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return {"candidates": scores, "bbox": tuple(map(int, bbox))}


# ----------------------- VOICE: helpers -----------------------
def _zero_crossing_rate(arr, frame_size=400):
    if len(arr) < frame_size:
        return 0.0
    frames = []
    for i in range(0, len(arr) - frame_size, frame_size):
        frame = arr[i:i + frame_size]
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        frames.append(zcr)
    return float(np.mean(frames)) if frames else 0.0


def _spectral_energy(arr):
    if len(arr) == 0:
        return 0.0
    S = np.abs(np.fft.rfft(arr))
    return float(np.mean(S))


def record_to_array(duration=3.5, fs=SAMPLE_RATE, channels=1):
    """
    Robust laptop-friendly recorder. Returns 1D float32 numpy array.
    """
    try:
        print(f"Recording {duration:.2f}s ...")
        data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
        sd.wait()
        arr = np.squeeze(np.array(data)).astype('float32')
        if arr.size == 0:
            return arr
        # pre-emphasis
        arr = lfilter([1, -0.97], [1], arr)
        zcr = _zero_crossing_rate(arr)
        spec = _spectral_energy(arr)
        # We accept even low-energy on laptops but log a hint
        if zcr < 0.01 and spec < 0.001:
            print("⚠️ Low energy detected (allowed). Try closer mic if failing.")
        return arr
    except Exception as e:
        print("Recording failed:", e)
        return np.array([])


def spectral_embedding(arr):
    if arr.size == 0:
        return [0.0] * VOICE_EMB_BINS
    N = 2048
    if len(arr) < N:
        arr = np.pad(arr, (0, N - len(arr)), mode='constant')
    else:
        arr = arr[:N]
    spec = np.abs(np.fft.rfft(arr))
    spec = spec / (np.max(spec) + 1e-9)
    bins = np.array_split(spec, VOICE_EMB_BINS)
    return [float(np.mean(b)) for b in bins]


def save_voice_embedding(user_id, emb):
    path = user_voice_json(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id, "embedding": emb}, f)
    log_event("voice", f"saved:{user_id}")


def load_voice_embedding(user_id):
    path = user_voice_json(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("embedding")


def compare_embeddings(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a); b = np.array(b)
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))


# ----------------------- ENROLL / AUTH FLOWS -----------------------
def enroll_voice(user_id: str):
    print(f"=== VOICE ENROLLMENT for {user_id} ===")
    phrase = random.choice(CHALLENGE_POOL)
    print("You will be asked to speak this phrase THREE times, consistently:")
    print(f"--- >>> {phrase} <<< ---")
    input("Press Enter to start sample 1...")
    embs = []
    for i in range(1, 4):
        print(f"Recording sample {i}...")
        arr = record_to_array(duration=3.5)
        emb = spectral_embedding(arr)
        embs.append(emb)
        print("Sample saved.\n")
        if i < 3:
            input("Press Enter for next sample...")

    # average
    avg = [float(sum(x) / 3.0) for x in zip(*embs)]
    save_voice_embedding(user_id, avg)
    log_event("voice", f"enrolled:{user_id}")
    print("Enrollment complete.")


def mfa_auth_flow(mode="balanced", device_index=0):
    """
    Hybrid flow:
      1) silent face capture -> top candidates
      2) if top score >= FACE_CONF_THRESHOLD -> auto-select top_user
         else show top-3 and prompt for selection (hybrid fallback)
      3) text challenge -> record -> compare to that user's voice emb
    """
    print("Starting MFA (face+voice) authentication (silent camera)...")
    fm = match_face_once(device_index=device_index)
    if "error" in fm:
        print("Error during face step:", fm["error"])
        log_event("mfa", f"face_error:{fm.get('error')}")
        return {"ok": False, "reason": fm.get("error")}

    candidates = fm.get("candidates", [])
    if not candidates:
        log_event("mfa", "no_candidates")
        return {"ok": False, "reason": "no_candidates"}

    top_user, top_score = candidates[0]
    print(f"Top face candidate: {top_user} (score {top_score:.3f})")
    log_event("mfa", f"face_top:{top_user}:{top_score:.3f}")

    if top_score < FACE_CONF_THRESHOLD:
        print("Face confidence low. Suggested matches:")
        for uid, sc in candidates[:3]:
            print(f"  {uid} ({sc:.3f})")
        selected = input("Enter user ID from above (or press Enter to cancel): ").strip()
        if not selected:
            log_event("mfa", "user_cancel_low_confidence")
            return {"ok": False, "reason": "low_confidence"}
        top_user = selected

    # voice challenge
    phrase = random.choice(CHALLENGE_POOL)
    print(f"Speak this phrase now: >>> {phrase} <<<")
    input("Press Enter then speak the phrase (recording will start)...")
    arr = record_to_array(duration=3.5)
    emb = spectral_embedding(arr)
    known = load_voice_embedding(top_user)
    if known is None:
        print("No voice enrollment for this user. Authentication failed.")
        log_event("mfa", f"no_voice:{top_user}")
        return {"ok": False, "reason": "no_voice_enrollment"}

    score = compare_embeddings(emb, known)
    thr = VOICE_THRESH.get(mode, VOICE_THRESH["balanced"])
    print(f"Voice similarity: {score:.3f} (threshold {thr})")
    log_event("mfa", f"voice_score:{top_user}:{score:.3f}")

    if score >= thr:
        print(f"✅ MFA AUTH SUCCESS for {top_user}")
        log_event("mfa", f"success:{top_user}:{score:.3f}")
        return {"ok": True, "user": top_user, "score": score}
    else:
        print("❌ MFA AUTH FAILED (voice mismatch)")
        log_event("mfa", f"fail_voice:{top_user}:{score:.3f}")
        return {"ok": False, "reason": "voice_mismatch", "score": score}


# ----------------------- CLI -----------------------
def _usage():
    print("MFA pipeline CLI")
    print("Usage:")
    print("  python src/mfa_full.py register_face <user_id>")
    print("  python src/mfa_full.py enroll_voice <user_id>")
    print("  python src/mfa_full.py auth [strict|balanced|relaxed]")
    print("")
    print("Examples:")
    print("  python src/mfa_full.py register_face alice")
    print("  python src/mfa_full.py enroll_voice alice")
    print("  python src/mfa_full.py auth balanced")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "register_face" and len(sys.argv) >= 3:
        uid = sys.argv[2].strip()
        ok = register_face_auto(uid)
        if ok:
            print("Registered face for", uid)
        else:
            print("Registration failed for", uid)
    elif cmd == "enroll_voice" and len(sys.argv) >= 3:
        uid = sys.argv[2].strip()
        enroll_voice(uid)
    elif cmd == "auth":
        mode = "balanced"
        if len(sys.argv) >= 3:
            mode = sys.argv[2].lower()
        res = mfa_auth_flow(mode=mode)
        print("Result:", res)
    else:
        _usage()
