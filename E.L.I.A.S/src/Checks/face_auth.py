# face_auth.py
# Realtime authentication using saved embeddings + multi-frame voting + blink liveness
import time, os, numpy as np, cv2
from face_embedding_core import detect_and_align, image_to_embedding, load_embedding, cosine_sim
import mediapipe as mp

# Config
MIN_SCORE = 0.55    # cosine similarity threshold (0..1) - tune this
VOTES_REQUIRED = 3  # out of TRIALS
TRIALS = 5
EYE_BLINK_THRESHOLD = 0.20  # EAR threshold for blink detection (tune)

mp_mesh = mp.solutions.face_mesh

def eye_aspect_ratio(landmarks, left_indices, right_indices, w, h):
    # landmarks in normalized coords list of (x,y)
    # compute simple vertical/horizontal ratio for eyes using mediapipe mesh indices
    def ear(indices):
        pts = [(int(landmarks[i][0]*w), int(landmarks[i][1]*h)) for i in indices]
        # we approximate: vertical distance / horizontal distance
        v = np.linalg.norm(np.array(pts[1]) - np.array(pts[5])) + np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        hdist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-6
        return v / hdist
    return ear(left_indices), ear(right_indices)

def detect_blink_sequence(cap, timeout=3.0):
    # attempt a simple blink check using face_mesh points
    start = time.time()
    blinked = False
    with mp_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret: break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(img)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            # simple map - use standard eye indices approximate (mediapipe refined indices recommended)
            lm_xy = [(p.x, p.y) for p in lm]
            h,w = frame.shape[:2]
            # left eye: select a set of indices; these approximate the eyelid points
            left_idx = [33, 160, 158, 133, 153, 144]   # approximate
            right_idx = [263, 387, 385, 362, 380, 373]
            left_ear, right_ear = eye_aspect_ratio(lm_xy, left_idx, right_idx, w, h)
            ear = min(left_ear, right_ear)
            if ear < EYE_BLINK_THRESHOLD:
                blinked = True
                break
    return blinked

def authenticate(allowed_user_id=None, device="cpu"):
    # load all stored embeddings if allowed_user_id is None else test only that user
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "face")
    ids = []
    embeddings = []
    for f in os.listdir(data_dir):
        if not f.endswith(".npy"): continue
        uid = f.replace(".npy","")
        ids.append(uid)
        embeddings.append(np.load(os.path.join(data_dir, f)))
    if len(embeddings)==0:
        print("No enrolled faces.")
        return {"ok": False, "reason": "no_enroll"}
    cap = cv2.VideoCapture(0)
    votes = {}
    trials = TRIALS
    print("Starting face auth. Please look at camera.")
    for t in range(trials):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue
        aligned, bbox = detect_and_align(frame)
        if aligned is None:
            print("No face")
            continue
        emb = image_to_embedding(aligned, device=device)
        best_score = -1.0
        best_id = None
        for uid, known in zip(ids, embeddings):
            s = cosine_sim(known, emb)
            if s > best_score:
                best_score = s
                best_id = uid
        print(f"Frame {t+1}/{trials} -> best {best_id} score={best_score:.3f}")
        if best_score >= MIN_SCORE:
            votes[best_id] = votes.get(best_id, 0) + 1
        time.sleep(0.08)
    cap.release()
    # apply majority rule
    if not votes:
        print("No confident votes")
        return {"ok": False, "reason": "no_match"}
    winner, v = max(votes.items(), key=lambda kv: kv[1])
    # liveness blink check now
    cap2 = cv2.VideoCapture(0)
    blinked = detect_blink_sequence(cap2, timeout=3.0)
    cap2.release()
    print("Blink detected:", blinked)
    if v >= VOTES_REQUIRED and blinked:
        return {"ok": True, "user": winner, "score_votes": v}
    else:
        return {"ok": False, "reason": "insufficient_votes_or_no_blink", "votes": votes}

if __name__ == "__main__":
    print("Test face auth")
    print(authenticate())
