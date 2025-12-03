import cv2
import mediapipe as mp
import numpy as np
import os

mp_face = mp.solutions.face_mesh


def get_face_embedding(frames_count=5):
    """Captures video and returns an averaged face embedding vector."""
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0, cv2.CAP_MSMF) # Windows fix if needed

    frames = []
    print("   [Face] Looking for face...")

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        while len(frames) < frames_count:
            ret, frame = cap.read()
            if not ret: continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                frames.append(vec)

                # Visual feedback
                cv2.putText(frame, f"Captured: {len(frames)}/{frames_count}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Scanner", frame)
            if cv2.waitKey(50) & 0xFF == 27: break  # ESC to quit

    cap.release()
    cv2.destroyAllWindows()

    if not frames: return None
    return np.mean(np.array(frames), axis=0)


def save_face(user_id, base_path):
    emb = get_face_embedding(frames_count=6)
    if emb is not None:
        path = os.path.join(base_path, "face", f"{user_id}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, emb)
        return True
    return False


def authenticate_face(base_path):
    """Scans face and finds best match in data folder."""
    target_emb = get_face_embedding(frames_count=3)
    if target_emb is None: return None, 0.0

    face_dir = os.path.join(base_path, "face")
    if not os.path.exists(face_dir): return None, 0.0

    best_id = None
    best_score = -1.0

    for f in os.listdir(face_dir):
        if f.endswith(".npy"):
            saved_emb = np.load(os.path.join(face_dir, f))
            score = np.dot(target_emb, saved_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(saved_emb))
            if score > best_score:
                best_score = score
                best_id = f.replace(".npy", "")

    return best_id, best_score