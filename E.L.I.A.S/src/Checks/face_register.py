# face_register.py
# capture N frames, compute embeddings (average) and save per user
import cv2, os, time, sys, numpy as np
from face_embedding_core import detect_and_align, image_to_embedding, save_embedding, DATA_FACE_DIR
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def capture_embeddings(user_id, n_capture=6, device="cpu", cam_backend=cv2.CAP_MSMF):
    cap = cv2.VideoCapture(0, apiPreference=cam_backend)
    if not cap.isOpened():
        # fallback to ANY
        cap = cv2.VideoCapture(0)
    print("Look at camera. Capturing", n_capture, "good frames.")
    embeddings = []
    tries = 0
    while len(embeddings) < n_capture and tries < n_capture * 6:
        ret, frame = cap.read()
        tries += 1
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        aligned, bbox = detect_and_align(frame)
        if aligned is None:
            cv2.putText(frame, "No face detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            cv2.imshow("Register", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
        emb = image_to_embedding(aligned, device=device)
        embeddings.append(emb)
        cv2.putText(frame, f"Captured {len(embeddings)}/{n_capture}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Register", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    if len(embeddings) == 0:
        print("No captures")
        return False
    # average embeddings (mean then normalize)
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb)+1e-10)
    save_embedding(user_id, mean_emb)
    print(f"Saved embedding for {user_id}: data/face/{user_id}.npy")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/face_register.py <user_id>")
        sys.exit(1)
    uid = sys.argv[1]
    capture_embeddings(uid)
