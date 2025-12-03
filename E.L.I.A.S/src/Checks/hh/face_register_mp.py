import cv2
import mediapipe as mp
import numpy as np
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(BASE, "data", "face")
os.makedirs(FACE_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh

def register_face(user_id):
    print(f"[Face Register] User = {user_id}")

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    frames = []

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        while len(frames) < 6:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if not res.multi_face_landmarks:
                cv2.imshow("Register Face", frame)
                cv2.waitKey(1)
                continue

            lm = res.multi_face_landmarks[0]

            vec = []
            for p in lm.landmark:
                vec.append([p.x, p.y, p.z])

            vec = np.array(vec).flatten()
            frames.append(vec)
            print(f"Captured {len(frames)}/6")

            cv2.imshow("Register Face", frame)
            cv2.waitKey(150)

    cap.release()
    cv2.destroyAllWindows()

    emb = np.mean(np.array(frames), axis=0)
    out = os.path.join(FACE_DIR, f"{user_id}.npy")
    np.save(out, emb)

    print(f"Saved face embedding: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_register_mp.py <user_id>")
        sys.exit(1)

    register_face(sys.argv[1])
