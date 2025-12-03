import cv2
import os
import numpy as np
import time

# Force MSMF backend (your laptop supports this)
BACKEND = cv2.CAP_MSMF

def register_user(user_id):
    print(f"--- AUTO FACE REGISTRATION FOR USER '{user_id}' ---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    user_dir   = os.path.join(base_dir, "data", "auth", "faces")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(user_dir, exist_ok=True)

    xml_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
    save_path = os.path.join(user_dir, f"{user_id}.jpg")

    # Load detector
    face_cascade = cv2.CascadeClassifier(xml_path)

    print("Opening camera (MSMF backend)...")
    cap = cv2.VideoCapture(0, BACKEND)

    if not cap.isOpened():
        print("❌ ERROR: Camera could not be opened.")
        return False

    best_face = None
    best_size = 0
    attempts = 30   # capture up to 30 frames to find a clean face

    print("Please look at the camera... Capturing best frame automatically.")

    for _ in range(attempts):
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) != 1:
            continue

        (x, y, w, h) = faces[0]
        face_area = w * h

        # Pick the largest, clearest face
        if face_area > best_size:
            best_size = face_area
            best_face = gray[y:y + h, x:x + w]

        time.sleep(0.05)

    cap.release()

    if best_face is None:
        print("❌ No valid face detected after scanning.")
        return False

    cv2.imwrite(save_path, best_face)
    print(f"✅ Face registered successfully → {save_path}")

    return True


if __name__ == "__main__":
    uid = input("Enter User ID to register: ").strip()
    register_user(uid)
