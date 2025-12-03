# src/test_cascade_and_cam.py
import os
import cv2
import sys

# Adjust if your structure differs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "haarcascade_frontalface_default.xml")

print("Project root:", PROJECT_ROOT)
print("Expecting cascade at:", MODEL_PATH)
print()

if not os.path.exists(MODEL_PATH):
    print("❌ File not found at expected path!")
    print("Please move the XML into: ", os.path.join(PROJECT_ROOT, "models"))
    sys.exit(1)

# Print file size and first lines
try:
    size = os.path.getsize(MODEL_PATH)
    print(f"File exists — size: {size} bytes")
    print("First 3 lines of file:")
    with open(MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(3):
            line = f.readline().rstrip()
            print("  ", line)
except Exception as e:
    print("⚠ Could not read file preview:", e)

# Load cascade and check validity
cascade = cv2.CascadeClassifier(MODEL_PATH)
print("Cascade empty() ->", cascade.empty())
print("Cascade loaded OK:", not cascade.empty())
print()

# Quick webcam face detect test (one frame)
print("Attempting to open webcam (index 0)...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    # fallback (some systems)
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam. Check camera permissions or try another index.")
    sys.exit(1)

ret, frame = cap.read()
cap.release()
if not ret or frame is None:
    print("❌ Webcam opened but could not read a frame.")
    sys.exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
print("Faces detected in single frame:", len(faces))
if len(faces) > 0:
    print(" -> Example bbox:", faces[0])
else:
    print(" -> No faces detected. Try moving closer to the camera or increasing lighting.")
