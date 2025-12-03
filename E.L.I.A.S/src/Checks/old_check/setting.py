# src/camera_set_and_test.py
import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try CAP_DSHOW on Windows; remove on some machines

# Try to set to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame.")
else:
    print("Captured frame shape:", frame.shape)   # (h, w, 3)
    # Save a sample image so you can inspect it
    out = "camera_test_snapshot.jpg"
    cv2.imwrite(out, frame)
    print("Saved snapshot to:", out)
