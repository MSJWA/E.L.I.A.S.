# src/test_cam.py
from camera_utils import open_camera
import cv2
def test_camera():
    cap, backend = open_camera(preferred=("MSMF","ANY","DSHOW"), set_resolution=(640,480), warmup_frames=3)
    print("Backend used:", backend)
    if cap is None:
        print("No camera.")
        return
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        print("OK — saving test_capture.jpg")
        cv2.imwrite("test_capture.jpg", frame)
    else:
        print("Frame read failed.")
if __name__ == "__main__":
    test_camera()
