import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    print("Resolution:", frame.shape)
