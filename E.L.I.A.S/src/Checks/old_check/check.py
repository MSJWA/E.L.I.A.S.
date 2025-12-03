import cv2

def test(idx, backend_name, api):
    print(f"\n--- Testing {backend_name} ---")
    cap = cv2.VideoCapture(idx, api)
    ret, frame = cap.read()
    print("Opened:", cap.isOpened(), " Read:", ret)
    if ret:
        print("Frame mean value:", frame.mean())
        cv2.imwrite(f"test_{backend_name}.jpg", frame)
        print("Saved test image.")
    cap.release()

# Test all major Windows backends
test(0, "CAP_DSHOW", cv2.CAP_DSHOW)
test(0, "CAP_MSMF", cv2.CAP_MSMF)
test(0, "CAP_VFW", cv2.CAP_VFW)
test(0, "CAP_ANY", cv2.CAP_ANY)

# Test different camera indices
for i in range(1, 4):
    print(f"\n--- Trying camera index {i} ---")
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    print("Opened:", cap.isOpened(), " Read:", ret)
    if ret:
        print("Frame mean:", frame.mean())
        cv2.imwrite(f"test_index_{i}.jpg", frame)
    cap.release()
