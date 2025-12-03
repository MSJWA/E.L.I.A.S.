# src/vision_liveness.py
import cv2
import numpy as np

ROI_WIDTH = 160
FLOW_FRAMES = 4
FLOW_SPACING_MS = 100

def collect_face_roi_frames(cap, bbox, n_frames=FLOW_FRAMES, spacing_ms=FLOW_SPACING_MS, roi_width=ROI_WIDTH):
    x, y, w, h = bbox
    frames = []
    cx = x + w//2
    cy = y + h//2
    half = roi_width // 2
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w_img, x0 + roi_width)
        y1 = min(h_img, y0 + roi_width)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        roi_resized = cv2.resize(roi, (roi_width, roi_width))
        frames.append(roi_resized)
        cv2.waitKey(spacing_ms)
    return frames

def optical_flow_liveness(frames):
    try:
        total = 0.0
        count = 0
        for i in range(len(frames)-1):
            f1 = frames[i]
            f2 = frames[i+1]
            flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            total += mag.mean()
            count += 1
        score = total / (count or 1)
        return float(1.0 - np.exp(-score/2.5))
    except Exception:
        return 0.0

def lbp_liveness_one_frame(frame):
    try:
        def lbp_image(img):
            h, w = img.shape
            out = np.zeros_like(img)
            for yy in range(1, h-1):
                for xx in range(1, w-1):
                    c = img[yy,xx]
                    code = 0
                    code |= (1<<0) if img[yy-1,xx-1] >= c else 0
                    code |= (1<<1) if img[yy-1,xx] >= c else 0
                    code |= (1<<2) if img[yy-1,xx+1] >= c else 0
                    code |= (1<<3) if img[yy,xx+1] >= c else 0
                    code |= (1<<4) if img[yy+1,xx+1] >= c else 0
                    code |= (1<<5) if img[yy+1,xx] >= c else 0
                    code |= (1<<6) if img[yy+1,xx-1] >= c else 0
                    code |= (1<<7) if img[yy,xx-1] >= c else 0
                    out[yy,xx] = code
            return out
        lbp = lbp_image(frame)
        var = float(lbp.var())
        return float(1.0 - np.exp(-var / 500.0))
    except Exception:
        return 0.0

def aggregate_liveness(scores: dict):
    optical = float(scores.get("optical", 0.0))
    lbp = float(scores.get("lbp", 0.0))
    blink = float(scores.get("blink", 0.0))
    agg = 0.6*optical + 0.35*lbp + 0.05*blink
    return float(max(0.0, min(1.0, agg)))
