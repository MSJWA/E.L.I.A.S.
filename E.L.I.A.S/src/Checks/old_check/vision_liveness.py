# src/vision_liveness.py
"""
Lightweight liveness / anti-spoof checks for demo use.

Contains:
 - collect_face_roi_frames(cap, bbox, n_frames, spacing_ms, roi_w)
 - optical_flow_liveness(frames_gray)
 - lbp_liveness_one_frame(gray_roi)
 - aggregate_liveness(scores, weights)

Design goals:
 - Fast on commodity laptops (small ROI e.g. 160px wide)
 - Low memory
 - Deterministic heuristics (no heavy ML)
 - Tunable thresholds for demo
"""

from __future__ import annotations
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional

# ------------------ Config (tune as needed) ------------------
ROI_WIDTH = 160             # width in pixels to resize ROI for analysis
FLOW_FRAMES = 4             # number of frames to sample for optical flow
FLOW_SPACING_MS = 180       # ms between sampled frames
OPTICAL_FLOW_ALPHA = 8.0    # mapping parameter for score
LBP_P = 8
LBP_R = 1

# ------------------ Helpers ------------------
def _resize_roi(frame: np.ndarray, bbox: Tuple[int,int,int,int], width: int = ROI_WIDTH) -> np.ndarray:
    """Crop bbox from frame and resize to given width while preserving aspect ratio."""
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return roi
    h0, w0 = roi.shape[:2]
    scale = width / float(w0)
    new_h = max(8, int(h0 * scale))
    roi_small = cv2.resize(roi, (width, new_h), interpolation=cv2.INTER_LINEAR)
    return roi_small


def collect_face_roi_frames(cap: cv2.VideoCapture,
                            bbox: Tuple[int,int,int,int],
                            n_frames: int = FLOW_FRAMES,
                            spacing_ms: int = FLOW_SPACING_MS,
                            roi_width: int = ROI_WIDTH) -> List[np.ndarray]:
    """
    Capture `n_frames` small grayscale ROI frames sampled from the camera.
    Returns list of grayscale numpy arrays (small).
    Works with an already-opened VideoCapture object (cap).
    """
    frames = []
    start = time.time()
    for i in range(n_frames):
        # grab current frame (non-blocking read)
        ret, frame = cap.read()
        if not ret:
            # short sleep and retry once
            time.sleep(0.05)
            ret, frame = cap.read()
            if not ret:
                break
        roi = _resize_roi(frame, bbox, width=roi_width)
        if roi is None or roi.size == 0:
            break
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        # spacing
        if i < n_frames - 1:
            time.sleep(spacing_ms / 1000.0)
    return frames


def optical_flow_liveness(frames_gray: List[np.ndarray]) -> float:
    """
    Compute an optical-flow-based motion score for small grayscale frames.
    Returns score in [0.0, 1.0] (higher => more likely live).
    Implementation: Farneback dense flow, median magnitude per frame pair, map via 1-exp(-alpha*mag).
    """
    if not frames_gray or len(frames_gray) < 2:
        return 0.0

    total_med = 0.0
    count = 0
    for i in range(len(frames_gray) - 1):
        a = frames_gray[i]
        b = frames_gray[i + 1]
        # ensure same size
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        # parameters chosen for speed on small ROI
        flow = cv2.calcOpticalFlowFarneback(a, b, None,
                                            pyr_scale=0.5, levels=2,
                                            winsize=11, iterations=2,
                                            poly_n=5, poly_sigma=1.1, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        med = float(np.median(mag))
        total_med += med
        count += 1

    avg_med = total_med / max(1, count)
    # Map average median magnitude to 0..1
    score = 1.0 - np.exp(- (avg_med * OPTICAL_FLOW_ALPHA))
    # Clamp
    score = float(np.clip(score, 0.0, 1.0))
    return score


def _uniform_lbp(image: np.ndarray, P: int = LBP_P, R: int = LBP_R) -> np.ndarray:
    """
    Simple uniform LBP implementation (returns label image).
    Lightweight version optimized for small ROI.
    """
    # pad image to avoid border issues
    img = image.astype(np.uint8)
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(R, h - R):
        for x in range(R, w - R):
            center = int(img[y, x])
            code = 0
            for p in range(P):
                theta = 2.0 * np.pi * p / P
                rx = x + int(round(R * np.cos(theta)))
                ry = y - int(round(R * np.sin(theta)))
                neighbor = int(img[ry, rx])
                code = (code << 1) | (1 if neighbor >= center else 0)
            out[y, x] = code
    return out


def lbp_histogram_from_gray(image_gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute a normalized histogram of LBP codes from a small grayscale image.
    Bins default to 256 (P up to 8 gives <=256 codes).
    """
    if image_gray is None or image_gray.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    lbp = _uniform_lbp(image_gray)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist.astype(np.float32)


def lbp_liveness_one_frame(gray_roi: np.ndarray, ref_hist: Optional[np.ndarray] = None) -> float:
    """
    Compute similarity between current ROI LBP hist and reference LBP hist (if provided).
    If no ref_hist is supplied, return a moderate default based on variance (heuristic).
    Returns similarity in [0..1].
    """
    n_bins = 64  # compact histogram for speed
    hist = lbp_histogram_from_gray(gray_roi, bins=n_bins)
    if ref_hist is None or ref_hist.shape[0] != n_bins:
        # fallback heuristic: measure texture energy (variance)
        v = float(np.var(gray_roi) / 255.0)
        return float(np.clip(v, 0.0, 1.0))
    # compute correlation similarity (pearson)
    try:
        corr = np.corrcoef(hist, ref_hist)[0, 1]
        # corr in [-1,1] -> map to [0,1]
        sim = float((corr + 1.0) / 2.0)
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0


def aggregate_liveness(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Combine individual liveness component scores into single [0..1] score.
    Default weights favor optical flow.
    """
    default_weights = {"optical": 0.5, "lbp": 0.25, "blink": 0.2}
    if weights is None:
        weights = default_weights
    num = 0.0
    den = 0.0
    for k, v in scores.items():
        w = weights.get(k, 0.0)
        num += w * float(v)
        den += w
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


# ------------------ Quick demo helper (not blocking) ------------------
def demo_live_check_from_bbox(video_index: int = 0,
                              bbox: Tuple[int,int,int,int] = None,
                              ref_lbp_hist: Optional[np.ndarray] = None) -> Dict:
    """
    High-level helper to run a quick liveness check from camera.
    bbox must be (x,y,w,h); if None, function will detect face using Haar cascade (fast).
    Returns dict: {'optical':.., 'lbp':.., 'aggregate':..}
    """
    cap = cv2.VideoCapture(video_index)
    if not cap.isOpened():
        return {"error": "camera_unavailable"}

    # if bbox not provided, do a single quick detect with built-in cascade
    if bbox is None:
        # attempt to find a face quickly
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "frame_failed"}
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_full, 1.1, 4, minSize=(80, 80))
        if len(faces) == 0:
            cap.release()
            return {"error": "no_face"}
        bbox = faces[0]

    # collect frames
    frames = collect_face_roi_frames(cap, bbox, n_frames=FLOW_FRAMES, spacing_ms=FLOW_SPACING_MS, roi_width=ROI_WIDTH)
    cap.release()
    if not frames:
        return {"error": "no_frames"}

    # optical score
    optical = optical_flow_liveness(frames)

    # LBP on first frame
    lbp_score = lbp_liveness_one_frame(frames[0], ref_hist=ref_lbp_hist)

    agg = aggregate_liveness({"optical": optical, "lbp": lbp_score, "blink": 0.0})
    return {"optical": optical, "lbp": lbp_score, "aggregate": agg}


# ------------------ Self-test quick run ------------------
if __name__ == "__main__":
    print("Quick liveness demo (open camera). Press Ctrl+C to stop.")
    r = demo_live_check_from_bbox()
    print(r)
