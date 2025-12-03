# src/camera_utils.py
import cv2
from typing import Optional, Tuple

# Backends map
_BACKENDS = {
    "MSMF": getattr(cv2, "CAP_MSMF", None),
    "DSHOW": getattr(cv2, "CAP_DSHOW", None),
    "VFW": getattr(cv2, "CAP_VFW", None),
    "ANY": getattr(cv2, "CAP_ANY", None),
}

def _try_open(index: int, backend) -> Optional[cv2.VideoCapture]:
    try:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap
    except Exception:
        return None

def open_camera(
    index: int = 0,
    preferred: Tuple[str, ...] = ("MSMF", "ANY", "DSHOW"),
    set_resolution: Tuple[int,int] = (1280, 720),
    warmup_frames: int = 3,
    exposure_tweak: bool = False
) -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    Try to open camera with preferred backends, set resolution and warm up.
    Returns (cap, backend_name) or (None, "").
    """
    for name in preferred:
        backend_const = _BACKENDS.get(name)
        cap = _try_open(index, backend_const)
        if cap is None:
            continue

        w, h = set_resolution
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except Exception:
            pass

        if exposure_tweak:
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            except Exception:
                pass

        ok = False
        for _ in range(max(1, warmup_frames)):
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    if frame.mean() > 15:
                        ok = True
                        break
                except Exception:
                    ok = True
                    break

        if ok:
            return cap, name

        try:
            cap.release()
        except Exception:
            pass

    # fallback
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, "RAW"
    except Exception:
        pass

    return None, ""
