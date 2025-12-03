# src/voice_utils.py
import os
import json
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter
from utils import user_voice_path, log_event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")
os.makedirs(VOICE_DIR, exist_ok=True)

SAMPLE_RATE = 16000

def log_voice_event(msg):
    log_event("voice", msg)

def zero_crossing_rate(arr, frame_size=400):
    if len(arr) < frame_size:
        return 0.0
    frames = []
    for i in range(0, len(arr) - frame_size, frame_size):
        frame = arr[i:i + frame_size]
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        frames.append(zcr)
    return float(np.mean(frames)) if frames else 0.0

def spectral_energy(arr):
    if len(arr) == 0:
        return 0.0
    S = np.abs(np.fft.rfft(arr))
    return float(np.mean(S))

def record_to_array(duration=3.5, fs=SAMPLE_RATE, channels=1):
    print(f"Recording {duration:.2f}s ...")
    data = sd.rec(int(duration * fs), samplerate=fs,
                  channels=channels, dtype='float32')
    sd.wait()
    arr = np.squeeze(np.array(data)).astype('float32')
    if arr.size == 0:
        return arr
    # pre-emphasis
    arr = lfilter([1, -0.97], [1], arr)
    zcr = zero_crossing_rate(arr)
    spec = spectral_energy(arr)
    VOICED = (zcr > 0.015) or (spec > 0.002)
    if not VOICED:
        # allow anyway (laptop mic) but warn
        print("⚠️ Low detected by energy heuristics (allowing due to laptop mic).")
        log_voice_event("low_energy_but_allowed")
    return arr

def spectral_embedding(arr):
    if arr.size == 0:
        return [0.0]*40
    N = 2048
    if len(arr) < N:
        arr = np.pad(arr, (0, N - len(arr)), mode='constant')
    else:
        arr = arr[:N]
    spec = np.abs(np.fft.rfft(arr))
    spec = spec / (np.max(spec) + 1e-6)
    bins = np.array_split(spec, 40)
    return [float(np.mean(b)) for b in bins]

def save_embedding(user_id, embedding):
    path = user_voice_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding}, f)
    log_voice_event(f"saved_emb_{user_id}")
    return True

def load_known_embedding(user_id="admin"):
    path = user_voice_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("embedding")

def compare_embeddings(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a); b = np.array(b)
    na = a / (np.linalg.norm(a) + 1e-6)
    nb = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(na, nb))
