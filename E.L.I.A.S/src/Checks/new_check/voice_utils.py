import os
import json
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")
LOG_DIR = os.path.join(BASE_DIR, "logs", "voice")

if not os.path.exists(VOICE_DIR):
    os.makedirs(VOICE_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

SAMPLE_RATE = 16000
ENROLL_SECONDS = 2.5


def log_voice_event(message):
    path = os.path.join(LOG_DIR, "voice_log.txt")
    with open(path, "a") as f:
        f.write(message + "\n")


# ---------------------------------------------------------
# NEW SPEECH DETECTION — fully laptop-mic compatible
# ---------------------------------------------------------
def zero_crossing_rate(arr, frame_size=400):
    if len(arr) < frame_size:
        return 0
    frames = []
    for i in range(0, len(arr) - frame_size, frame_size):
        frame = arr[i:i + frame_size]
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
        frames.append(zcr)
    return float(np.mean(frames))


def spectral_energy(arr):
    if len(arr) == 0:
        return 0.0
    S = np.abs(np.fft.rfft(arr))
    return float(np.mean(S))


# ---------------------------------------------------------
# MAIN RECORDING FUNCTION (MIC-SAFE)
# ---------------------------------------------------------
def record_to_array(duration=3.5, fs=SAMPLE_RATE, channels=1):
    print(f"Recording {duration:.2f}s ...")
    data = sd.rec(int(duration * fs), samplerate=fs,
                  channels=channels, dtype='float32')
    sd.wait()

    arr = np.squeeze(np.array(data)).astype('float32')

    if arr.size == 0:
        return arr

    # Pre-emphasis (clarity)
    arr = lfilter([1, -0.97], [1], arr)

    # ---- NEW: speech detection ----
    zcr = zero_crossing_rate(arr)
    spec = spectral_energy(arr)

    # Laptop microphones → heavy suppression → use LOW thresholds
    VOICED = (zcr > 0.015) or (spec > 0.002)

    if not VOICED:
        print("⚠️ Warning: Very low speech energy, but allowing it anyway (laptop mic mode).")
        return arr

    return arr


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def spectral_embedding(arr):
    if arr.size == 0:
        return np.zeros(40).tolist()

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
    path = os.path.join(VOICE_DIR, f"voice_{user_id}.json")
    with open(path, "w") as f:
        json.dump({"embedding": embedding}, f)
    return True


def load_known_embedding(user_id="admin"):
    path = os.path.join(VOICE_DIR, f"voice_{user_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("embedding")


def compare_embeddings(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))
