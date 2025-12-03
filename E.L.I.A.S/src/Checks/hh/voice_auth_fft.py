import sounddevice as sd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE, "data", "voice")

FS = 16000
DUR = 3.0

def voice_check(user_id):
    print("Recording voice for authentication...")
    rec = sd.rec(int(FS * DUR), FS, 1, dtype=np.float32)
    sd.wait()
    rec = rec.flatten()

    rms = np.sqrt(np.mean(rec**2))
    if rms < 0.003:
        return False, 0.0

    fft = np.abs(np.fft.rfft(rec))

    path = os.path.join(VOICE_DIR, f"{user_id}.npy")
    if not os.path.exists(path):
        return False, 0.0

    ref = np.load(path)

    score = np.dot(fft, ref) / (np.linalg.norm(fft) * np.linalg.norm(ref))

    return score > 0.70, score
