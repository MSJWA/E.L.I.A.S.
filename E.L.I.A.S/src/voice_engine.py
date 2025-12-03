import sounddevice as sd
import numpy as np
import os

FS = 16000
DUR = 3.0


def record_and_fft():
    rec = sd.rec(int(FS * DUR), samplerate=FS, channels=1, dtype=np.float32)
    sd.wait()
    rec = rec.flatten()

    # Check volume
    if np.max(np.abs(rec)) < 0.01: return None

    # Return FFT magnitude
    return np.abs(np.fft.rfft(rec))


def save_voice(user_id, base_path):
    print(f"   [Voice] Speak naturally for 3 seconds...")
    emb = record_and_fft()

    if emb is not None:
        path = os.path.join(base_path, "voice", f"{user_id}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, emb)
        return True
    return False


def authenticate_voice(user_id, base_path):
    path = os.path.join(base_path, "voice", f"{user_id}.npy")
    if not os.path.exists(path): return False

    print(f"   [Voice] Verifying user '{user_id}'... Speak now.")
    incoming_fft = record_and_fft()

    if incoming_fft is None: return False

    saved_fft = np.load(path)
    score = np.dot(incoming_fft, saved_fft) / (np.linalg.norm(incoming_fft) * np.linalg.norm(saved_fft))

    print(f"   [Voice Score]: {score:.3f}")
    return score > 0.50  # Threshold