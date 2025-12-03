import sounddevice as sd
import numpy as np
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE, "data", "voice")
os.makedirs(VOICE_DIR, exist_ok=True)

FS = 16000
DUR = 2.2

def enroll_voice(user_id):
    print(f"[Voice Enroll] User = {user_id}")

    samples = []

    for i in range(3):
        input(f"Press ENTER then speak sample #{i+1} ...")
        print("Recording...")

        rec = sd.rec(int(FS * DUR), FS, 1, dtype=np.float32)
        sd.wait()
        rec = rec.flatten()

        rms = np.sqrt(np.mean(rec**2))
        if rms < 0.003:
            print("Too quiet. Try again.")
            continue

        samples.append(rec)
        print("✓ Sample saved")

    if len(samples) == 0:
        print("FAILED: no usable audio.")
        return

    ffts = [np.abs(np.fft.rfft(s)) for s in samples]
    emb = np.mean(np.array(ffts), axis=0)

    np.save(os.path.join(VOICE_DIR, f"{user_id}.npy"), emb)
    print("Saved voice embedding.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voice_enroll_fft.py <user_id>")
        sys.exit(1)
    enroll_voice(sys.argv[1])
