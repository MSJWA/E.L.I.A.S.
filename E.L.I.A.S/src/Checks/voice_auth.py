# voice_auth.py
# Lightweight voice verification using MFCC mean vector + cosine similarity
import os, numpy as np, sounddevice as sd, librosa, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_VOICE_DIR = os.path.join(BASE, "data", "voice")

SAMPLE_RATE = 22050
AUTH_SECONDS = 2.5
SIM_THRESHOLD = 0.70  # voice similarity threshold (tune)

def record(seconds=AUTH_SECONDS):
    print(f"Recording {seconds:.2f}s ... speak now")
    data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return np.squeeze(data).astype(np.float32)

def embed_from_audio(waveform, sr=SAMPLE_RATE):
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])
    feat = feat / (np.linalg.norm(feat)+1e-10)
    return feat

def verify(user_id, challenge_phrase=None):
    file = os.path.join(DATA_VOICE_DIR, f"{user_id}.npz")
    if not os.path.exists(file):
        return {"ok": False, "reason": "no_enroll"}
    data = np.load(file)
    known = data["emb"]
    print("Speak the phrase now:")
    input("Press Enter then speak ...")
    arr = record(AUTH_SECONDS)
    emb = embed_from_audio(arr)
    sim = float(np.dot(known, emb) / (np.linalg.norm(known)*np.linalg.norm(emb)+1e-10))
    print("Voice similarity:", sim)
    ok = sim >= SIM_THRESHOLD
    return {"ok": ok, "score": sim}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python src/voice_auth.py <user_id>")
        sys.exit(1)
    print(verify(sys.argv[1]))
