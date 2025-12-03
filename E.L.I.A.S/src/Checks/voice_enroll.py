# voice_enroll.py
# Lightweight MFCC-based voice enrollment: stores mean MFCC vector per user
import os, sys, numpy as np, sounddevice as sd, librosa, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_VOICE_DIR = os.path.join(BASE, "data", "voice")
os.makedirs(DATA_VOICE_DIR, exist_ok=True)

SAMPLE_RATE = 22050
ENROLL_SECONDS = 2.5

def record(seconds=2.5):
    print(f"Recording {seconds:.2f}s ... speak now")
    data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return np.squeeze(data).astype(np.float32)

def embed_from_audio(waveform, sr=SAMPLE_RATE):
    # compute MFCCs and return mean vector
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
    # optionally add deltas
    delta = librosa.feature.delta(mfcc)
    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])
    # normalize
    feat = feat / (np.linalg.norm(feat)+1e-10)
    return feat

def enroll(user_id, rounds=3):
    embeddings = []
    for i in range(rounds):
        input(f"Press Enter then speak sample #{i+1} (duration {ENROLL_SECONDS}s)...")
        arr = record(ENROLL_SECONDS)
        emb = embed_from_audio(arr)
        embeddings.append(emb)
        print("Sample saved")
        time.sleep(0.5)
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    outp = os.path.join(DATA_VOICE_DIR, f"{user_id}.npz")
    np.savez(outp, emb=mean_emb)
    print("Saved admin voice embedding to:", outp)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/voice_enroll.py <user_id>")
        sys.exit(1)
    enroll(sys.argv[1])
