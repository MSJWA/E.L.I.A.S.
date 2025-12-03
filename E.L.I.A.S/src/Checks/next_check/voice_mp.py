import sounddevice as sd
import numpy as np
import librosa
import os, time
from utils import VOICE_DIR, save_embedding, log_event, load_embedding

SR = 16000
ENROLL_SECONDS = 2.5

def record_audio(seconds=ENROLL_SECONDS, sr=SR):
    print(f"Recording {seconds}s ...")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return rec.flatten()

def extract_mfcc_embedding(wav, sr=SR, n_mfcc=20):
    wav = wav.astype(float)
    # remove silence by RMS threshold
    rms = librosa.feature.rms(y=wav).mean()
    if rms < 1e-4:
        return None
    mf = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    # mean across time frames -> fixed size vector
    return np.mean(mf, axis=1)

def enroll_voice(user_id, repeats=3):
    embs = []
    for i in range(repeats):
        input(f"Press Enter and speak sample #{i+1} (short phrase) ...")
        wav = record_audio()
        emb = extract_mfcc_embedding(wav)
        if emb is None:
            print("Very low volume / silence detected. Try again.")
            continue
        embs.append(emb)
        print("Recorded.")
    if not embs:
        print("No usable voice samples.")
        return False
    mean_emb = np.mean(np.stack(embs), axis=0)
    path = os.path.join(VOICE_DIR, f"{user_id}.npy")
    save_embedding(path, mean_emb)
    log_event("voice_enroll", f"user {user_id} enrolled", {"samples": len(embs)})
    print(f"Saved voice embedding: {path}")
    return True

def verify_voice_for_user(user_id, wav=None, threshold=0.65):
    path = os.path.join(VOICE_DIR, f"{user_id}.npy")
    if not os.path.exists(path):
        return {"ok": False, "reason": "no_voice_registered"}
    known = load_embedding(path)
    if wav is None:
        wav = record_audio()
    emb = extract_mfcc_embedding(wav)
    if emb is None:
        return {"ok": False, "reason": "low_volume"}
    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    score = float(cosine_similarity([emb], [known])[0][0])
    return {"ok": score >= threshold, "score": score}
