# src/voice_enroll.py
import random
from voice_utils import record_to_array, spectral_embedding, save_embedding, log_voice_event

CHALLENGES = [
    "blue river",
    "quiet forest",
    "silver moon",
    "hidden valley",
    "gentle sunrise",
    "orange lantern",
    "silent horizon"
]

def enroll_user_voice(user_id="user_001"):
    print("=== VOICE ENROLLMENT ===")
    phrase = random.choice(CHALLENGES)
    print(f"Enrollment phrase: >>> {phrase} <<<")
    print("You will be asked to speak the phrase 3 times consistently.")
    embs = []
    for i in range(1,4):
        input(f"Press Enter then SAY sample #{i}")
        arr = record_to_array(duration=3.5)
        emb = spectral_embedding(arr)
        embs.append(emb)
    # average
    avg = [float(sum(x)/3.0) for x in zip(*embs)]
    save_embedding(user_id, avg)
    log_voice_event(f"enrolled_voice_{user_id}")
    print("Enrollment saved.")
