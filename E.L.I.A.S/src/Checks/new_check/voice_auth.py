import time
import random
import numpy as np

from voice_utils import (
    record_to_array,
    spectral_embedding,
    load_known_embedding,
    compare_embeddings,
    log_voice_event,
    SAMPLE_RATE
)


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
CHALLENGE_PHRASES = [
    "blue river",
    "quiet forest",
    "silver moon",
    "hidden valley",
    "gentle sunrise",
    "orange lantern",
    "silent horizon"
]

MODE_THRESHOLDS = {
    "strict": 0.82,
    "balanced": 0.72,
    "relaxed": 0.60
}


# -----------------------------------------------------------
# MAIN AUTHENTICATION FUNCTION
# -----------------------------------------------------------
def authenticate_voice(mode="balanced", user_id="admin"):

    if mode not in MODE_THRESHOLDS:
        print(f"Unknown mode '{mode}', using 'balanced'.")
        mode = "balanced"

    threshold = MODE_THRESHOLDS[mode]

    print("\n=== VOICE AUTHENTICATION ===")

    # 1. Challenge phrase
    phrase = random.choice(CHALLENGE_PHRASES)

    print(f"🔐 Challenge phrase: >>>  {phrase}  <<<")
    print("Speak clearly when recording starts.")
    input("Press Enter to start recording...")

    # 2. Record audio (using new laptop-safe logic)
    audio = record_to_array(duration=3.5)

    if audio.size == 0:
        print("❌ ERROR: No audio captured.")
        return {"success": False, "score": 0.0}

    # 3. Convert to embedding
    emb = spectral_embedding(audio)

    # 4. Load admin/master embedding
    known = load_known_embedding(user_id=user_id)

    if known is None:
        print("❌ No enrolled user voice found. Register first.")
        log_voice_event("AUTH FAILED — no registered user")
        return {"success": False, "score": 0.0}

    # 5. Similarity score
    score = compare_embeddings(emb, known)
    score = float(score)

    print(f"\nSimilarity score: {score:.3f} (threshold {threshold})")

    # 6. Decision
    if score >= threshold:
        print("✅ ACCESS GRANTED (voice match)")
        log_voice_event(f"AUTH SUCCESS score={score:.3f}")
        return {"success": True, "score": score}

    else:
        print("❌ ACCESS DENIED (insufficient voice match)")
        log_voice_event(f"AUTH FAIL score={score:.3f}")
        return {"success": False, "score": score}


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    mode = input("Mode (strict/balanced/relaxed): ").strip().lower()

    if mode == "":
        mode = "balanced"

    result = authenticate_voice(mode=mode)

    print("\nFinal result:", result)
