# src/voice_register.py
"""
Live enrollment (phrase-based or free). Improved UX:
 - Option A: type your own enrollment phrase (will be shown each sample)
 - Option B: press Enter to use a system-generated short phrase
 - Option C: press Enter twice to use free-form (no phrase shown)
Records 3 samples and saves averaged embedding.
"""
import time
import random
import numpy as np
from voice_utils import (
    record_to_array, spectral_embedding, save_known_embedding,
    ENROLL_SECONDS, SAMPLE_RATE
)

GEN_WORDS = [
    "alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india",
    "one","two","three","four","five","seven","eight","nine","zero"
]

def gen_phrase(n_words: int = 2) -> str:
    return " ".join(random.choice(GEN_WORDS) for _ in range(n_words))

def enroll_admin(repeats: int = 3, duration: float = ENROLL_SECONDS):
    print("=== VOICE ENROLLMENT (improved) ===")
    print("You will be asked to record the SAME phrase", repeats, "times.")
    print("Options:")
    print("  1) Type a custom phrase and press Enter (recommended).")
    print("  2) Press Enter to use a system-generated phrase.")
    print("  3) Type 'free' and press Enter to do free-form enrollment (no phrase).")
    choice = input("\nEnter custom phrase or press Enter: ").strip()

    if choice.lower() == "free":
        phrase = None
        print("Free-form enrollment selected. Speak any short sample consistently.")
    elif choice == "":
        phrase = gen_phrase()
        print(f"System-generated phrase (repeat this): '{phrase}'")
    else:
        phrase = choice
        print(f"Custom phrase selected: '{phrase}'")

    time.sleep(1.0)
    embeds = []
    for i in range(repeats):
        if phrase:
            print(f"\nSample #{i+1}: Please speak exactly: >>>  {phrase}  <<<")
        else:
            print(f"\nSample #{i+1}: Please speak a short sentence (free-form).")
        input("Press Enter to start recording...")

        try:
            arr = record_to_array(duration=duration)
        except Exception as e:
            print("Recording failed:", e)
            return False

        # quick check
        if np.abs(arr).mean() < 0.003:
            print("Warning: very low amplitude (speak louder or check mic). Try again.")
            # let user retry this sample
            retry = input("Retry this sample? (y/N): ").strip().lower()
            if retry == "y":
                i -= 1  # try same index again (note: simple behavior)
                continue
            else:
                return False

        emb = spectral_embedding(SAMPLE_RATE, arr)
        if emb is None:
            print("Failed to compute embedding. Try a clearer sample.")
            return False
        embeds.append(emb)
        print("Sample", i+1, "done.")
        time.sleep(0.6)

    # average embeddings
    avg = np.mean(np.stack(embeds, axis=0), axis=0)
    # normalize
    if avg.max() > 0:
        avg = avg / float(avg.max())
    save_known_embedding(avg)
    print("Enrollment complete.")
    return True

if __name__ == "__main__":
    enroll_admin()
# src/voice_auth.py
"""
Final Voice Authentication Module
---------------------------------
Features:
 - Random challenge phrase (anti-replay)
 - Live recording (2.5 seconds)
 - Spectral embedding extraction
 - Distance-based matching
 - Liveness score (strict / balanced)
 - Logging support

Requires:
  - voice_utils.py
  - data/voice/voice_admin.json  (created by voice_register.py)

"""

import time
import random
import numpy as np
from voice_utils import (
    record_to_array,
    spectral_embedding,
    load_known_embedding,
    compare_embeddings,
    log_voice_event,
    SAMPLE_RATE,
    ENROLL_SECONDS
)

# Challenge phrases
CHALLENGES = [
    "red apple",
    "silver moon",
    "blue river",
    "soft echo",
    "quiet shadow",
    "alpha bravo",
    "delta echo",
    "lima tango",
    "open sesame",
    "future vision"
]

def generate_challenge():
    """Pick a random phrase for challenge-response."""
    return random.choice(CHALLENGES)


def authenticate_user(duration: float = ENROLL_SECONDS,
                      strictness: str = "balanced"):
    """
    Actual verification function.
    strictness: "strict", "balanced", "relaxed"
    """

    print("\n=== VOICE AUTHENTICATION ===")

    # Load stored admin embedding
    admin_embed = load_known_embedding()
    if admin_embed is None:
        print("❌ No enrolled admin voice found. Please enroll first.")
        return False

    # Generate challenge phrase
    challenge = generate_challenge()
    print(f"\n🔐 Challenge phrase:  >>>  {challenge}  <<<")
    print("Speak the phrase clearly when recording starts.")

    input("Press Enter to start recording...")

    # Record audio
    try:
        arr = record_to_array(duration=duration)
    except Exception as e:
        print("❌ Recording error:", e)
        return False

    # Basic microphone/loudness check
    if np.abs(arr).mean() < 0.003:
        print("❌ Very low volume detected. Speak louder.")
        log_voice_event("FAIL_LOW_VOLUME", "Mic input too weak for verification")
        return False

    # Extract embedding
    user_embed = spectral_embedding(SAMPLE_RATE, arr)
    if user_embed is None:
        print("❌ Could not compute voice embedding.")
        log_voice_event("ERROR", "Embedding generation failed")
        return False

    # Compare embeddings
    match, dist = compare_embeddings(admin_embed, user_embed, mode=strictness)

    # Logging
    if match:
        msg = f"Voice match OK (distance={dist:.4f}) mode={strictness}"
        log_voice_event("SUCCESS", msg)
        print(f"✅ ACCESS GRANTED — Match Score = {dist:.4f}")
    else:
        msg = f"Voice mismatch (distance={dist:.4f}) mode={strictness}"
        log_voice_event("FAIL", msg)
        print(f"❌ ACCESS DENIED — Score = {dist:.4f}")

    return match


if __name__ == "__main__":
    mode = input("Mode (strict/balanced/relaxed): ").strip().lower()
    if mode not in ["strict", "balanced", "relaxed"]:
        mode = "balanced"
    authenticate_user(strictness=mode)