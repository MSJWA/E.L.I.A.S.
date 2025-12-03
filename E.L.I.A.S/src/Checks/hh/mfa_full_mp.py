from face_auth_mp import identify_face
from voice_auth_fft import voice_check
import random

phrases = [
    "blue river",
    "quiet forest",
    "silver moon",
    "soft thunder",
    "clear morning",
]

def mfa_auth():
    print("=== MFA (Face + Voice) ===")

    uid, fscore = identify_face()
    if uid is None:
        print("❌ No face detected")
        return {"ok": False}

    print(f"Face match → {uid}  (score={fscore:.3f})")

    phrase = random.choice(phrases)
    print(f"Speak this phrase: >>> {phrase} <<<")
    input("Press ENTER and speak...")

    ok, vscore = voice_check(uid)

    if ok:
        print(f"✅ MFA SUCCESS for {uid}")
        return {"ok": True, "user": uid, "score": (fscore * vscore)}

    print("❌ Voice mismatch")
    return {"ok": False}


if __name__ == "__main__":
    mfa_auth()
