import sys, os, random, time
from face_mp import register_face, recognize_face_once
from voice_mp import enroll_voice, verify_voice_for_user
from utils import log_event

PHRASES = [
 "quiet forest", "silent horizon", "blue river", "golden key", "purple comet",
 "calm ocean", "bright lamp", "simple code"
]

def usage():
    print("MFA pipeline CLI")
    print("Usage:")
    print("  python src/mfa_full.py register_face <user_id>")
    print("  python src/mfa_full.py enroll_voice <user_id>")
    print("  python src/mfa_full.py auth [strict|balanced|relaxed]")
    sys.exit(1)

def register_face_cmd(user):
    ok = register_face(user)
    print("Done." if ok else "Failed")

def enroll_voice_cmd(user):
    ok = enroll_voice(user)
    print("Done." if ok else "Failed")

def auth_cmd(mode="balanced"):
    print("Starting MFA (face+voice) authentication (silent camera)...")
    # face detect first (silent)
    face_res = recognize_face_once(threshold=0.55 if mode!="strict" else 0.7)
    if not face_res.get("ok"):
        print("Result:", face_res)
        log_event("auth", "face_failed", face_res)
        return
    user = face_res["user"]
    face_score = face_res["score"]
    print(f"Top face candidate: {user} (score {face_score:.3f})")
    # voice challenge
    phrase = random.choice(PHRASES)
    print("Speak this phrase now: >>>", phrase, "<<<")
    input("Press Enter then speak the phrase (recording will start)...")
    vres = verify_voice_for_user(user, wav=None, threshold=0.68 if mode=="strict" else 0.6 if mode=="balanced" else 0.52)
    if not vres.get("ok"):
        print("Voice failed:", vres)
        log_event("auth", "voice_failed", {"user": user, **vres})
        return
    voice_score = vres["score"]
    # combine
    if mode == "strict":
        face_w, voice_w = 0.5, 0.5
        required = 0.75
    elif mode == "relaxed":
        face_w, voice_w = 0.5, 0.5
        required = 0.55
    else:  # balanced
        face_w, voice_w = 0.6, 0.4
        required = 0.68
    combined = face_w * face_score + voice_w * voice_score
    ok = combined >= required
    print("Voice similarity:", f"{voice_score:.3f}")
    print("✅ MFA AUTH SUCCESS for", user) if ok else print("❌ MFA AUTH FAILED")
    print("Result:", {"ok": ok, "user": user if ok else None, "score": combined})
    log_event("auth", "result", {"user": user, "face_score": face_score, "voice_score": voice_score, "combined": combined, "ok": ok})

def main():
    if len(sys.argv) < 2:
        usage()
    cmd = sys.argv[1]
    if cmd == "register_face" and len(sys.argv) == 3:
        register_face_cmd(sys.argv[2])
    elif cmd == "enroll_voice" and len(sys.argv) == 3:
        enroll_voice_cmd(sys.argv[2])
    elif cmd == "auth":
        mode = sys.argv[2] if len(sys.argv) > 2 else "balanced"
        auth_cmd(mode)
    else:
        usage()

if __name__ == "__main__":
    main()
