# mfa_full.py
# CLI wrapper: register_face <id>, enroll_voice <id>, auth [mode]
import sys, os, time, json
from face_register import capture_embeddings
from face_auth import authenticate
from voice_enroll import enroll as voice_enroll
from voice_auth import verify as voice_verify

def usage():
    print("MFA pipeline CLI")
    print("Usage:")
    print("  python src/mfa_full.py register_face <user_id>")
    print("  python src/mfa_full.py enroll_voice <user_id>")
    print("  python src/mfa_full.py auth")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage(); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "register_face":
        uid = sys.argv[2] if len(sys.argv)>=3 else input("Enter user id: ")
        capture_embeddings(uid)
    elif cmd == "enroll_voice":
        uid = sys.argv[2] if len(sys.argv)>=3 else input("Enter user id: ")
        voice_enroll(uid)
    elif cmd == "auth":
        print("Starting MFA: silent face + voice challenge.")
        face_result = authenticate()
        print("Face result:", face_result)
        if not face_result.get("ok"):
            print("Face failed:", face_result.get("reason"))
            sys.exit(1)
        user = face_result.get("user")
        print(f"Face matched: {user}. Now voice verification.")
        vres = voice_verify(user)
        print("Voice result:", vres)
        if vres.get("ok"):
            print(f"✅ MFA AUTH SUCCESS for {user}")
        else:
            print("❌ MFA FAILED: voice did not match.")
    else:
        usage()
