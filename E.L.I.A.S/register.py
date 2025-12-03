import os
import time
from src.face_engine import save_face
from src.voice_engine import save_voice

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    print("=" * 40)
    print("👤 NEW USER REGISTRATION")
    print("=" * 40)

    uid = input("Enter a Username (e.g. 'ali'): ").strip()
    if not uid: return

    # 1. Face
    print(f"\n📸 STEP 1: Registering Face for '{uid}'")
    input("Press ENTER to start camera...")
    if save_face(uid, BASE_DIR):
        print("✅ Face Registered Successfully.")
    else:
        print("❌ Face Registration Failed.")
        return

    # 2. Voice
    print(f"\n🎙️ STEP 2: Registering Voice for '{uid}'")
    input("Press ENTER, then speak your passphrase...")
    if save_voice(uid, BASE_DIR):
        print("✅ Voice Registered Successfully.")
    else:
        print("❌ Voice Registration Failed.")
        return

    print("\n" + "=" * 40)
    print(f"🎉 REGISTRATION COMPLETE FOR {uid}")
    print("You can now run 'app.py'")
    print("=" * 40)


if __name__ == "__main__":
    main()