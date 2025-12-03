import os
import time
import sys

# Add 'src' to path to ensure imports work smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.face_engine import authenticate_face
from src.voice_engine import authenticate_voice
from src.plugins import run_command            # <--- The single file plugin system
from src.voice_listener import listen_command  # <--- The microphone listener

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def login_flow():
    print("🔒 SYSTEM LOCKED. INITIATING MFA LOGIN.")
    
    # --- PHASE 1: FACE ---
    print("\n[1/2] Scanning Face...")
    user_id, score = authenticate_face(BASE_DIR)
    
    # Strict face check (98% confidence required)
    if not user_id or score < 0.98:
        print("❌ Login Failed: Face not recognized.")
        return None
        
    print(f"✅ Face Verified: Welcome, {user_id} (Confidence: {int(score*100)}%)")
    time.sleep(1)
    
    # --- PHASE 2: VOICE (Security Check) ---
    print(f"\n[2/2] Voice Verification for {user_id}")
    input("Press ENTER and speak your phrase...")
    
    if authenticate_voice(user_id, BASE_DIR):
        print("✅ Voice Verified.")
        return user_id
    else:
        print("❌ Login Failed: Voice mismatch.")
        return None

def main():
    # 1. Perform MFA Login
    user = login_flow()
    
    if not user:
        return # Stop if login failed
        
    # 2. Success -> Enter Voice Assistant Mode
    print("\n" + "="*50)
    print(f"🔓 ACCESS GRANTED. Welcome to Dashboard, {user}.")
    print("🎤 MODE: VOICE ACTIVATED") 
    print("   (Say: 'Open Youtube', 'What time is it?', 'Weather Islamabad', 'Exit')")
    print("="*50)
    
    running = True
    while running:
        # Listen to microphone
        cmd = listen_command()
        
        if cmd:
            # Process the command using plugins
            response = run_command(cmd)
            
            # Check if user wants to quit
            if response == "EXIT":
                print("👋 Goodbye!")
                running = False
            else:
                print(f"🤖 {response}")
        else:
            # If nothing was heard, just loop back quietly
            pass

if __name__ == "__main__":
    main()