import speech_recognition as sr

def listen_command():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\n🎤 Listening... (Speak now)")
        
        # Adjust for background noise automatically
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            # Listen for up to 5 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("⏳ Processing...")
            
            # Use Google Web Speech API (Free & High Quality)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: '{text}'")
            return text.lower()
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected.")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return None
        except sr.RequestError:
            print("❌ Network error (Internet required).")
            return None