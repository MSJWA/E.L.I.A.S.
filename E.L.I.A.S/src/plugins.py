import os
import datetime
import random
import webbrowser
import platform
import subprocess
import spacy

# --- 1. SETUP NLP ---
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ NLP model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# --- 2. PLUGIN FUNCTIONS ---

# Basic
def get_time():
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."

def roll_dice():
    result = random.randint(1, 6)
    return f"🎲 You rolled a {result}!"

def tell_joke():
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "There are 10 types of people in the world: those who understand binary, and those who don't.",
        "I checked all the alpha-beta pruning... turns out I just needed a gardener."
    ]
    return random.choice(jokes)

# System
def open_calculator():
    if platform.system() == "Windows":
        subprocess.Popen("calc")
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-a", "Calculator"])
    elif platform.system() == "Linux":
        subprocess.Popen(["gnome-calculator"])
    return "Opened Calculator."

def open_youtube():
    webbrowser.open("https://www.youtube.com")
    return "Opened YouTube."

def open_google():
    webbrowser.open("https://www.google.com")
    return "Opened Google."

def check_weather(location="islamabad"):
    webbrowser.open(f"https://www.google.com/search?q=weather+{location}")
    return f"Checking weather for {location}..."

# --- 3. THE MAIN COMMAND RUNNER ---

def run_command(user_text):
    """
    Analyzes the user's text and runs the matching function.
    """
    text = user_text.lower().strip()
    
    # NLP / Keyword Matching
    if "time" in text:
        return get_time()
    
    elif "joke" in text:
        return tell_joke()
    
    elif "roll" in text or "dice" in text:
        return roll_dice()
    
    elif "calculator" in text or "math" in text:
        return open_calculator()
    
    elif "youtube" in text:
        return open_youtube()
        
    elif "google" in text:
        return open_google()

    elif "weather" in text:
        # Simple entity extraction for location
        location = "islamabad" # Default
        if nlp:
            doc = nlp(user_text)
            for ent in doc.ents:
                if ent.label_ == "GPE": # Geopolitical Entity (City/Country)
                    location = ent.text
        return check_weather(location)
        
    elif "exit" in text or "quit" in text:
        return "EXIT"

    else:
        return "🤷 I didn't understand that command."