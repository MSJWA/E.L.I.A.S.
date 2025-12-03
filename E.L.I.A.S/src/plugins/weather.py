import webbrowser
import spacy

nlp = spacy.load("en_core_web_sm")

def run(text):
    location = "islamabad" # Default
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE": # Geopolitical Entity
            location = ent.text
            
    webbrowser.open(f"https://www.google.com/search?q=weather+{location}")
    return f"🌤️ Checking weather for {location}..."