import spacy

# Import all your new plugin files
from . import calculator, youtube, weather, google, time_date, joke, news, system_control, screenshot, wiki

print("🧠 Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ NLP model missing. Run: python -m spacy download en_core_web_sm")
    nlp = None

def run_command(user_text):
    text = user_text.lower().strip()
    
    # --- 1. KEYWORD MATCHING (Fastest) ---
    if "calc" in text: return calculator.run(text)
    if "youtube" in text: return youtube.run(text)
    if "google" in text: return google.run(text)
    if "weather" in text: return weather.run(text)
    if "time" in text or "date" in text: return time_date.run(text)
    if "joke" in text: return joke.run(text)
    if "news" in text: return news.run(text)
    if "screenshot" in text: return screenshot.run(text)
    if "wiki" in text: return wiki.run(text)
    if "shutdown" in text or "restart" in text or "lock" in text: return system_control.run(text)
    if "exit" in text or "quit" in text: return "EXIT"

    # --- 2. NLP INTENT MATCHING (Smart) ---
    if nlp:
        doc = nlp(text)
        
        # Define Intents
        intents = {
            "youtube": nlp("watch videos play movie stream"),
            "calculator": nlp("math add subtract multiply divide"),
            "weather": nlp("rain sun temperature hot cold outside"),
            "screenshot": nlp("capture screen picture monitor"),
            "news": nlp("headlines global events update"),
            "joke": nlp("funny laugh humor comedy"),
            "wiki": nlp("information define explain describe")
        }
        
        best_intent = None
        best_score = 0.0
        
        for intent, ref_doc in intents.items():
            score = doc.similarity(ref_doc)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Threshold 0.65 to avoid random guesses
        if best_score > 0.65:
            if best_intent == "youtube": return youtube.run(text)
            if best_intent == "calculator": return calculator.run(text)
            if best_intent == "weather": return weather.run(text)
            if best_intent == "screenshot": return screenshot.run(text)
            if best_intent == "news": return news.run(text)
            if best_intent == "joke": return joke.run(text)
            if best_intent == "wiki": return wiki.run(text)

    return "🤷 I didn't understand that."