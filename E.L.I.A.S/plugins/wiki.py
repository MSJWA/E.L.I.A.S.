import wikipedia

def run(text):
    query = text.replace("wiki", "").replace("who is", "").replace("what is", "").strip()
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"📚 Wikipedia: {summary}"
    except:
        return "❌ Could not find that topic."