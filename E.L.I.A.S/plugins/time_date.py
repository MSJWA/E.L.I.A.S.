import webbrowser

def run(text):
    # Extract search query
    query = text.replace("google", "").replace("search", "").replace("find", "").strip()
    if not query:
        webbrowser.open("https://www.google.com")
        return "🔎 Opening Google..."
        
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return f"🔎 Googling: '{query}'"