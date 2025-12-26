import webbrowser

def run(text):
    # Check if user wants to search for a specific video
    if "search" in text or "play" in text:
        # Remove trigger words to find the query
        query = text.replace("youtube", "").replace("search", "").replace("play", "").strip()
        if query:
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"📺 Searching YouTube for '{query}'..."
            
    webbrowser.open("https://www.youtube.com")
    return "📺 Opening YouTube..."