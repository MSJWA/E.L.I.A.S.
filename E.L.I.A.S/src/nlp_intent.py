import re

def parse_intent(text):
    """
    Analyzes user text to find intents.
    Week 2 Update: Added support for 'system_open_app'.
    """
    text = text.lower().strip()

    # 1. System App Intent (Mobeen's Day 8 Task)
    # Matches: "open calculator", "launch notepad", "start paint"
    app_pattern = r"\b(open|launch|start)\b\s+(?P<app_name>calculator|notepad|paint|cmd|word)"
    match = re.search(app_pattern, text)
    if match:
        return {
            "action": "system_open_app",
            "data": match.group("app_name")
        }

    # 2. Open Website
    website_pattern = r"\b(open|launch)\b\s+(?P<url>.+\.\w+)"
    match = re.search(website_pattern, text)
    if match:
        return {
            "action": "open_website",
            "data": match.group("url")
        }

    # 3. YouTube Search
    youtube_pattern = r"\bsearch youtube for\b\s+(?P<query>.+)"
    match = re.search(youtube_pattern, text)
    if match:
        return {
            "action": "search_youtube",
            "data": match.group("query")
        }

    # Default
    return {"action": None, "data": None}