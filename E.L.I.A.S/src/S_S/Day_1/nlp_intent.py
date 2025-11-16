import re

# --- Intent Patterns ---
INTENT_PATTERNS = [
    (
        "open_website",
        re.compile(
            r"\b(?:open|go to|launch)\b\s+(?P<url>[A-Za-z0-9\-\._]+\.[A-Za-z]{2,6})",
            re.I,
        ),
    ),

    # ⭐ Your added pattern
    (
        "search_youtube",
        re.compile(r"search youtube (?P<query>.+)", re.I),
    ),
]


# --- Intent Parsing Function ---
def parse_intent(text):
    text = text.strip()

    for intent, pat in INTENT_PATTERNS:
        match = pat.search(text)
        if match:
            return {"intent": intent, "entities": match.groupdict()}

    return {"intent": "unknown", "entities": {}}


# --- Local Testing Block ---
if __name__ == "__main__":
    print("NLP Test Module Running...")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = parse_intent(query)
        print("Parsed: ", result)