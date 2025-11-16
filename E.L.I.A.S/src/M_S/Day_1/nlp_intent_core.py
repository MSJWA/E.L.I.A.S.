import re

# simple pattern list
INTENT_PATTERNS = [
    ("open_website", re.compile(r'\b(?:open|go to|launch)\b\s+(?P<url>[A-Za-z0-9\-\._]+\.[A-Za-z]{2,6})', re.I)),
    ("search_youtube", re.compile(r'\bsearch youtube for\b\s+(?P<query>.+)', re.I))

]

def parse_intent(text):
    text = text.strip()
    for intent, pat in INTENT_PATTERNS:
        match = pat.search(text)
        if match:
            return {"intent": intent, "entities": match.groupdict()}
    return {"intent": "unknown", "entities": {}}

print("E.L.I.A.S. intent core — type something:")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    print("→", parse_intent(user_input))
