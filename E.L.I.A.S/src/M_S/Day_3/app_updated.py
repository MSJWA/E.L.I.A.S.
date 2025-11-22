# src/app.py  — Minimal skeleton (Day 3 style)
# Purpose: simple REPL + intent parsing + routing placeholders so teammates can plug in plugins/vision.

# Import the parser. Change this if your file is named differently:
# from nlp_intent_core import parse_intent
from nlp_intent_core import parse_intent

def route_intent(result):
    """
    Placeholder dispatcher.
    result is expected to be: {"intent": "...", "entities": {...}}
    Replace the print statements with actual plugin calls when ready.
    """
    intent = result.get("intent", "unknown")
    entities = result.get("entities", {})

    if intent == "open_website":
        # Placeholder: plugins.open_website.run(entities)
        print("-> placeholder: would call plugins.open_website.run(entities)")

    elif intent == "search_youtube":
        # Placeholder: plugins.youtube_search.run(entities)
        print("-> placeholder: would call plugins.youtube_search.run(entities)")

    elif intent == "play_music":
        # Placeholder: plugins.play_music.run(entities)
        print("-> placeholder: would call plugins.play_music.run(entities)")

    else:
        print("Unknown or unsupported intent.")


def main():
    print("ELIAS Core App Running...\nType 'exit' or 'quit' to stop.\n")

    # Note: Vision/auth integration will be added later (Friday).
    # If you have vision ready and want to gate commands, import it and check here.

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Shutting down.")
            break

        # Parse intent
        try:
            parsed = parse_intent(user_input)
        except Exception as e:
            print("Error while parsing input:", e)
            continue

        # Show parsed result for debugging
        print("Intent Detected:", parsed)

        # Route to placeholder actions
        try:
            route_intent(parsed)
        except Exception as e:
            print("Error while routing intent:", e)

if __name__ == "__main__":
    main()
