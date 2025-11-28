import os
import sys
import webbrowser

# ---------------------------------------------------------
# FIX PYTHON PATH (so imports work anywhere)
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

# ---------------------------------------------------------
# IMPORT NLP PARSER
# ---------------------------------------------------------
from nlp_intent import parse_intent

# ---------------------------------------------------------
# IMPORT PLUGINS
# ---------------------------------------------------------
from plugins.open_website import run as open_website
from plugins.youtube_search import run as youtube_search
from plugins.translate import run as translate_run
from plugins.define import run as define_run
from plugins.wiki import run as wiki_run
from plugins.math_solve import run as math_run
from plugins.convert_units import run as convert_run
from plugins.play_music import run as play_music
from plugins.open_app import run as open_app
from plugins.create_note import run as create_note

# ---------------------------------------------------------
# ROUTER FUNCTION
# ---------------------------------------------------------
def route_intent(result):
    intent = result.get("intent")
    entities = result.get("entities", {})

    print(f"[ROUTER] Detected intent: {intent}")
    print(f"[ROUTER] Entities: {entities}")

    # -------------------------------------------
    # INTENT → PLUGIN MAPPING
    # -------------------------------------------
    if intent == "open_website":
        return open_website(entities)

    elif intent == "search_youtube":
        return youtube_search(entities)

    elif intent == "google_search":
        query = entities.get("query", "")
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Opened Google search for: {query}"

    elif intent == "translate_text":
        return translate_run(entities)

    elif intent == "define_word":
        return define_run(entities)

    elif intent == "wiki_search":
        return wiki_run(entities)

    elif intent == "math_solve":
        return math_run(entities)

    elif intent == "convert_units":
        return convert_run(entities)

    elif intent == "play_music":
        return play_music(entities)

    elif intent == "open_app":
        return open_app(entities)

    elif intent == "create_note":
        return create_note(entities)

    else:
        return f"[ROUTER] Unknown or unsupported intent: {intent}"


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    print("===================================")
    print("      E.L.I.A.S — Assistant        ")
    print("===================================")
    print("Type a command (or 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Shutting down ELIAS…")
            break

        # NLP PARSE
        result = parse_intent(user_input)
        print("\n[NLU OUTPUT]:", result)

        # ROUTE
        response = route_intent(result)

        print("[SYSTEM]:", response)
        print("--------------------------------------------------\n")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
