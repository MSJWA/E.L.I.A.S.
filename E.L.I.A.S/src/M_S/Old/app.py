# src/app.py  — Integrated version (Day 4 / Friday)
# Purpose: REPL -> parse intent -> dispatch to real plugins -> logging + optional vision auth

import os
import time
from datetime import datetime

# --- NLP parser import ------------------------------------------------------
# Change this line if your NLP module has a different name:
# from nlp_intent_core import parse_intent
from nlp_intent_core import parse_intent

# --- Optional vision import (face authentication) --------------------------
# If vision.py exists and exports detect_face(), it will be used.
try:
    from vision import detect_face
    VISION_AVAILABLE = True
except Exception:
    detect_face = None
    VISION_AVAILABLE = False

# --- Plugin imports (safe/fall-back) --------------------------------------
# Each plugin should expose a `run(entities: dict)` function.
def safe_import(module_path, attr="run"):
    """
    Try to import `run` from module_path (e.g. 'plugins.open_website').
    Returns the callable or None.
    """
    try:
        parts = module_path.rsplit(".", 1)
        if len(parts) == 1:
            mod = __import__(module_path)
        else:
            mod = __import__(parts[0], fromlist=[parts[1]])
        return getattr(mod, attr)
    except Exception:
        return None

import sys
sys.path.append("..")   # allow parent folder access

open_website = safe_import("plugins.open_website")
youtube_search = safe_import("plugins.youtube_search")
create_note = safe_import("plugins.create_note")
# Add more plugin imports here when created:
# play_music = safe_import("plugins.play_music")

# --- ACTION MAP ------------------------------------------------------------
ACTION_MAP = {
    "open_website": open_website,
    "search_youtube": youtube_search,
    "create_note": create_note,
    # "play_music": play_music,
}

# --- Logging setup ---------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
os.makedirs(LOG_DIR, exist_ok=True)


def timestamp_now():
    return datetime.utcnow().isoformat() + "Z"


def log_entry(user_input, intent, entities, plugin_result):
    try:
        ts = timestamp_now()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} | input: {user_input!r} | intent: {intent} | entities: {entities} | result: {plugin_result}\n")
    except Exception:
        # avoid blocking the main loop if logging fails
        pass


# --- Router / Dispatcher ---------------------------------------------------
def route_intent(result):
    """
    Dispatch the parsed intent to the corresponding plugin.
    `result` is expected to be: {"intent": "...", "entities": {...}}
    Returns a dict result: {"ok": bool, "message": ...}
    """
    intent = result.get("intent", "unknown")
    entities = result.get("entities", {}) or {}

    action = ACTION_MAP.get(intent)
    if action is None:
        return {"ok": False, "message": f"No plugin registered for intent '{intent}'."}

    if not callable(action):
        return {"ok": False, "message": f"Plugin for '{intent}' not available."}

    # Call plugin safely
    try:
        # Plugins may return a dict or string; normalize to dict
        res = action(entities)
        if isinstance(res, str):
            return {"ok": True, "message": res}
        if isinstance(res, dict):
            return {"ok": res.get("ok", True), "message": res.get("message", res)}
        return {"ok": True, "message": res}
    except Exception as e:
        return {"ok": False, "message": f"Plugin error: {e}"}


# --- Main REPL -------------------------------------------------------------
def main():
    print("ELIAS Core App (integrated) Running...\nType 'exit' or 'quit' to stop.\n")

    # Optional: perform a one-time vision check before accepting commands
    if VISION_AVAILABLE:
        try:
            ok = detect_face()
            if not ok:
                print("Face not detected. If you want to require auth, enable detection in code.")
                # Option: block until face is detected. For now we warn and continue.
        except Exception as e:
            print("Vision module raised an error (continuing):", e)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Shutting down.")
            break

        # Parse
        try:
            parsed = parse_intent(user_input)
        except Exception as e:
            print("Error parsing input:", e)
            continue

        print("Intent Detected:", parsed)

        # Optional: check auth for sensitive intents (uncomment if you want)
        # if VISION_AVAILABLE and parsed.get("intent") in ("sensitive_intent",):
        #     if not detect_face():
        #         print("Authentication failed. Cannot run this action.")
        #         log_entry(user_input, parsed.get("intent"), parsed.get("entities"), {"ok": False, "message": "auth failed"})
        #         continue

        # Dispatch
        result = route_intent(parsed)
        print("Result:", result)

        # Log
        log_entry(user_input, parsed.get("intent"), parsed.get("entities"), result)

        # tiny pause to keep loop polite
        time.sleep(0.05)

    print("Goodbye.")


if __name__ == "__main__":
    main()
