# src/app.py
"""
Main controller: REPL -> parse_intent -> lazy plugin dispatch -> logging.
Optimizations:
 - Lazy import for plugins (reduce startup work)
 - O(1) intent->action lookup
 - Minimal, safe logging (append-only)
 - Small, clear control flow
"""

import os
import time
import importlib
from datetime import datetime
from typing import Callable, Optional, Dict

# Import parser (ensure module name matches file)
from nlp_intent_core import parse_intent

# Optional vision placeholder
try:
    from vision import detect_face
    VISION_AVAILABLE = True
except Exception:
    detect_face = None
    VISION_AVAILABLE = False

# --------------- Plugin registry (string paths) ----------------------------
# Keep only small strings here; import later lazily.
_PLUGIN_PATHS: Dict[str, str] = {
    "open_website": "plugins.open_website",
    "search_youtube": "plugins.youtube_search",
    "create_note": "plugins.create_note",
}

# Cache of imported callables: intent -> callable
_ACTION_CACHE: Dict[str, Optional[Callable]] = {}

def _import_plugin(path: str, attr: str = "run") -> Optional[Callable]:
    """Import plugin module lazily and return the callable run function or None."""
    try:
        mod = importlib.import_module(path)
        return getattr(mod, attr)
    except Exception:
        return None

def get_action(intent_name: str) -> Optional[Callable]:
    """Return the action callable for an intent, importing on first use."""
    if intent_name in _ACTION_CACHE:
        return _ACTION_CACHE[intent_name]
    path = _PLUGIN_PATHS.get(intent_name)
    action = _import_plugin(path) if path else None
    _ACTION_CACHE[intent_name] = action
    return action

# ----------------- Logging (minimal, append-only) --------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "log.txt")

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

def log_entry(user_input: str, intent: str, entities: dict, result: dict):
    # Small structured single-line JSON-like log (avoid heavy JSON libs)
    try:
        line = f"{_now_iso()} | input:{user_input!r} | intent:{intent} | entities:{entities} | result:{result}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        # do not break the main loop if logging fails
        pass

# ------------------ Dispatcher --------------------------------------------
def route_intent(parsed: dict) -> dict:
    intent = parsed.get("intent", "unknown")
    entities = parsed.get("entities", {}) or {}

    action = get_action(intent)
    if action is None:
        return {"ok": False, "message": f"No plugin for intent '{intent}'."}

    try:
        res = action(entities)
    except Exception as e:
        return {"ok": False, "message": f"Plugin raised: {e}"}

    # Normalize result
    if isinstance(res, str):
        return {"ok": True, "message": res}
    if isinstance(res, dict):
        return {"ok": res.get("ok", True), "message": res.get("message", res)}
    return {"ok": True, "message": res}

# ------------------ REPL / Main -------------------------------------------
def main():
    print("ELIAS (integrated) Running. Type 'exit' or 'quit' to stop.")
    # Optional vision gate
    if VISION_AVAILABLE:
        try:
            if not detect_face():
                print("Warning: face not detected (continuing in non-auth mode).")
        except Exception:
            print("Vision module error; continuing without auth.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        parsed = parse_intent(user_input)
        print("Intent Detected:", parsed)

        result = route_intent(parsed)
        print("Result:", result)

        log_entry(user_input, parsed.get("intent"), parsed.get("entities"), result)
        # tiny pause to avoid very tight loop
        time.sleep(0.03)

if __name__ == "__main__":
    main()
