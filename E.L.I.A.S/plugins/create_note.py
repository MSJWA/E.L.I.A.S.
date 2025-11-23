# plugins/create_note.py
"""
create_note plugin
- Efficient append-only writes (low memory / fast).
- Small code surface, clear error handling.
- Saves notes in ELIAS/notes/notes.txt (one line per note).
Contract: run(entities: dict) -> dict like {"ok": bool, "message": str}
"""

import os
from datetime import datetime

# small internal helpers kept local (low overhead)
_NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notes")
_NOTES_FILE = os.path.join(_NOTES_DIR, "notes.txt")

def _ensure_notes_dir():
    # minimal check/create; idempotent
    if not os.path.isdir(_NOTES_DIR):
        try:
            os.makedirs(_NOTES_DIR, exist_ok=True)
        except Exception:
            # race / permission errors handled by caller
            pass

def _timestamp():
    return datetime.utcnow().isoformat() + "Z"

def run(entities):
    """
    entities: expects 'text' or 'note' or 'query' keys.
    Writes a single line per note: "[ISO_TIMESTAMP] note text"
    Returns a small dict for router normalization.
    Time complexity: O(L) where L = length of note text (writing). Space: O(1) extra.
    """
    text = (entities.get("text") or entities.get("note") or entities.get("query") or "").strip()
    if not text:
        return {"ok": False, "message": "No note text provided."}

    _ensure_notes_dir()
    line = f"[{_timestamp()}] {text}\n"

    try:
        # append mode: minimal memory, buffered I/O
        with open(_NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()  # ensure data pushed to OS buffer
        return {"ok": True, "message": "Note saved."}
    except Exception as e:
        return {"ok": False, "message": f"Failed to save note: {e}"}
