# plugins/create_note.py
"""
Create note plugin — appends a timestamped note to notes.txt inside a safe data folder.
Input: entities (expects 'text' or 'query' or entire user_input passed by the router)
Output: dict {"ok": bool, "message": str}
"""

import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
NOTES_FILE = os.path.join(DATA_DIR, "notes.txt")

def _now_ts():
    return datetime.utcnow().isoformat() + "Z"

def run(entities):
    # Look for text in common keys
    text = entities.get("text") or entities.get("note") or entities.get("query") or ""
    text = text.strip()
    if not text:
        return {"ok": False, "message": "No note text provided."}

    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"{_now_ts()} | {text}\n")
        return {"ok": True, "message": "Note saved."}
    except Exception as e:
        return {"ok": False, "message": f"Failed to save note: {e}"}
