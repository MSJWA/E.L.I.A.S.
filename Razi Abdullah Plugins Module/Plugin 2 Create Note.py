import os
from datetime import datetime

def run(entities):
    text = entities.get("text")
    if not text:
        return {"ok": False, "message": "No note text provided."}

    notes_dir = "notes"
    os.makedirs(notes_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    filepath = os.path.join(notes_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        return {"ok": True, "message": f"Note saved as {filename}"}

    except Exception as e:
        return {"ok": False, "message": f"Error saving note: {e}"}
