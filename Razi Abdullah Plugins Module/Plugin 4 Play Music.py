import os
import webbrowser

def run(entities):
    track = entities.get("track")

    if not track:
        return {"ok": False, "message": "No track name provided."}

    track = track.strip()
    path = os.path.join("music", track + ".mp3")

    if not os.path.exists(path):
        return {"ok": False, "message": f"Track '{track}.mp3' not found in /music folder."}

    try:
        webbrowser.open(path)
        return {"ok": True, "message": f"Playing {track}"}
    except Exception as e:
        return {"ok": False, "message": f"Error: {e}"}
