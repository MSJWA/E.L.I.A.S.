import subprocess
import sys

APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "chrome": "chrome.exe",
    "vscode": "code",
}

def run(entities):
    app = entities.get("app")
    if not app:
        return {"ok": False, "message": "No app specified."}

    app = app.lower().strip()

    if app not in APP_MAP:
        return {"ok": False, "message": f"Unknown app '{app}'."}

    try:
        subprocess.Popen([APP_MAP[app]])
        return {"ok": True, "message": f"Opening {app}..."}

    except Exception as e:
        return {"ok": False, "message": f"Failed to open {app}: {e}"}
