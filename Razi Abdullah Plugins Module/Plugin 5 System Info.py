import platform
import os

def run(entities):
    try:
        info = {
            "os": platform.system(),
            "release": platform.release(),
            "user": os.getlogin()
        }
        return {"ok": True, "message": info}
    except Exception as e:
        return {"ok": False, "message": str(e)}
