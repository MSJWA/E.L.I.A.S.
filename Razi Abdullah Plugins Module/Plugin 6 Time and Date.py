from datetime import datetime

def run(entities):
    now = datetime.now()
    return {
        "ok": True,
        "message": now.strftime("Today is %A, %d %B %Y — Time: %I:%M %p")
    }
