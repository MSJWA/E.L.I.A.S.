# plugins/open_website.py
"""
Robust cross-platform opener.
Tries webbrowser first, then platform fallback (os.startfile / open / xdg-open).
Returns a small dict: {"ok": bool, "message": str} for consistent app handling.
"""

import sys
import os
import subprocess
import webbrowser
from urllib.parse import urlsplit, urlunsplit

def _normalize_url(raw: str) -> str:
    s = raw.strip()
    # if already has scheme, just return normalized form
    parts = urlsplit(s if "://" in s else "https://" + s)
    # minimal sanitation: ensure netloc exists
    if not parts.netloc:
        return ""
    return urlunsplit(parts._replace(fragment=""))

def run(entities: dict):
    url_raw = entities.get("url") or entities.get("query") or ""
    url = _normalize_url(url_raw)
    if not url:
        return {"ok": False, "message": "No valid URL provided."}

    # Try webbrowser first (non-blocking). If it returns True, assume success.
    try:
        opened = webbrowser.open_new_tab(url)
        if opened:
            return {"ok": True, "message": f"Opening {url} (webbrowser)"}
    except Exception:
        # swallow; try OS fallback
        pass

    # Platform fallback — small and fast system calls
    try:
        if sys.platform.startswith("win"):
            # os.startfile is Windows-native and reliable
            os.startfile(url)
            return {"ok": True, "message": f"Opening {url} (os.startfile)"}
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "message": f"Opening {url} (open)"}
        # Linux / other
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True, "message": f"Opening {url} (xdg-open)"}
    except Exception as e:
        return {"ok": False, "message": f"Failed to open URL: {e}"}
