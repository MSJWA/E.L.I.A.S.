# plugins/search_youtube.py
"""
search_youtube plugin
- Builds a YouTube search URL and opens it with webbrowser, with OS fallbacks.
Contract: run(entities: dict) -> {"ok": bool, "message": str}
"""

import webbrowser
import urllib.parse
import sys
import subprocess
import os

def _build_youtube_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"

def _os_open(url: str):
    # lightweight OS-level fallback; non-blocking subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)
            return True
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        # assume linux/unix
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def run(entities):
    """
    entities: expects 'query' (or fallback keys).
    Time: O(Q) for encoding & system call. Space: O(1).
    """
    query = (entities.get("query") or entities.get("q") or "").strip()
    if not query:
        return {"ok": False, "message": "No YouTube search query provided."}

    url = _build_youtube_url(query)

    # First try python's webbrowser (fast, non-blocking). If it reports False, try OS fallback.
    try:
        opened = webbrowser.open_new_tab(url)  # often returns True if successful
        if opened:
            return {"ok": True, "message": f"Searching YouTube for: {query}"}
    except Exception:
        # continue to fallback
        pass

    # OS-level fallback (reliable on most systems)
    if _os_open(url):
        return {"ok": True, "message": f"Searching YouTube for: {query} (fallback)"}

    return {"ok": False, "message": "Failed to open YouTube search."}
