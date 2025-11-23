# plugins/search_youtube.py
"""
Search YouTube plugin.
Input: entities dict (expects 'query').
Output: dict {"ok": bool, "message": str}
"""

import webbrowser
import subprocess
import sys
from urllib.parse import quote_plus, urlunsplit, urlsplit

def _make_youtube_search_url(query: str) -> str:
    q = query.strip()
    if not q:
        return ""
    return f"https://www.youtube.com/results?search_query={quote_plus(q)}"

def run(entities):
    query = (entities.get("query") or entities.get("q") or "").strip()
    if not query:
        return {"ok": False, "message": "No search query provided."}

    url = _make_youtube_search_url(query)
    try:
        opened = webbrowser.open_new_tab(url)
        if opened:
            return {"ok": True, "message": f"Searching YouTube for: {query}"}
    except Exception:
        pass

    # OS fallback
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(url)
            return {"ok": True, "message": f"Searching YouTube for: {query} (os.startfile)"}
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", url])
            return {"ok": True, "message": f"Searching YouTube for: {query} (open)"}
        subprocess.Popen(["xdg-open", url])
        return {"ok": True, "message": f"Searching YouTube for: {query} (xdg-open)"}
    except Exception as e:
        return {"ok": False, "message": f"Failed to open YouTube: {e}"}
