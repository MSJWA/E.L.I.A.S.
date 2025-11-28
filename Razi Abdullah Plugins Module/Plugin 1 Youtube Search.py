import webbrowser
import urllib.parse

def run(entities):
    query = entities.get("query")

    if not query:
        return {"ok": False, "message": "No query provided for YouTube search."}

    query = query.strip()
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)

    try:
        webbrowser.open(url)
        return {"ok": True, "message": f"Searching YouTube for: {query}"}
    except Exception as e:
        return {"ok": False, "message": f"Failed: {e}"}
