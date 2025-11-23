import webbrowser

def run(entities):
    """
    Plugin: open_website
    Input: entities = {'url': 'google.com'}
    Output: dict with status and message.
    """

    # 1. extract URL field
    url = entities.get("url")

    if not url:
        return {
            "ok": False,
            "message": "No URL provided."
        }

    # 2. clean input (remove accidental spaces)
    url = url.strip()

    # 3. ensure protocol
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 4. open safely
    try:
        webbrowser.open(url)
        return {
            "ok": True,
            "message": f"Opening {url}"
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Failed to open URL: {str(e)}"
        }
