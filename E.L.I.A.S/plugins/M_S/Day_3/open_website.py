import webbrowser
import subprocess
import sys
import os

def run(entities):
    """
    Open a website reliably across IDEs:
    - VS Code
    - PyCharm
    - Jupyter
    - Colab
    """

    url = entities.get("url")
    if not url:
        return {"ok": False, "message": "No URL provided."}

    url = url.strip()

    # ensure protocol
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # 1️⃣ Try normal python webbrowser first
        print("PLUGIN TRIGGERED: opening", url)
    
        browser_opened = webbrowser.open(url)
        if browser_opened:
            return {"ok": True, "message": f"Opening {url} (webbrowser)"}
        
        # 2️⃣ If webbrowser fails, fallback to OS-level opening
        if sys.platform.startswith("win"):
            os.startfile(url)  # Windows fallback

        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", url])  # macOS fallback

        else:
            subprocess.Popen(["xdg-open", url])  # Linux fallback

        return {"ok": True, "message": f"Opening {url} (OS fallback)"}
    
    except Exception as e:
        return {"ok": False, "message": f"Failed to open URL: {str(e)}"}
