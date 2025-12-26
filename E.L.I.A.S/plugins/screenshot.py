import pyautogui
import os
import datetime

def run(text):
    folder = "screenshots"
    os.makedirs(folder, exist_ok=True)
    filename = f"screen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(folder, filename)
    
    pyautogui.screenshot().save(path)
    return f"📸 Screenshot saved to {path}"