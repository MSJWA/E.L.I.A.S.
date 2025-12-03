import subprocess
import platform

def run(text):
    if platform.system() == "Windows":
        subprocess.Popen("calc")
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-a", "Calculator"])
    elif platform.system() == "Linux":
        subprocess.Popen(["gnome-calculator"])
    return "🧮 Calculator opened."