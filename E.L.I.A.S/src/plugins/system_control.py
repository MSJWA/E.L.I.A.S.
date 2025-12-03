import os
import platform

def run(text):
    if "shutdown" in text:
        os.system("shutdown /s /t 5") if platform.system() == "Windows" else os.system("shutdown now")
        return "⚠️ Shutting down in 5 seconds..."
    
    if "restart" in text:
        os.system("shutdown /r /t 5") if platform.system() == "Windows" else os.system("reboot")
        return "⚠️ Restarting in 5 seconds..."
        
    if "lock" in text:
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "🔒 System Locked."
        
    return "❌ Command not recognized."