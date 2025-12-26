import random

JOKES = [
    "Why do Java programmers wear glasses? Because they don't C#.",
    "I told my computer I needed a break, and now it won't stop sending me Kit-Kats.",
    "A SQL query walks into a bar, walks up to two tables and asks, 'Can I join you?'"
]

def run(text):
    return f"🤣 {random.choice(JOKES)}"