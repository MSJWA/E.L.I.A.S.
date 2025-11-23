# src/nlp_intent_core.py
"""
Optimized intent parser (hybrid):
- Compiles regex patterns once (O(1) per pattern compile at import).
- Uses small keyword prefilter (average-case speedup).
- Clear, small memory footprint (patterns + keyword index).
- parse_intent() is safe, fast, and returns {"intent", "entities"}.
"""

import re
from typing import Dict, Tuple, List

# ---------- PATTERN DEFINITIONS (compile once) ----------
# Each entry: (intent_name, compiled_pattern, keyword_set)
# Keyword set is used for a cheap pre-check before running regex.
_INTENT_CONFIG: List[Tuple[str, re.Pattern, Tuple[str, ...]]] = [
    ("open_website",
     re.compile(r'\b(?:open|go to|launch)\b\s+(?P<url>[A-Za-z0-9\-\._]+\.[A-Za-z]{2,6}(?:/[^\s]*)?)', re.I),
     ("open", "go", "launch")),
    ("search_youtube",
     re.compile(r'\b(?:search|find)\b(?:\s+youtube(?:\s+for)?)?\b\s*(?P<query>.+)', re.I),
     ("search", "youtube", "find")),
    ("create_note",
     re.compile(r'\b(?:create|add|write)\b\s+(?:note|a note)?\s*(?P<text>.+)', re.I),
     ("create", "add", "write", "note")),
    # Add more patterns below with appropriate keywords.
]

# Build internal structures for quick iteration
_INTENT_PATTERNS = [(name, pat) for name, pat, _ in _INTENT_CONFIG]
_KEYWORD_INDEX = {name: set(keywords) for name, _, keywords in _INTENT_CONFIG}

# ---------- Parser ----------
def _token_set(text: str):
    # cheap normalization + tokenization (O(n) time & O(k) space)
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))

def parse_intent(text: str) -> Dict:
    """
    Parse text and return {"intent": str, "entities": dict}.
    Average-case: fast via keyword prefilter.
    Worst-case: checks all patterns (O(m * n) regex runtime).
    """
    if not text:
        return {"intent": "unknown", "entities": {}}

    text_str = text.strip()
    tokens = _token_set(text_str)

    # Prefilter: only run regex for patterns with overlapping keywords.
    # This reduces regex calls in common cases.
    candidates = []
    for intent_name, pat in _INTENT_PATTERNS:
        kwset = _KEYWORD_INDEX.get(intent_name)
        if not kwset or (kwset & tokens):
            candidates.append((intent_name, pat))

    # If no candidate by keywords, fall back to all patterns (robustness)
    if not candidates:
        candidates = _INTENT_PATTERNS

    # Check patterns in defined order (priority)
    for intent_name, pat in candidates:
        m = pat.search(text_str)
        if m:
            # Use groupdict for named captures -> entities dict
            return {"intent": intent_name, "entities": m.groupdict() or {}}

    return {"intent": "unknown", "entities": {}}


# Optional: allow running the parser standalone for quick tests
if __name__ == "__main__":
    print("NLP Intent Core (test mode). Type 'exit' to quit.")
    while True:
        s = input("You: ").strip()
        if s.lower() in ("exit", "quit"):
            break
        print(parse_intent(s))
