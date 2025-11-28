# src/nlp_intent.py
"""
NLU Intent Parser — E.L.I.A.S. (final merged Day 1 → Day 16)

Features included:
- Normalization (prefix/suffix, quotes, spacing, "dot" -> .)
- Conservative fuzzy correction (small vocab, token-level)
- Ordered regex-based intent matching (specific -> general)
- SAFE fallback heuristics
- Loose heuristics for math/convert/wiki/define
- create_note, open_website, search_youtube, google_search, play_music, open_app
- Confidence scoring + explainability fields (rule, source)
- Small-talk, how-to, wikipedia search, sarcasm & emotion hints
- Multi-intent discovery helper (parse_all_intents)
- Minimal in-memory user_state placeholders for session context
- Defensive programming: safe returns, tidy entities
"""

import re
import difflib
from typing import Dict, Optional, Tuple, List, Any

# ---------------------------
# Configuration / Constants
# ---------------------------

UPLOADED_ASSET_PATH = "/mnt/data/E.L.I.A.S. - Copy.pptx"

_FUZZY_VOCAB = sorted({
    "open", "launch", "start", "run",
    "search", "google", "youtube", "yt",
    "play", "music", "translate", "convert",
    "define", "what", "who", "wiki", "wikipedia",
    "solve", "calculate", "timer", "weather", "time", "date",
})

_FALLBACK_INTENTS = {"open_website", "google_search", "search_youtube", "play_music", "open_app", "create_note"}

# Very small lexicons for emotion/sarcasm hints (lightweight heuristics)
_POSITIVE_WORDS = {"good", "great", "nice", "awesome", "happy", "love"}
_NEGATIVE_WORDS = {"bad", "sad", "angry", "upset", "hate", "terrible"}
_SARCASM_MARKERS = {"yeah right", "as if", "sure", "I guess", "I suppose", "lol"}

# ---------------------------
# Normalization helpers
# ---------------------------

_POLITE_PREFIX_RE = re.compile(
    r"^(?:hey elias|hey|hi|hello|please|could you|can you|would you|will you|kindly|pls|can u|could u)\b[\s,:-]*",
    re.I,
)
_POLITE_SUFFIX_RE = re.compile(r"(?:\bfor me\b|\bplease\b|\bthanks\b|\bthank you\b)[\s.!?]*$", re.I)
_QUOTES_RE = re.compile(r"[“”«»\"']")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

def normalize_text(text: str) -> str:
    """Normalize input: remove polite prefixes/suffixes, quotes, collapse spaces, normalize 'dot' -> '.'"""
    if not text:
        return text
    t = text.strip()
    t = _QUOTES_RE.sub("", t)
    # iteratively remove polite prefixes
    while True:
        new = _POLITE_PREFIX_RE.sub("", t, count=1).strip()
        if new == t:
            break
        t = new
    t = _POLITE_SUFFIX_RE.sub("", t).strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = re.sub(r"\b(dot)\b", ".", t, flags=re.I)
    t = re.sub(r"\s*\.\s*", ".", t)
    return t

# ---------------------------
# Conservative fuzzy correction
# ---------------------------

def fuzzy_correct_text(text: str, cutoff: float = 0.85) -> str:
    """Token-level conservative fuzzy correction using a tiny vocabulary."""
    if not text:
        return text
    parts = re.split(r'(\W+)', text)  # keep separators
    for i, tok in enumerate(parts):
        if not tok or re.search(r'\W', tok):
            continue
        low = tok.lower()
        if len(low) <= 2:
            continue
        matches = difflib.get_close_matches(low, _FUZZY_VOCAB, n=1, cutoff=cutoff)
        if matches:
            rep = matches[0]
            if tok[0].isupper():
                rep = rep.capitalize()
            parts[i] = rep
    return ''.join(parts)

# ---------------------------
# Utility helpers
# ---------------------------

def tidy_entities(gd: Dict[str, Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (gd or {}).items():
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        out[k] = s
    return out

def looks_like_url(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = re.search(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+\.[a-z]{2,6}(?:/[^\s]*)?)", s, flags=re.I)
    if m:
        val = m.group(1)
        val = re.sub(r"[^\w\./\-]+$", "", val)
        return val
    return None

def _lower_tokens_set(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))

# ---------------------------
# Intent patterns (ordered)
# ---------------------------

INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [

    # convert_units
    ("convert_units",
     re.compile(r"\b(?:convert|change)\b\s+(?P<value>[\d\.]+)\s*(?P<from_unit>[A-Za-z]+)\s+(?:to|into)\s+(?P<to_unit>[A-Za-z]+)\b", re.I)),

    # translate_text
    ("translate_text",
     re.compile(r"\btranslate\b\s+(?P<text>.+?)\s+(?:to|into)\s+(?P<lang>[A-Za-z ]{1,30})\b", re.I)),

    # math_solve
    ("math_solve",
     re.compile(r"\b(?:solve|calculate|compute|what is|what's)\b\s+(?P<expr>[-+\d\.\s\*/\^()%]+)", re.I)),

    # open_website
    ("open_website",
     re.compile(r"\b(?:open|go to|visit|launch)\b\s+(?P<url>(?:https?://)?(?:www\.)?[A-Za-z0-9\-\._]+\.[A-Za-z]{2,6}(?:/[^\s]*)?)(?:\b|$)", re.I)),

    # search_youtube
    ("search_youtube",
     re.compile(r"\b(?:search|look up|find)\b\s+(?:youtube|yt)\s+(?:for\s+)?(?P<query>.+)", re.I)),

    # google_search
    ("google_search",
     re.compile(r"\b(?:google search|search google for|search google|search on google)\b\s+(?P<query>.+)", re.I)),

    # create_note
    ("create_note",
     re.compile(r"\b(?:create|make|note|save)\b(?:\s+(?:a|an|the))?\s*(?:note\b\s*(?:that)?\s*)?(?P<text>.+)", re.I)),

    # define_word
    ("define_word",
     re.compile(r"\b(?:define|what is|what's|explain)\b\s+(?P<term>[A-Za-z0-9 _\-\+]+)\b", re.I)),

    # wiki_search (same as wikipedia)
    ("wiki_search",
     re.compile(r"\b(?:who is|who was|tell me about|biography of|wiki)\b\s+(?P<query>.+)", re.I)),

    # how_to / "how to"
    ("how_to",
     re.compile(r"\bhow to\b\s*(?P<task>.+)", re.I)),

    # play_music
    ("play_music",
     re.compile(r"\b(?:play|put on|listen to|play me)\b\s+(?P<track>.+)", re.I)),

    # open_app
    ("open_app",
     re.compile(r"\b(?:open|launch|start|run)\b\s+(?P<app>[A-Za-z0-9_ \-\+\.]+)\b", re.I)),

    # get_time
    ("get_time",
     re.compile(r"\b(what\s+time\s+is\s+it|tell\s+me\s+the\s+time|current\s+time|time\s+please|time)\b", re.I)),

    # get_date
    ("get_date",
     re.compile(r"\b(?:what(?:'s| is)? the date|today(?:'s)? date|date today|what day is it)\b", re.I)),

    # get_weather
    ("get_weather",
     re.compile(r"\b(?:weather(?: in)? (?P<city>[A-Za-z ]+)|what'?s the weather|weather)\b", re.I)),

    # tell_joke
    ("tell_joke",
     re.compile(r"\b(?:tell me a joke|tell a joke|make me laugh|joke)\b", re.I)),

    # set_timer
    ("set_timer",
     re.compile(r"\b(?:set timer for|timer|countdown)\s+(?P<time>\d+)\s*(?:minutes|min)?\b", re.I)),

    # small talk / greetings
    ("small_talk",
     re.compile(r"\b(?:hi|hello|hey|good morning|good evening|good night|how are you|what's up)\b", re.I)),
]

# ---------------------------
# Day 11-16 helpers: sarcasm & emotion lightweight detectors
# ---------------------------

def detect_sarcasm(text: str) -> bool:
    t = text.lower()
    # simple heuristics: explicit markers or phrases
    for marker in _SARCASM_MARKERS:
        if marker.lower() in t:
            return True
    # exclamation used with negative words often indicates sarcasm
    if "!" in t and any(w in t for w in ("yeah", "sure", "right")):
        return True
    return False

def detect_emotion(text: str) -> Optional[str]:
    tset = _lower_tokens_set(text)
    if tset & _POSITIVE_WORDS:
        return "positive"
    if tset & _NEGATIVE_WORDS:
        return "negative"
    return None

# ---------------------------
# Confidence helper and explainability
# ---------------------------

def _confidence_for_match(mode: str) -> float:
    if mode == "direct":
        return 1.0
    if mode == "fallback":
        return 0.92
    if mode == "loose":
        return 0.6
    if mode == "multi":
        return 0.8
    return 0.0

# ---------------------------
# Minimal session storage placeholder (Day 16)
# ---------------------------
_user_state: Dict[str, Dict[str, Any]] = {}  # user_id -> state dict

def set_user_state(user_id: str, key: str, value: Any) -> None:
    s = _user_state.setdefault(user_id, {})
    s[key] = value

def get_user_state(user_id: str, key: str, default: Any = None) -> Any:
    return _user_state.get(user_id, {}).get(key, default)

# ---------------------------
# SAFE fallback heuristics
# ---------------------------

def fallback_intent(text: str) -> Tuple[Optional[str], Dict[str, str]]:
    t = text.strip()
    url = looks_like_url(t)
    if url:
        return "open_website", {"url": url}

    if re.search(r"\b(?:youtube|yt)\b", t, re.I):
        q = re.sub(r"\b(?:search|look up|find|on)\b", "", t, flags=re.I).strip()
        q = re.sub(r"\b(?:youtube|yt)\b", "", q, flags=re.I).strip()
        if q:
            return "search_youtube", {"query": q}

    if re.search(r"\bgoogle\b", t, re.I):
        q = re.sub(r"\b(?:google|search|on)\b", "", t, flags=re.I).strip()
        if q:
            return "google_search", {"query": q}
        m = re.match(r"google\s+(?P<q>.+)", t, flags=re.I)
        if m and m.group("q"):
            return "google_search", {"query": m.group("q").strip()}

    if re.search(r"\b(?:play|music|song|listen to|put on)\b", t, re.I):
        q = re.sub(r"\b(?:please|could you|can you|play|play me|put on|listen to)\b", "", t, flags=re.I).strip()
        if len(q) >= 1:
            return "play_music", {"track": q}

    if re.search(r"\b(?:note|remember|save)\b", t, re.I):
        q = re.sub(r"\b(?:note|remember|save|please)\b", "", t, flags=re.I).strip()
        if q:
            return "create_note", {"text": q}

    m = re.search(r"\b(?:open|launch|start|run)\b\s+(?P<app>.+)", t, re.I)
    if m:
        app = m.group("app").strip()
        if app:
            return "open_app", {"app": app}

    return None, {}

# ---------------------------
# Multi-intent helper (Day 15/16)
# ---------------------------

def parse_all_intents(text: str) -> List[Dict]:
    """
    Return list of all matching intents (ordered by pattern list).
    Each entry contains intent, entities and confidence.
    Use this for diagnostics or multi-action suggestions.
    """
    if not text:
        return []
    cleaned = normalize_text(fuzzy_correct_text(text))
    results = []
    for intent_name, pattern in INTENT_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            entities = tidy_entities(m.groupdict())
            results.append({
                "intent": intent_name,
                "entities": entities,
                "confidence": _confidence_for_match("direct"),
                "rule": pattern.pattern
            })
    return results

# ---------------------------
# Smart hint generator
# ---------------------------

def smart_hint(text: str) -> str:
    t = text.lower()
    if re.search(r"\bweather\b", t):
        return 'Did you mean: "what\'s the weather in <city>?"'
    if re.search(r"\btime\b", t):
        return 'Try: "what time is it" or "tell me the time"'
    if re.search(r"\btranslate\b", t):
        return 'Try: "translate <text> to <language>" (e.g. translate hello to spanish)'
    if re.search(r"\bgoogle\b|\bsearch\b", t):
        return 'Try: "search google for <query>" or "google search <query>"'
    if re.search(r"\byoutube\b|\byt\b", t):
        return 'Try: "search youtube for <query>" or "search yt for <query>"'
    if re.search(r"\bplay\b|\bmusic\b|\bsong\b", t):
        return 'Try: "play <song name>" or "play music <genre>"'
    return "Try: 'open google.com' or 'search youtube for lofi'."

# ---------------------------
# Main parse function
# ---------------------------

def parse_intent(text: str, user_id: Optional[str] = None) -> Dict:
    """
    Parse text into a single primary intent result:
    returns {
      "intent": str,
      "entities": dict,
      "confidence": float,
      "explain": {"rule": pattern, "source": "direct|fallback|loose"},
      optional "hint": str,
      optional "emotion": str,
      optional "sarcasm": bool
    }
    """
    if not text:
        return {"intent": "unknown", "entities": {}, "confidence": 0.0, "hint": "Please type a command."}

    # Step 0: fuzzy + normalize
    fuzzed = fuzzy_correct_text(text, cutoff=0.85)
    cleaned = normalize_text(fuzzed)

    # Step 1: direct regex matching (ordered)
    for intent_name, pattern in INTENT_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            entities = tidy_entities(m.groupdict())
            if "url" in entities:
                urlnorm = looks_like_url(entities["url"])
                if urlnorm:
                    entities["url"] = urlnorm
            res = {
                "intent": intent_name,
                "entities": entities,
                "confidence": _confidence_for_match("direct"),
                "explain": {"rule": pattern.pattern, "source": "direct"}
            }
            # additional diagnostics
            emo = detect_emotion(cleaned)
            if emo:
                res["emotion"] = emo
            sar = detect_sarcasm(cleaned)
            if sar:
                res["sarcasm"] = True
            # optional: set some user state hint
            if user_id and intent_name == "create_note":
                set_user_state(user_id, "last_note", entities.get("text", ""))
            return res

    # Step 2: SAFE fallback
    fallback_name, fallback_entities = fallback_intent(cleaned)
    if fallback_name and fallback_name in _FALLBACK_INTENTS:
        ents = tidy_entities(fallback_entities)
        res = {
            "intent": fallback_name,
            "entities": ents,
            "confidence": _confidence_for_match("fallback"),
            "explain": {"rule": "fallback heuristics", "source": "fallback"}
        }
        emo = detect_emotion(cleaned)
        if emo:
            res["emotion"] = emo
        return res

    # Step 3: Looser heuristics
    m = re.search(r"\b(?:who is|who was|tell me about|biography of)\b\s+(?P<q>.+)", cleaned, re.I)
    if m and m.group("q"):
        return {
            "intent": "wiki_search",
            "entities": {"query": m.group("q").strip()},
            "confidence": _confidence_for_match("loose"),
            "explain": {"rule": r"loose:wikipedia", "source": "loose"}
        }

    m = re.search(r"\b(?:what does|meaning of|what is)\b\s+(?P<term>[A-Za-z0-9 _\-\+]+)", cleaned, re.I)
    if m and m.group("term"):
        return {
            "intent": "define_word",
            "entities": {"term": m.group("term").strip()},
            "confidence": _confidence_for_match("loose"),
            "explain": {"rule": r"loose:define", "source": "loose"}
        }

    m = re.search(r"(?P<expr>(?:\d+[\s]*[+\-/*^][\s]*\d+)(?:[\s]*[+\-/*^][\s]*\d+)*)", cleaned)
    if m and m.group("expr"):
        return {
            "intent": "math_solve",
            "entities": {"expr": m.group("expr").strip()},
            "confidence": _confidence_for_match("loose"),
            "explain": {"rule": r"loose:math", "source": "loose"}
        }

    m = re.search(r"(?P<value>[\d\.]+)\s*(?P<from_unit>[A-Za-z]+)\s+(?:to|into)\s+(?P<to_unit>[A-Za-z]+)", cleaned, re.I)
    if m and m.group("value"):
        return {
            "intent": "convert_units",
            "entities": {
                "value": m.group("value"),
                "from_unit": m.group("from_unit"),
                "to_unit": m.group("to_unit")
            },
            "confidence": _confidence_for_match("loose"),
            "explain": {"rule": r"loose:convert", "source": "loose"}
        }

    # Step 4: small-talk soft match (try quick patterns)
    for intent_name, pattern in INTENT_PATTERNS:
        if intent_name == "small_talk":
            if pattern.search(cleaned):
                return {
                    "intent": "small_talk",
                    "entities": {},
                    "confidence": _confidence_for_match("direct"),
                    "explain": {"rule": pattern.pattern, "source": "direct"}
                }

    # Step 5: Nothing matched -> unknown + hint + emotion/sarcasm diagnostics
    res = {"intent": "unknown", "entities": {}, "confidence": 0.0, "hint": smart_hint(cleaned)}
    emo = detect_emotion(cleaned)
    if emo:
        res["emotion"] = emo
    if detect_sarcasm(cleaned):
        res["sarcasm"] = True
    # provide multi-intent alternative suggestions (lightweight)
    alternatives = parse_all_intents(cleaned)
    if alternatives:
        res["alternatives"] = alternatives[:3]  # send up to 3 suggestions
    return res

# ---------------------------
# Quick interactive test (only when run directly)
# ---------------------------

if __name__ == "__main__":
    print("NLU Intent Parser (final). Try examples:")
    examples = [
        "open google.com",
        "search yt for lofi",
        "translate hello to spanish",
        "convert 10km to miles",
        "define entropy",
        "who is alan turing",
        "solve 12 * 14",
        "create a note buy milk",
        "how to tie a tie",
        "play despacito",
        "what's the weather in Lahore",
        "hi, how are you?"
    ]
    for ex in examples:
        print(f"\nYou: {ex}")
        print(parse_intent(ex))
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        print(parse_intent(line))
