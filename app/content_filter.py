"""
Linguistic NSFW & Explicit Content Filter for lyrics-sync.

Scans plain-text lyrics or Whisper AI transcripts for:
- Profanity / vulgarity
- Explicit sexual content
- Hate speech & slurs
- Graphic violence & self-harm

Returns content rating ('clean', 'mild', 'explicit') and detected categories in < 5ms.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Mild terms that only warrant a 'mild' rating if no severe terms exist
_MILD_TERMS = {
    "damn", "hell", "crap", "ass", "butt", "piss", "pissed", "goddamn", "bastard"
}

# Strong profanity
_STRONG_PROFANITY = [
    r"\bfuck(?:ing|er|ed|s)?\b",
    r"\bshit(?:ting|ty|s)?\b",
    r"\bbitch(?:es|y|ing)?\b",
    r"\basshole(?:s)?\b",
    r"\bcunt(?:s)?\b",
    r"\bcock(?:s)?\b",
    r"\bmotherfuck(?:er|ing|ers)?\b",
    r"\bdickhead(?:s)?\b",
    r"\btwaty?\b",
    r"\bwank(?:er|ing)?\b",
]

# Sexual content
_SEXUAL_TERMS = [
    r"\bpuss(?:y|ies)\b",
    r"\bdick(?:s)?\b",
    r"\bblowjob(?:s)?\b",
    r"\bhandjob(?:s)?\b",
    r"\bcum(?:ming|s)?\b",
    r"\bclit(?:oris)?\b",
    r"\bvagina(?:s)?\b",
    r"\bpenis(?:es)?\b",
    r"\bdildo(?:s)?\b",
    r"\bmasturbat(?:e|ing|ion)\b",
    r"\borgasm(?:s|ic)?\b",
    r"\bdeepthroat(?:ing)?\b",
    r"\bgangbang(?:ed|ing)?\b",
    r"\bthreesome(?:s)?\b",
    r"\banal sex\b",
    r"\boral sex\b",
    r"\bhardcore sex\b",
    r"\bsuck my dick\b",
    r"\btits?\b",
    r"\bboobs?\b",
]

# Hate speech & slurs
_HATE_SPEECH_TERMS = [
    r"\bnigg(?:a|er|as|ers)\b",
    r"\bfag(?:got|gots)?\b",
    r"\bdyke(?:s)?\b",
    r"\btrann(?:y|ies)\b",
    r"\bkike(?:s)?\b",
    r"\bspic(?:s)?\b",
    r"\bchink(?:s)?\b",
    r"\bwetback(?:s)?\b",
    r"\bretard(?:ed|s)?\b",
]

# Graphic violence & self-harm
_VIOLENCE_TERMS = [
    r"\bslit (?:your|my|his|her|their) throat\b",
    r"\bkill (?:your|my)self\b",
    r"\bmurder (?:you|him|her|them)\b",
    r"\bbloodbath\b",
    r"\bdecapitat(?:e|ed|ing|ion)\b",
    r"\beviscerat(?:e|ed|ing)\b",
    r"\bgore\b",
]

_COMPILED_RULES: List[Tuple[str, re.Pattern]] = [
    ("profanity", re.compile("|".join(_STRONG_PROFANITY), re.IGNORECASE)),
    ("sexual_content", re.compile("|".join(_SEXUAL_TERMS), re.IGNORECASE)),
    ("hate_speech", re.compile("|".join(_HATE_SPEECH_TERMS), re.IGNORECASE)),
    ("violence", re.compile("|".join(_VIOLENCE_TERMS), re.IGNORECASE)),
    ("mild_profanity", re.compile(r"\b(?:" + "|".join(_MILD_TERMS) + r")\b", re.IGNORECASE)),
]


def evaluate_content_rating(lyrics_text: Optional[str]) -> Dict[str, Any]:
    """
    Evaluate explicit content in lyrics text.

    Returns:
      {
        "is_explicit": bool,
        "rating": "clean" | "mild" | "explicit",
        "categories": list[str],
        "matched_count": int,
        "matched_terms": list[str]
      }
    """
    if not lyrics_text or not lyrics_text.strip():
        return {
            "is_explicit": False,
            "rating": "clean",
            "categories": [],
            "matched_count": 0,
            "matched_terms": [],
        }

    matched_categories: Set[str] = set()
    matched_terms: List[str] = []
    has_severe = False
    mild_count = 0

    for category, pattern in _COMPILED_RULES:
        matches = pattern.findall(lyrics_text)
        if not matches:
            continue

        for m in matches:
            norm = m.lower().strip()
            if norm not in matched_terms:
                matched_terms.append(norm)

        if category == "mild_profanity":
            mild_count += len(matches)
            if not has_severe:
                matched_categories.add("profanity")
        else:
            has_severe = True
            matched_categories.add(category)

    total_matches = len(matched_terms)
    if not matched_terms:
        return {
            "is_explicit": False,
            "rating": "clean",
            "categories": [],
            "matched_count": 0,
            "matched_terms": [],
        }

    if has_severe:
        rating = "explicit"
        is_explicit = True
    elif mild_count > 2:
        rating = "mild"
        is_explicit = True
    else:
        rating = "mild"
        is_explicit = False

    return {
        "is_explicit": is_explicit,
        "rating": rating,
        "categories": sorted(list(matched_categories)),
        "matched_count": total_matches,
        "matched_terms": matched_terms[:10],
    }
