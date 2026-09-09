"""
Extract embedded lyrics, genres, and style metadata from MP3 ID3 tags.

Reads USLT, TXXX:LYRICS, and SYLT frames and returns both
timed (LRC) and plain (stripped) representations.
Also extracts ID3 TCON genres, TBPM, and parses AI style preambles
(e.g., [Style: Techno, EBM, driving bass]) commonly present in Suno/Udio tracks.
"""

import re
from typing import Optional, List, Tuple, Dict, Any
from mutagen.id3 import ID3, SYLT
from mutagen.mp3 import MP3

# Matches LRC timestamp: [MM:SS.xx] or [M:SS] or [M:SS.xx]
_LRC_TS_RE = re.compile(r"^\s*\[\d+:\d{2}(?:\.\d+)?\]")
_LRC_LINE_RE = re.compile(r"^\s*\[(\d+):(\d{2}(?:\.\d+)?)\](.*)")
_LYRICS_TXXX_DESCS = {"LYRICS", "LRC", "SYNCEDLYRICS", "UNSYNCEDLYRICS"}

# Matches style/genre preambles at the start of lines or text
_STYLE_PREAMBLE_RE = re.compile(
    r"^(?:\[(?:Style|Genre|Tags|Prompt|Mood):\s*([^\]]+)\]|(?:Style|Genre|Tags|Mood):\s*(.+))(?:\r?\n)?",
    re.IGNORECASE | re.MULTILINE
)

# Known genres taxonomy for extracting primary genres from arbitrary style prompts
_GENRE_KEYWORDS = {
    "techno", "ebm", "electronic", "dance", "edm", "house", "trance", "synthwave",
    "retrowave", "industrial", "cyberpunk", "ambient", "pop", "rock", "metal",
    "punk", "hip hop", "hip-hop", "rap", "r&b", "soul", "funk", "disco", "jazz",
    "blues", "country", "folk", "reggae", "classical", "acoustic", "lo-fi", "lofi",
    "drum and bass", "dnb", "dubstep", "hyperpop", "hardstyle", "garage", "trap",
    "indie", "alternative", "grunge", "ska", "gothic", "darkwave", "heavy metal",
    "black metal", "death metal", "electro", "synth-pop", "synthpop", "breakbeat"
}

_IGNORE_STYLE_TOKENS = {
    "no slop", "slop", "hq", "high quality", "clean", "explicit", "fast", "slow",
    "high-energy", "mid-tempo", "song", "track", "music", "masterpiece"
}


def _strip_lrc_timestamps(lrc_text: str) -> str:
    """Remove leading LRC timestamps from each line and return plain lyrics."""
    plain_lines = []
    for line in lrc_text.splitlines():
        stripped = _LRC_TS_RE.sub("", line).strip()
        if stripped:
            plain_lines.append(stripped)
    return "\n".join(plain_lines)


def _has_lrc_timestamps(text: str) -> bool:
    """Return True if the text contains at least one LRC timestamp line."""
    return any(_LRC_TS_RE.match(line) for line in text.splitlines())


def parse_style_prompt(prompt_text: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Parse a style/genre prompt string into:
      - genres: Normalized list of recognizable music genres (e.g., ["Techno", "EBM"]).
      - style_tags: Full list of descriptive tags (e.g., ["driving bass", "female vocals"]).
      - raw_prompt: Cleaned unparsed prompt string.
    """
    if not prompt_text:
        return [], [], None

    raw = prompt_text.strip()
    # Strip any enclosing brackets or prefix like [Style: ...] or Style: ...
    raw_clean = re.sub(
        r"^\[?(?:Style|Genre|Tags|Prompt|Mood):\s*", "", raw, flags=re.IGNORECASE
    ).rstrip("]").strip()

    # Split by comma, semicolon, or slash
    parts = [p.strip() for p in re.split(r"[,;/]+", raw_clean) if p.strip()]

    genres: List[str] = []
    style_tags: List[str] = []

    for part in parts:
        lower = part.lower()
        if lower in _IGNORE_STYLE_TOKENS:
            continue

        style_tags.append(part)

        # Match against known music genres
        for g in _GENRE_KEYWORDS:
            if re.search(rf"\b{re.escape(g)}\b", lower):
                # Normalize genre name to Title Case
                genres.append(part.title())
                break

    # Deduplicate preserving order
    dedup_genres = list(dict.fromkeys(genres))
    dedup_styles = list(dict.fromkeys(style_tags))

    return dedup_genres, dedup_styles, raw_clean if raw_clean else None


def strip_style_preamble(text: str) -> Tuple[str, Optional[str]]:
    """
    Detect and strip leading [Style: ...] or Style: ... preambles from lyrics text.
    Returns: (cleaned_text, extracted_raw_prompt_or_None)
    """
    if not text:
        return "", None

    match = _STYLE_PREAMBLE_RE.search(text)
    if match and match.start() < 100:  # Only treat as preamble if near the beginning
        raw_prompt = match.group(1) or match.group(2)
        cleaned_text = _STYLE_PREAMBLE_RE.sub("", text, count=1).strip()
        return cleaned_text, raw_prompt.strip()

    return text.strip(), None


def _load_tags(mp3_path: str):
    """Load tags via MP3 first, with ID3 fallback for edge cases."""
    try:
        audio = MP3(mp3_path)
        if audio.tags is not None:
            return audio.tags
    except Exception:
        pass

    try:
        return ID3(mp3_path)
    except Exception:
        return None


def _sylt_to_lrc(sylt_frame: SYLT) -> str:
    """Convert a SYLT frame to LRC-format text."""
    lines = []
    for text, timestamp_ms in sylt_frame.text:
        minutes = timestamp_ms // 60000
        seconds = (timestamp_ms % 60000) / 1000
        lrc_ts = f"[{minutes:02d}:{seconds:05.2f}]"
        clean = text.strip().rstrip("\n").rstrip("\x00")
        if clean:
            lines.append(f"{lrc_ts}{clean}")
    return "\n".join(lines)


def extract_lyrics_from_mp3(mp3_path: str) -> Dict[str, Any]:
    """
    Extract embedded lyrics, genres, and style metadata from an MP3 file's ID3 tags.

    Returns a dict with:
      - plain_lyrics: str | None (preambles stripped)
      - timed_lyrics_lrc: str | None (preambles stripped)
      - genres: list[str]
      - style_tags: list[str]
      - style_prompt_raw: str | None
      - bpm: int | None
      - sources: { uslt: bool, txxx_lyrics: bool, sylt: bool, tcon: bool, tbpm: bool }
      - notes: str | None
    """
    sources = {
        "uslt": False,
        "txxx_lyrics": False,
        "sylt": False,
        "tcon": False,
        "tbpm": False,
    }
    timed_lrc: Optional[str] = None
    plain: Optional[str] = None
    notes_parts: List[str] = []
    all_genres: List[str] = []
    all_style_tags: List[str] = []
    raw_prompt_found: Optional[str] = None
    bpm_val: Optional[int] = None

    tags = _load_tags(mp3_path)
    if tags is None:
        # No ID3 tags or unreadable file — return empty result
        return {
            "plain_lyrics": None,
            "timed_lyrics_lrc": None,
            "genres": [],
            "style_tags": [],
            "style_prompt_raw": None,
            "bpm": None,
            "sources": sources,
            "notes": "No ID3 tags found.",
        }

    # ── 1. Read standard ID3 TCON (Genre) and TBPM (BPM) ─────────────────────
    tcon = tags.get("TCON")
    if tcon and hasattr(tcon, "text") and tcon.text:
        sources["tcon"] = True
        for genre_item in tcon.text:
            # Handle potential delimiter in genre string
            for g_split in re.split(r"[/,;]+", str(genre_item)):
                g_clean = g_split.strip()
                if g_clean and g_clean not in all_genres:
                    all_genres.append(g_clean)

    tbpm = tags.get("TBPM")
    if tbpm and hasattr(tbpm, "text") and tbpm.text:
        sources["tbpm"] = True
        try:
            bpm_val = int(round(float(str(tbpm.text[0]).strip())))
        except (ValueError, TypeError):
            pass

    # ── 2. Collect USLT frames ──────────────────────────────────────────────
    uslt_frames = [v for k, v in tags.items() if k.startswith("USLT")]
    if uslt_frames:
        sources["uslt"] = True
        # Prefer frame whose text contains LRC timestamps
        timed_uslt = [f for f in uslt_frames if _has_lrc_timestamps(f.text)]
        if timed_uslt:
            chosen = timed_uslt[0]
            if len(timed_uslt) > 1:
                notes_parts.append(
                    f"Multiple timed USLT frames; using desc='{chosen.desc}'."
                )
            timed_lrc = chosen.text.strip()
        else:
            # No timed USLT — use first non-empty as plain
            for f in uslt_frames:
                if f.text.strip():
                    plain = f.text.strip()
                    notes_parts.append(
                        f"USLT frame contains plain text (desc='{f.desc}')."
                    )
                    break

    # ── 3. TXXX:LYRICS (case-insensitive desc match) ───────────────────────
    txxx_frames = [
        v for k, v in tags.items()
        if k.startswith("TXXX") and getattr(v, "desc", "").strip().upper() in _LYRICS_TXXX_DESCS
    ]
    if txxx_frames:
        sources["txxx_lyrics"] = True
        for frame in txxx_frames:
            txxx_text = frame.text[0] if frame.text else ""
            if not isinstance(txxx_text, str):
                txxx_text = str(txxx_text)
            txxx_text = txxx_text.strip()
            if not txxx_text:
                continue

            if timed_lrc is None and _has_lrc_timestamps(txxx_text):
                timed_lrc = txxx_text
                break

            if plain is None and timed_lrc is None:
                plain = txxx_text

    # ── 4. SYLT frame — convert to LRC as fallback ─────────────────────────
    sylt_frames = [v for k, v in tags.items() if k.startswith("SYLT")]
    if sylt_frames:
        sources["sylt"] = True
        if timed_lrc is None:
            for frame in sylt_frames:
                lrc_from_sylt = _sylt_to_lrc(frame)
                if lrc_from_sylt.strip():
                    timed_lrc = lrc_from_sylt
                    notes_parts.append("Timed LRC derived from SYLT frame.")
                    break

    # ── 5. Check comments for style prompts if not yet found ───────────────
    comm_frames = [v for k, v in tags.items() if k.startswith("COMM")]
    for cf in comm_frames:
        comm_text = str(getattr(cf, "text", [""])[0] if getattr(cf, "text", None) else "")
        if comm_text and not raw_prompt_found:
            m = _STYLE_PREAMBLE_RE.search(comm_text)
            if m:
                raw_prompt_found = (m.group(1) or m.group(2)).strip()

    # ── 6. Derive plain from timed if not already set ─────────────────────
    if timed_lrc and plain is None:
        plain = _strip_lrc_timestamps(timed_lrc)

    # ── 7. Strip style preambles from plain and timed lyrics ───────────────
    if plain:
        plain_clean, prompt_from_plain = strip_style_preamble(plain)
        plain = plain_clean
        if prompt_from_plain and not raw_prompt_found:
            raw_prompt_found = prompt_from_plain

    if timed_lrc:
        # Check if first timed line contains a style preamble
        timed_clean, prompt_from_timed = strip_style_preamble(timed_lrc)
        timed_lrc = timed_clean
        if prompt_from_timed and not raw_prompt_found:
            raw_prompt_found = prompt_from_timed

    # ── 8. Parse discovered style prompt ───────────────────────────────────
    if raw_prompt_found:
        p_genres, p_styles, p_raw = parse_style_prompt(raw_prompt_found)
        all_genres.extend(g for g in p_genres if g not in all_genres)
        all_style_tags.extend(s for s in p_styles if s not in all_style_tags)
        raw_prompt_found = p_raw

    return {
        "plain_lyrics": plain if plain else None,
        "timed_lyrics_lrc": timed_lrc if timed_lrc else None,
        "genres": all_genres,
        "style_tags": all_style_tags,
        "style_prompt_raw": raw_prompt_found,
        "bpm": bpm_val,
        "sources": sources,
        "notes": " ".join(notes_parts) if notes_parts else None,
    }
