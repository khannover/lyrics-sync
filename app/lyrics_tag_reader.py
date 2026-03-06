"""
Extract embedded lyrics from MP3 ID3 tags.

Reads USLT, TXXX:LYRICS, and SYLT frames and returns both
timed (LRC) and plain (stripped) representations.
"""

import re
from typing import Optional
from mutagen.id3 import ID3, SYLT
from mutagen.mp3 import MP3

# Matches LRC timestamp: [MM:SS.xx] or [M:SS] or [M:SS.xx]
_LRC_TS_RE = re.compile(r"^\[\d+:\d{2}(?:\.\d+)?\]")
_LRC_LINE_RE = re.compile(r"^\[(\d+):(\d{2}(?:\.\d+)?)\](.*)")


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


def extract_lyrics_from_mp3(mp3_path: str) -> dict:
    """
    Extract embedded lyrics from an MP3 file's ID3 tags.

    Returns a dict with:
      - plain_lyrics: str | None
      - timed_lyrics_lrc: str | None
      - sources: { uslt: bool, txxx_lyrics: bool, sylt: bool }
      - notes: str | None
    """
    sources = {"uslt": False, "txxx_lyrics": False, "sylt": False}
    timed_lrc: Optional[str] = None
    plain: Optional[str] = None
    notes_parts = []

    try:
        tags = ID3(mp3_path)
    except Exception:
        # No ID3 tags or unreadable file — return empty result
        return {
            "plain_lyrics": None,
            "timed_lyrics_lrc": None,
            "sources": sources,
            "notes": "No ID3 tags found.",
        }

    # ── 1. Collect USLT frames ──────────────────────────────────────────────
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

    # ── 2. TXXX:LYRICS (case-insensitive desc match) ───────────────────────
    txxx_frames = [
        v for k, v in tags.items()
        if k.startswith("TXXX") and v.desc.upper() == "LYRICS"
    ]
    if txxx_frames:
        sources["txxx_lyrics"] = True
        txxx_text = txxx_frames[0].text[0] if txxx_frames[0].text else ""
        if txxx_text.strip():
            if timed_lrc is None and _has_lrc_timestamps(txxx_text):
                timed_lrc = txxx_text.strip()
            elif plain is None and timed_lrc is None:
                plain = txxx_text.strip()

    # ── 3. SYLT frame — convert to LRC as fallback ─────────────────────────
    sylt_frames = [v for k, v in tags.items() if k.startswith("SYLT")]
    if sylt_frames:
        sources["sylt"] = True
        if timed_lrc is None:
            lrc_from_sylt = _sylt_to_lrc(sylt_frames[0])
            if lrc_from_sylt.strip():
                timed_lrc = lrc_from_sylt
                notes_parts.append("Timed LRC derived from SYLT frame.")

    # ── Derive plain from timed if not already set ─────────────────────────
    if timed_lrc and plain is None:
        plain = _strip_lrc_timestamps(timed_lrc)

    return {
        "plain_lyrics": plain if plain else None,
        "timed_lyrics_lrc": timed_lrc if timed_lrc else None,
        "sources": sources,
        "notes": " ".join(notes_parts) if notes_parts else None,
    }
