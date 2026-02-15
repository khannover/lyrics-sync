"""
Write synchronized lyrics into MP3 ID3 tags.

Strategy for maximum player compatibility:
1. SYLT frame  — ID3v2 spec standard (few players support it)
2. USLT frame  — with embedded LRC-format timestamps (widely supported)
3. LYRICS3v2   — legacy support via USLT
4. SYLT as custom TXXX — some players look here
"""

from typing import List, Tuple
from mutagen.id3 import ID3, SYLT, USLT, TXXX, Encoding
from mutagen.mp3 import MP3


def _ms_to_lrc_timestamp(ms: int) -> str:
    """Convert milliseconds to LRC format [MM:SS.xx]"""
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000
    return f"[{minutes:02d}:{seconds:05.2f}]"


def _build_lrc(synced_lyrics: List[Tuple[str, int]]) -> str:
    """Build an LRC-format string from synced lyrics."""
    lines = []
    for text, timestamp_ms in synced_lyrics:
        lrc_ts = _ms_to_lrc_timestamp(timestamp_ms)
        lines.append(f"{lrc_ts}{text}")
    return "\n".join(lines)


def write_sylt_tag(
    mp3_path: str,
    synced_lyrics: List[Tuple[str, int]],
    lang: str = "eng",
    desc: str = "",
):
    """
    Write synchronized lyrics in multiple formats for broad player support.

    Writes:
    - SYLT frame (ID3v2 standard)
    - USLT frame with LRC-formatted text (AIMP, MusicBee, foobar2000, etc.)
    - TXXX:LYRICS with LRC text (fallback for some players)
    """
    audio = MP3(mp3_path)

    if audio.tags is None:
        audio.add_tags()

    # --- Remove existing lyrics frames ---
    to_remove = [key for key in audio.tags
                 if key.startswith("SYLT")
                 or key.startswith("USLT")
                 or key.startswith("TXXX:LYRICS")]
    for key in to_remove:
        del audio.tags[key]

    # --- 1. SYLT frame (spec-correct) ---
    sylt_data = []
    for text, timestamp_ms in synced_lyrics:
        sylt_data.append((text + "\n", timestamp_ms))

    audio.tags.add(SYLT(
        encoding=Encoding.UTF8,
        lang=lang,
        desc=desc,
        format=2,   # milliseconds
        type=1,      # lyrics
        text=sylt_data,
    ))

    # --- 2. USLT frame with embedded LRC timestamps ---
    lrc_text = _build_lrc(synced_lyrics)

    audio.tags.add(USLT(
        encoding=Encoding.UTF8,
        lang=lang,
        desc="synced",
        text=lrc_text,
    ))

    # Also add a plain USLT without desc for players that expect that
    audio.tags.add(USLT(
        encoding=Encoding.UTF8,
        lang=lang,
        desc="",
        text=lrc_text,
    ))

    # --- 3. TXXX:LYRICS fallback ---
    audio.tags.add(TXXX(
        encoding=Encoding.UTF8,
        desc="LYRICS",
        text=[lrc_text],
    ))

    audio.save(v2_version=3)  # ID3v2.3 for max compatibility


def write_lrc_file(
    lrc_path: str,
    synced_lyrics: List[Tuple[str, int]],
    metadata: dict = None,
):
    """
    Also write a standalone .lrc file.
    Many players auto-detect LRC files next to the MP3.

    Place the .lrc file next to the .mp3 with the same base name.
    """
    lines = []

    if metadata:
        if "title" in metadata:
            lines.append(f"[ti:{metadata['title']}]")
        if "artist" in metadata:
            lines.append(f"[ar:{metadata['artist']}]")
        lines.append("")

    for text, timestamp_ms in synced_lyrics:
        lrc_ts = _ms_to_lrc_timestamp(timestamp_ms)
        lines.append(f"{lrc_ts}{text}")

    with open(lrc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")