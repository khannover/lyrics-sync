"""
Write synchronized lyrics into MP3 ID3 tags.

Strategy for maximum player compatibility:
1. SYLT frame  — ID3v2 spec standard (few players support it)
2. USLT frame  — with embedded LRC-format timestamps (widely supported)
3. LYRICS3v2   — legacy support via USLT
4. SYLT as custom TXXX — some players look here
"""

from typing import List, Tuple
import logging
from mutagen.id3 import ID3, SYLT, USLT, TXXX, Encoding
from mutagen.mp3 import MP3

logger = logging.getLogger(__name__)


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
    embed_mode: str = "overwrite",
):
    """
    Write synchronized lyrics into MP3 ID3 tags.

    embed_mode="overwrite" (default):
        Removes SYLT, USLT, TXXX:LYRICS frames, then writes:
        - SYLT frame (ID3v2 standard)
        - USLT frame (desc="synced") with LRC text
        - USLT frame (desc="")       with LRC text
        - TXXX:LYRICS with LRC text

    embed_mode="sylt_only":
        Removes only SYLT frames, then writes only the SYLT frame.
        Existing USLT / TXXX:LYRICS (plain unsynced lyrics) are preserved.
    """
    audio = MP3(mp3_path)

    if audio.tags is None:
        audio.add_tags()

    if embed_mode == "sylt_only":
        # --- Remove only existing SYLT frames ---
        to_remove = [key for key in audio.tags if key.startswith("SYLT")]
    else:
        # --- Remove existing SYLT, USLT, and TXXX:LYRICS frames ---
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

    # ID3v2.3 does not support UTF-8 text encoding in a fully portable way.
    # Use UTF-16 to preserve emoji/non-Latin text while keeping v2.3 compatibility.
    frame_encoding = Encoding.UTF16

    audio.tags.add(SYLT(
        encoding=frame_encoding,
        lang=lang,
        desc=desc,
        format=2,   # milliseconds
        type=1,      # lyrics
        text=sylt_data,
    ))

    if embed_mode != "sylt_only":
        # --- 2. USLT frame with embedded LRC timestamps ---
        lrc_text = _build_lrc(synced_lyrics)

        audio.tags.add(USLT(
            encoding=frame_encoding,
            lang=lang,
            desc="synced",
            text=lrc_text,
        ))

        # Also add a plain USLT without desc for players that expect that
        audio.tags.add(USLT(
            encoding=frame_encoding,
            lang=lang,
            desc="",
            text=lrc_text,
        ))

        # --- 3. TXXX:LYRICS fallback ---
        audio.tags.add(TXXX(
            encoding=frame_encoding,
            desc="LYRICS",
            text=[lrc_text],
        ))

    try:
        audio.save(v2_version=3)  # ID3v2.3 for max compatibility
    except UnicodeEncodeError:
        # Last-resort fallback for edge-case chars that fail in v2.3 conversion.
        logger.warning("Unicode encode issue saving v2.3 tags, retrying as ID3v2.4")
        audio.save(v2_version=4)


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