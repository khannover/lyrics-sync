"""Unit tests for app.lyrics_tag_reader."""

import os
import tempfile
import pytest

from mutagen.id3 import ID3, SYLT, USLT, TXXX, Encoding
from mutagen.mp3 import MP3

# We need a minimal valid MP3 to attach ID3 tags to.
# This is a 1-frame silent MPEG1 Layer3 audio blob (128 kbps, 44.1 kHz, stereo).
_SILENT_MP3_BYTES = bytes(
    [
        # ID3v2.3 header: "ID3" + version 2.3 + flags 0 + size 0
        0x49, 0x44, 0x33, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        # One minimal MPEG audio frame (header only, all zeros payload)
        0xFF, 0xFB, 0x90, 0x00,
    ] + [0x00] * 413  # silent frame data (417 bytes total for 128kbps)
)


def _make_mp3(tmp_path: str) -> str:
    """Write a minimal MP3 file and return its path."""
    path = os.path.join(tmp_path, "test.mp3")
    with open(path, "wb") as f:
        f.write(_SILENT_MP3_BYTES)
    return path


# ── Import target after ensuring path is accessible ──────────────────────────

from app.lyrics_tag_reader import (
    _strip_lrc_timestamps,
    _has_lrc_timestamps,
    _sylt_to_lrc,
    extract_lyrics_from_mp3,
)


# ── _strip_lrc_timestamps ─────────────────────────────────────────────────────

class TestStripLrcTimestamps:
    def test_strips_standard_timestamps(self):
        lrc = "[00:01.00]Hello\n[00:05.50]World"
        assert _strip_lrc_timestamps(lrc) == "Hello\nWorld"

    def test_strips_single_digit_minute(self):
        lrc = "[1:30]Line one"
        assert _strip_lrc_timestamps(lrc) == "Line one"

    def test_ignores_lines_without_timestamps(self):
        text = "No timestamp here"
        assert _strip_lrc_timestamps(text) == "No timestamp here"

    def test_skips_empty_lines_after_strip(self):
        lrc = "[00:01.00]   \n[00:02.00]Real line"
        assert _strip_lrc_timestamps(lrc) == "Real line"

    def test_empty_string(self):
        assert _strip_lrc_timestamps("") == ""


# ── _has_lrc_timestamps ───────────────────────────────────────────────────────

class TestHasLrcTimestamps:
    def test_detects_timestamp(self):
        assert _has_lrc_timestamps("[01:23.45]Some lyric")

    def test_no_timestamp(self):
        assert not _has_lrc_timestamps("Just plain text")

    def test_partial_match_in_multiline(self):
        text = "plain line\n[00:10.00]timed line"
        assert _has_lrc_timestamps(text)


# ── _sylt_to_lrc ──────────────────────────────────────────────────────────────

class TestSyltToLrc:
    def _make_sylt(self, entries):
        return SYLT(
            encoding=Encoding.UTF8,
            lang="eng",
            desc="",
            format=2,   # milliseconds
            type=1,
            text=entries,
        )

    def test_basic_conversion(self):
        sylt = self._make_sylt([("Hello\n", 1000), ("World\n", 5000)])
        lrc = _sylt_to_lrc(sylt)
        assert "[00:01.00]Hello" in lrc
        assert "[00:05.00]World" in lrc

    def test_empty_entries_skipped(self):
        sylt = self._make_sylt([("\n", 1000), ("Real\n", 2000)])
        lrc = _sylt_to_lrc(sylt)
        assert "Real" in lrc
        lines = [l for l in lrc.splitlines() if l.strip()]
        assert len(lines) == 1

    def test_empty_sylt(self):
        sylt = self._make_sylt([])
        assert _sylt_to_lrc(sylt) == ""


# ── extract_lyrics_from_mp3 ───────────────────────────────────────────────────

class TestExtractLyricsFromMp3:
    def test_no_tags_returns_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.mp3")
            # Write raw bytes with no ID3 header
            with open(path, "wb") as f:
                f.write(b"\xFF\xFB\x90\x00" + b"\x00" * 413)
            result = extract_lyrics_from_mp3(path)
        assert result["plain_lyrics"] is None
        assert result["timed_lyrics_lrc"] is None
        assert result["sources"] == {"uslt": False, "txxx_lyrics": False, "sylt": False}

    def test_uslt_timed_extracted(self):
        lrc_text = "[00:01.00]Line one\n[00:05.00]Line two"
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            # mutagen may raise on our stub; use ID3() directly
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=lrc_text))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["sources"]["uslt"] is True
        assert result["timed_lyrics_lrc"] == lrc_text
        assert result["plain_lyrics"] == "Line one\nLine two"

    def test_uslt_plain_extracted(self):
        plain_text = "Line one\nLine two"
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=plain_text))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["sources"]["uslt"] is True
        assert result["timed_lyrics_lrc"] is None
        assert result["plain_lyrics"] == plain_text

    def test_txxx_timed_extracted_when_no_uslt(self):
        lrc_text = "[00:01.00]Hello\n[00:02.00]World"
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(TXXX(encoding=Encoding.UTF8, desc="LYRICS", text=[lrc_text]))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["sources"]["txxx_lyrics"] is True
        assert result["timed_lyrics_lrc"] == lrc_text

    def test_sylt_fallback_when_no_uslt_txxx(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(SYLT(
                encoding=Encoding.UTF8, lang="eng", desc="", format=2, type=1,
                text=[("Hello\n", 1000), ("World\n", 5000)],
            ))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["sources"]["sylt"] is True
        assert result["timed_lyrics_lrc"] is not None
        assert "[00:01.00]Hello" in result["timed_lyrics_lrc"]
        assert result["plain_lyrics"] == "Hello\nWorld"

    def test_uslt_timed_preferred_over_plain(self):
        """When both timed and plain USLT frames exist, timed is chosen."""
        lrc_text = "[00:01.00]Timed line"
        plain_text = "Plain line"
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="synced", text=lrc_text))
            tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="plain", text=plain_text))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["timed_lyrics_lrc"] == lrc_text
        assert result["plain_lyrics"] == "Timed line"

    def test_all_sources_reported(self):
        lrc_text = "[00:01.00]Hi"
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=lrc_text))
            tags.add(TXXX(encoding=Encoding.UTF8, desc="LYRICS", text=[lrc_text]))
            tags.add(SYLT(
                encoding=Encoding.UTF8, lang="eng", desc="", format=2, type=1,
                text=[("Hi\n", 1000)],
            ))
            tags.save(path)

            result = extract_lyrics_from_mp3(path)

        assert result["sources"] == {"uslt": True, "txxx_lyrics": True, "sylt": True}
