"""Unit tests for Tier 1 style prompt parsing and ID3 tag extraction."""

import pytest
from app.lyrics_tag_reader import (
    parse_style_prompt,
    strip_style_preamble,
    extract_lyrics_from_mp3,
)


class TestParseStylePrompt:
    def test_parse_standard_bracketed_prompt(self):
        prompt = "[Style: High-energy Techno, EBM, driving bass, female vocals, NO SLOP]"
        genres, tags, raw = parse_style_prompt(prompt)
        assert "High-Energy Techno" in genres or "Techno" in genres
        assert "Ebm" in genres
        assert "driving bass" in tags
        assert "female vocals" in tags
        assert "NO SLOP" not in tags  # Ignored junk token
        assert "High-energy Techno" in raw

    def test_parse_colon_format(self):
        prompt = "Style: Dark Synthwave, Cyberpunk, 120bpm, aggressive synthesizer"
        genres, tags, raw = parse_style_prompt(prompt)
        assert "Dark Synthwave" in genres or "Synthwave" in genres
        assert "Cyberpunk" in genres
        assert "aggressive synthesizer" in tags

    def test_parse_empty_or_none(self):
        genres, tags, raw = parse_style_prompt("")
        assert genres == []
        assert tags == []
        assert raw is None

    def test_parse_genre_deduplication(self):
        prompt = "[Genre: Rock, Hard Rock, rock, Alternative]"
        genres, tags, raw = parse_style_prompt(prompt)
        assert len(genres) == len(set(genres))


class TestStripStylePreamble:
    def test_strips_bracketed_style_preamble(self):
        lyrics = "[Style: Heavy Metal, fast drums]\n[Verse 1]\nThunder rolling"
        clean, raw = strip_style_preamble(lyrics)
        assert raw == "Heavy Metal, fast drums"
        assert clean == "[Verse 1]\nThunder rolling"

    def test_leaves_normal_lyrics_untouched(self):
        lyrics = "[Verse 1]\nJust a regular song\n[Chorus]\nSinging along"
        clean, raw = strip_style_preamble(lyrics)
        assert raw is None
        assert clean == lyrics

    def test_handles_empty_string(self):
        clean, raw = strip_style_preamble("")
        assert clean == ""
        assert raw is None

    def test_does_not_strip_style_deep_in_lyrics(self):
        lyrics = "[Verse 1]\nLine 1\nLine 2\n[Style: Pop]\nLine 3" * 5
        clean, raw = strip_style_preamble(lyrics)
        # Should not strip if far past beginning
        assert "[Verse 1]" in clean
