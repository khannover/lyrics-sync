"""Unit tests for alignment helpers that do not invoke Whisper."""

from app.alignment import (
    _align_lines_to_words,
    _detect_language_hint,
    _tokenize,
)


def test_detect_language_hint_khmer_script():
    assert _detect_language_hint("សួស្តី អ្នកសុខសប្បាយទេ") == "km"


def test_detect_language_hint_mixed_languages_returns_none():
    assert _detect_language_hint("the night que la vida") is None


def test_tokenize_handles_mixed_script_text():
    tokens = _tokenize("Hello—世界 สวัสดี")
    assert "Hello" in tokens
    assert "世" in tokens and "界" in tokens
    assert "ส" in tokens


def test_tokenize_keeps_hangul_words_grouped():
    assert _tokenize("안녕 세상") == ["안녕", "세상"]


def test_align_lines_to_words_returns_word_timings():
    result = _align_lines_to_words(
        ["Hello world"],
        [
            {"word": "hello", "start": 0.5, "end": 0.8},
            {"word": "world", "start": 0.9, "end": 1.2},
        ],
    )

    assert result.lines[0][1] == 500
    assert len(result.word_timings) == 2
    assert result.word_timings[0].word.lower() == "hello"
    assert result.word_timings[0].matched is True
