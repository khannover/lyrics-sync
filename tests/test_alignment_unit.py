"""Unit tests for alignment helpers that do not invoke Whisper."""

import app.alignment as alignment
from app.main import _build_word_sidecar_payload, _word_level_sylt_entries
from app.alignment import _align_lines_to_words, _detect_language_hint, _tokenize


def test_detect_language_hint_khmer_script():
    assert _detect_language_hint("សួស្តី អ្នកសុខសប្បាយទេ") == "km"


def test_detect_language_hint_mixed_languages_returns_none():
    assert _detect_language_hint("the night que la vida") is None


def test_tokenize_handles_mixed_script_text():
    tokens = _tokenize("Hello—世界 สวัสดี")
    assert "Hello" in tokens
    assert "世" in tokens and "界" in tokens
    assert "สวัสดี" in tokens


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


def test_align_lyrics_to_audio_fallback_builds_proportional_word_timings(monkeypatch, tmp_path):
    lyrics_text = "\n".join(f"line {idx} words" for idx in range(7))

    monkeypatch.setattr(alignment, "_convert_to_wav", lambda *args, **kwargs: str(tmp_path / "audio.wav"))
    monkeypatch.setattr(alignment, "_get_audio_duration_ms", lambda _path: 70_000)
    monkeypatch.setattr(alignment, "_create_vocal_focus_wav", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        alignment,
        "_transcribe_with_word_timestamps",
        lambda *args, **kwargs: alignment.TranscriptionPassResult(
            words=[{"word": "line", "start": 0.1, "end": 0.2}],
            detected_language="en",
            language_probability=0.99,
            pass_name=kwargs.get("pass_name", "fast"),
        ),
    )
    monkeypatch.setattr(
        alignment,
        "_align_lines_to_words",
        lambda lyrics_lines, whisper_words: alignment.StructuredAlignment(
            lines=[(line, 100) for line in lyrics_lines],
            word_timings=[],
        ),
    )
    monkeypatch.setattr(
        alignment,
        "_align_lines_progressive",
        lambda lyrics_lines, whisper_words: alignment.StructuredAlignment(
            lines=[(line, 100) for line in lyrics_lines],
            word_timings=[],
        ),
    )

    result = alignment.align_lyrics_to_audio(
        "track.mp3",
        lyrics_text,
        str(tmp_path),
        timestamp_mode="word",
    )

    assert result.quality == "fallback"
    assert result.report["timestamp_mode"] == "word"
    assert result.word_timings
    assert all(timing.matched is False for timing in result.word_timings)
    assert _word_level_sylt_entries(result) == []

    payload = _build_word_sidecar_payload(result)
    assert payload["timestamp_mode"] == "word"
    assert payload["lines"][0]["words"]
    assert payload["lines"][0]["words"][0]["matched"] is False
