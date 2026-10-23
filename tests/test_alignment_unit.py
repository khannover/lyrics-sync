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


def test_detect_language_hint_ignores_english_stage_directions_in_german_song():
    lyrics = """[Style: Electro Swing, Fast Beat]
[Intro: The DJ drops the bass and you hear the crowd]
Der Postbote klingelt sturm um acht
Ich hab die Tür noch nicht mal aufgemacht
Und das Paket fliegt durch den Raum"""
    assert _detect_language_hint(lyrics) == "de"


def test_align_lines_to_words_extracts_spoken_quotes_from_marker_lines():
    lines = [
        '[Spoken intro: "Ey verpiss dich mal von meinem Tisch"]',
        "Wein glitzert golden in der Nacht",
    ]
    whisper_words = [
        {"word": "Ey", "start": 0.12, "end": 0.30},
        {"word": "verpiss", "start": 0.32, "end": 0.78},
        {"word": "dich", "start": 0.80, "end": 1.00},
        {"word": "mal", "start": 1.02, "end": 1.20},
        {"word": "von", "start": 1.22, "end": 1.40},
        {"word": "meinem", "start": 1.42, "end": 1.80},
        {"word": "Tisch", "start": 1.82, "end": 2.20},
        {"word": "Wein", "start": 22.82, "end": 23.20},
        {"word": "glitzert", "start": 23.25, "end": 23.80},
        {"word": "golden", "start": 23.85, "end": 24.30},
        {"word": "in", "start": 24.35, "end": 24.50},
        {"word": "der", "start": 24.55, "end": 24.75},
        {"word": "Nacht", "start": 24.80, "end": 25.30},
    ]
    result = _align_lines_to_words(lines, whisper_words)
    assert result.lines[0][1] == 120
    assert result.lines[1][1] == 22820


def test_align_lines_to_words_picks_best_candidate_instead_of_median_across_gap():
    lines = [
        "von meinem Tisch",
        "Wein glitzert",
    ]
    whisper_words = [
        {"word": "von", "start": 1.00, "end": 1.20},
        {"word": "meinem", "start": 1.25, "end": 1.50},
        {"word": "Tisch", "start": 1.55, "end": 1.85},
        {"word": "noise", "start": 5.00, "end": 5.30},
        {"word": "echo", "start": 9.00, "end": 9.30},
        {"word": "Wein", "start": 22.82, "end": 23.20},
        {"word": "glitzert", "start": 23.25, "end": 23.80},
    ]
    result = _align_lines_to_words(lines, whisper_words)
    assert result.lines[1][1] == 22820


def test_build_contiguous_clip_timestamps_and_dedupe():
    clips = alignment._build_contiguous_clip_timestamps(75_000)
    assert clips == [0.0, 30.0, 30.0, 60.0, 60.0, 75.0]

    raw = [
        {"word": "Auge", "start": 10.0, "end": 10.2},
        {"word": "Auge", "start": 10.3, "end": 10.5},
        {"word": "Auge", "start": 10.6, "end": 10.8},
        {"word": "Auge", "start": 10.9, "end": 11.1},
        {"word": "weiter", "start": 11.5, "end": 11.9},
    ]
    deduped = alignment._dedupe_hallucinated_words(raw)
    assert [w["word"] for w in deduped] == ["Auge", "Auge", "weiter"]


def test_align_lines_to_words_rejects_anagram_false_match():
    lines = [
        "Warm intro words",
        "Listen to the night",
    ]
    whisper_words = [
        {"word": "warm", "start": 1.00, "end": 1.30},
        {"word": "intro", "start": 1.35, "end": 1.70},
        {"word": "words", "start": 1.75, "end": 2.10},
        {"word": "silent", "start": 2.50, "end": 2.90},
        {"word": "listen", "start": 12.00, "end": 12.40},
        {"word": "to", "start": 12.45, "end": 12.60},
        {"word": "the", "start": 12.65, "end": 12.80},
        {"word": "night", "start": 12.85, "end": 13.30},
    ]
    result = _align_lines_to_words(lines, whisper_words)
    assert result.lines[0][1] == 1000
    assert result.lines[1][1] == 12000


def test_align_lines_to_words_filters_pre_gap_outlier_word_from_line_timestamp():
    lines = [
        "Ich steh im Regen",
        "Und der Wein glitzert golden",
    ]
    whisper_words = [
        {"word": "Ich", "start": 1.00, "end": 1.20},
        {"word": "steh", "start": 1.25, "end": 1.50},
        {"word": "im", "start": 1.55, "end": 1.75},
        {"word": "Regen", "start": 1.80, "end": 2.20},
        # Whisper missed "Und" after a 20s instrumental gap and starts at "der"
        {"word": "der", "start": 22.82, "end": 23.00},
        {"word": "Wein", "start": 23.05, "end": 23.40},
        {"word": "glitzert", "start": 23.45, "end": 23.95},
        {"word": "golden", "start": 24.00, "end": 24.50},
    ]
    result = _align_lines_to_words(lines, whisper_words)
    assert result.lines[0][1] == 1000
    # Line 1 must anchor right before "der" (~22620ms), NOT back at "Regen" (1800ms)
    assert 22000 <= result.lines[1][1] <= 22820
    line1_words = [wt for wt in result.word_timings if wt.line_index == 1]
    assert line1_words[0].word == "Und"
    assert line1_words[0].matched is False
    assert line1_words[0].start_ms == result.lines[1][1]
    assert line1_words[1].word == "der"
    assert line1_words[1].matched is True
    assert line1_words[1].start_ms == 22820


def test_lyric_word_entries_strips_inline_stage_cues_but_keeps_backing_vocals():
    lines = [
        "[Verse 1] Walking in the rain (x2)",
        "Chorus: We shine (oh yeah) [Guitar Solo]",
    ]
    entries = alignment._lyric_word_entries(lines)
    words_line0 = [e["normalized"] for e in entries if e["line_index"] == 0]
    words_line1 = [e["normalized"] for e in entries if e["line_index"] == 1]
    assert words_line0 == ["walking", "in", "the", "rain"]
    assert words_line1 == ["we", "shine", "oh", "yeah"]


def test_align_lines_to_words_phonetic_normalization_and_midline_interpolation():
    lines = ["Yeah we loooove MysteryToken walkin' home"]
    whisper_words = [
        {"word": "yeah", "start": 5.00, "end": 5.30},
        {"word": "we", "start": 5.35, "end": 5.55},
        {"word": "love", "start": 5.60, "end": 6.00},
        {"word": "walking", "start": 6.80, "end": 7.20},
        {"word": "home", "start": 7.25, "end": 7.70},
    ]
    result = _align_lines_to_words(lines, whisper_words)
    wt_by_word = {wt.word: wt for wt in result.word_timings}
    assert wt_by_word["loooove"].matched is True
    assert wt_by_word["walkin'"].matched is True
    assert wt_by_word["MysteryToken"].matched is False
    assert 6000 <= wt_by_word["MysteryToken"].start_ms <= 6800
    assert wt_by_word["MysteryToken"].start_ms < wt_by_word["MysteryToken"].end_ms


def test_align_lines_progressive_recovers_from_unmappable_line_without_jumping_cursor():
    lines = [
        "walking down the lonely street",
        "completely unmappable phantom lyric line",
        "shining bright under the moon",
    ]
    whisper_words = [
        {"word": "walking", "start": 2.0, "end": 2.4},
        {"word": "down", "start": 2.5, "end": 2.8},
        {"word": "the", "start": 2.9, "end": 3.1},
        {"word": "lonely", "start": 3.2, "end": 3.6},
        {"word": "street", "start": 3.7, "end": 4.1},
        {"word": "shining", "start": 8.0, "end": 8.4},
        {"word": "bright", "start": 8.5, "end": 8.9},
        {"word": "under", "start": 9.0, "end": 9.3},
        {"word": "the", "start": 9.4, "end": 9.6},
        {"word": "moon", "start": 9.7, "end": 10.2},
        # Distant noise near the end of the song that shares a slight character overlap
        {"word": "phantom", "start": 120.0, "end": 120.5},
    ]
    result = alignment._align_lines_progressive(lines, whisper_words)
    assert result.lines[0][1] == 2000
    # Line 1 was unmappable locally and must be interpolated between Line 0 (4.1s) and Line 2 (8.0s)
    assert 4100 <= result.lines[1][1] < 8000
    # Line 2 must still match at 8000ms instead of being pushed past 120000ms
    assert result.lines[2][1] == 8000


