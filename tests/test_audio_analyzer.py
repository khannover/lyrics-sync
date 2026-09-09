"""Unit tests for multi-tier audio analyzer cascade."""

import pytest
from unittest.mock import patch
from app.audio_analyzer import analyze_audio_profile


class TestAudioAnalyzerCascade:
    @patch("app.audio_analyzer.extract_lyrics_from_mp3")
    @patch("app.audio_analyzer.extract_audio_features")
    @patch("app.audio_analyzer.classify_audio_genre")
    def test_tier1_match_skips_tier3_unless_forced(
        self, mock_classify, mock_features, mock_lyrics
    ):
        mock_lyrics.return_value = {
            "plain_lyrics": "Hello world",
            "timed_lyrics_lrc": None,
            "genres": ["Techno", "EBM"],
            "style_tags": ["driving bass"],
            "style_prompt_raw": "Techno, EBM, driving bass",
            "bpm": 130,
            "sources": {"uslt": True},
            "notes": None,
        }
        mock_features.return_value = {
            "bpm": 130,
            "energy": "high",
            "key": "A minor",
            "rms": 0.15,
            "spectral_centroid": 3000.0,
            "analysis_time_sec": 0.5,
        }

        # Case 1: force_neural = False -> Tier 3 should NOT run
        profile = analyze_audio_profile("/mock/track.mp3", force_neural=False)
        mock_classify.assert_not_called()
        assert profile["primary_genre"] == "Techno"
        assert profile["genres"] == ["Techno", "EBM"]
        assert profile["bpm"] == 130
        assert profile["energy"] == "high"
        assert profile["tier_breakdown"]["tier3_neural"]["ran"] is False

        # Case 2: force_neural = True -> Tier 3 should run
        mock_classify.return_value = {
            "primary_genre": "Electro",
            "top_genres": [{"genre": "Electro", "label": "Electronic---Electro", "confidence": 0.8}],
            "inference_time_sec": 0.1,
        }
        profile_forced = analyze_audio_profile("/mock/track.mp3", force_neural=True)
        mock_classify.assert_called_once()
        assert profile_forced["primary_genre"] == "Techno"
        assert "Electro" in profile_forced["genres"]
        assert profile_forced["tier_breakdown"]["tier3_neural"]["ran"] is True

    @patch("app.audio_analyzer.extract_lyrics_from_mp3")
    @patch("app.audio_analyzer.extract_audio_features")
    @patch("app.audio_analyzer.classify_audio_genre")
    def test_tier1_empty_triggers_tier3(
        self, mock_classify, mock_features, mock_lyrics
    ):
        mock_lyrics.return_value = {
            "plain_lyrics": "Some lyrics",
            "timed_lyrics_lrc": None,
            "genres": [],
            "style_tags": [],
            "style_prompt_raw": None,
            "bpm": None,
            "sources": {"uslt": True},
            "notes": None,
        }
        mock_features.return_value = {
            "bpm": 100,
            "energy": "low",
            "key": "C major",
            "rms": 0.05,
            "spectral_centroid": 1200.0,
            "analysis_time_sec": 0.4,
        }
        mock_classify.return_value = {
            "primary_genre": "Ambient",
            "top_genres": [
                {"genre": "Ambient", "label": "Electronic---Ambient", "confidence": 0.6},
                {"genre": "Downtempo", "label": "Electronic---Downtempo", "confidence": 0.4},
            ],
            "inference_time_sec": 0.1,
        }

        profile = analyze_audio_profile("/mock/track.mp3", force_neural=False)
        mock_classify.assert_called_once()
        assert profile["primary_genre"] == "Ambient"
        assert "Ambient" in profile["genres"]
        assert profile["bpm"] == 100
        assert profile["energy"] == "low"
        assert profile["tier_breakdown"]["tier3_neural"]["ran"] is True
