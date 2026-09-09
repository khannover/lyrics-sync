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
        assert "content_rating" in profile
        assert "copyright_fingerprint" in profile
        assert "ai_provenance" in profile

    @patch("app.audio_analyzer.extract_lyrics_from_mp3")
    @patch("app.audio_analyzer.generate_audio_fingerprint")
    @patch("app.audio_analyzer.detect_ai_provenance")
    def test_safety_and_provenance_integration(
        self, mock_prov, mock_fp, mock_lyrics
    ):
        mock_lyrics.return_value = {
            "plain_lyrics": "Fuck this holy shit",
            "timed_lyrics_lrc": None,
            "genres": ["Punk"],
            "style_tags": [],
            "style_prompt_raw": None,
            "bpm": 150,
            "sources": {"uslt": True},
            "notes": None,
        }
        mock_fp.return_value = {
            "acoustid_fingerprint": "AQADTEST12345",
            "duration_sec": 120.0,
            "algorithm": "chromaprint",
        }
        mock_prov.return_value = {
            "is_synthetic": True,
            "confidence": 0.99,
            "detected_source": "suno",
            "signals": {"metadata_marker": True},
        }

        profile = analyze_audio_profile("/mock/track.mp3", include_signal=False)
        assert profile["content_rating"]["is_explicit"] is True
        assert profile["content_rating"]["rating"] == "explicit"
        assert "profanity" in profile["content_rating"]["categories"]
        assert profile["copyright_fingerprint"]["acoustid_fingerprint"] == "AQADTEST12345"
        assert profile["ai_provenance"]["is_synthetic"] is True
        assert profile["ai_provenance"]["detected_source"] == "suno"
