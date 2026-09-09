"""Unit tests for Tier 2 acoustic signal feature extraction."""

import numpy as np
import pytest
from app.audio_features import estimate_musical_key, extract_audio_features


class TestEstimateMusicalKey:
    def test_c_major_profile_correlation(self):
        # Simulated C major chroma: C (0), E (4), G (7) dominate
        chroma = np.zeros(12)
        chroma[0] = 1.0  # C
        chroma[4] = 0.8  # E
        chroma[7] = 0.9  # G
        chroma[2] = 0.3  # D
        chroma[9] = 0.3  # A
        chroma[11] = 0.3 # B
        key, conf = estimate_musical_key(chroma)
        assert "major" in key
        assert key.startswith("C")

    def test_a_minor_profile_correlation(self):
        # Simulated A minor chroma: A (9), C (0), E (4) dominate
        chroma = np.zeros(12)
        chroma[9] = 1.0  # A
        chroma[0] = 0.8  # C
        chroma[4] = 0.9  # E
        chroma[2] = 0.3  # D
        chroma[7] = 0.3  # G
        key, conf = estimate_musical_key(chroma)
        assert "minor" in key
        assert key.startswith("A")

    def test_zero_chroma_handles_gracefully(self):
        chroma = np.zeros(12)
        key, conf = estimate_musical_key(chroma)
        assert isinstance(key, str)
        assert isinstance(conf, float)


class TestExtractAudioFeatures:
    def test_missing_file_returns_defaults(self):
        res = extract_audio_features("/nonexistent/audio.mp3")
        assert res["bpm"] is None
        assert res["energy"] == "medium"
        assert res["key"] is None
        assert res["rms"] == 0.0
