"""Unit tests for Chromaprint / AcoustID fingerprinting."""

import os
import pytest
from app.audio_fingerprint import generate_audio_fingerprint


class TestAudioFingerprint:
    def test_missing_file_returns_none(self):
        res = generate_audio_fingerprint("/nonexistent/audio_file.mp3")
        assert res["acoustid_fingerprint"] is None
        assert res["duration_sec"] is None
        assert res["algorithm"] == "chromaprint"

    def test_real_audio_fingerprint(self):
        # Use container track if exists
        test_track = "/tmp/track1.mp3"
        if not os.path.exists(test_track):
            pytest.skip("Test track not present")

        res = generate_audio_fingerprint(test_track)
        assert res["acoustid_fingerprint"] is not None
        assert len(res["acoustid_fingerprint"]) > 50
        assert res["duration_sec"] is not None
        assert res["duration_sec"] > 10.0
        assert res["algorithm"] == "chromaprint"
