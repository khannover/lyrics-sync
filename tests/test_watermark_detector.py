"""Unit tests for AI watermark and provenance detection."""

import os
import pytest
from app.watermark_detector import detect_ai_provenance


class TestWatermarkDetector:
    def test_missing_file_handles_gracefully(self):
        res = detect_ai_provenance("/nonexistent/file.mp3")
        assert res["is_synthetic"] is False
        assert res["confidence"] == 0.0
        assert res["detected_source"] is None
        assert res["signals"]["metadata_marker"] is False

    def test_suno_track_detection(self):
        test_track = "/tmp/track1.mp3"
        if not os.path.exists(test_track):
            pytest.skip("Test track not present")

        res = detect_ai_provenance(test_track)
        assert res["is_synthetic"] is True
        assert res["confidence"] >= 0.90
        assert res["detected_source"] == "suno"
        assert res["signals"]["metadata_marker"] is True
        assert res["signals"]["spectral_cutoff_khz"] > 0
