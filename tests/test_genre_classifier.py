"""Unit tests for Tier 3 neural genre classifier."""

import pytest
from app.genre_classifier import classify_audio_genre, _find_model_files


class TestGenreClassifier:
    def test_model_files_exist(self):
        onnx_path, json_path = _find_model_files()
        assert onnx_path is not None, "ONNX model not found"
        assert json_path is not None, "JSON labels not found"
        assert onnx_path.exists()
        assert json_path.exists()

    def test_missing_audio_file_handles_gracefully(self):
        res = classify_audio_genre("/nonexistent/file.mp3")
        assert res["top_genres"] == []
        assert res["primary_genre"] is None
        assert res["inference_time_sec"] >= 0.0
