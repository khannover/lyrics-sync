"""Unit tests for NSFW and explicit content filtering."""

import pytest
from app.content_filter import evaluate_content_rating


class TestContentFilter:
    def test_clean_lyrics(self):
        lyrics = (
            "The morning sun rises high\n"
            "Birds are singing in the sky\n"
            "We dance together in the rain"
        )
        res = evaluate_content_rating(lyrics)
        assert res["is_explicit"] is False
        assert res["rating"] == "clean"
        assert res["categories"] == []
        assert res["matched_count"] == 0

    def test_empty_and_whitespace(self):
        assert evaluate_content_rating("")["rating"] == "clean"
        assert evaluate_content_rating("   \n\t  ")["rating"] == "clean"
        assert evaluate_content_rating(None)["rating"] == "clean"

    def test_mild_profanity_single(self):
        lyrics = "Damn it all, hell has no fury like this."
        res = evaluate_content_rating(lyrics)
        assert res["rating"] == "mild"
        assert res["is_explicit"] is False
        assert "profanity" in res["categories"]
        assert res["matched_count"] >= 1

    def test_mild_profanity_excessive_triggers_explicit(self):
        lyrics = "Damn damn hell crap ass damn"
        res = evaluate_content_rating(lyrics)
        assert res["rating"] == "mild"
        assert res["is_explicit"] is True

    def test_strong_profanity(self):
        lyrics = "Get the fuck out of here, holy shit!"
        res = evaluate_content_rating(lyrics)
        assert res["is_explicit"] is True
        assert res["rating"] == "explicit"
        assert "profanity" in res["categories"]
        assert any("fuck" in t for t in res["matched_terms"])

    def test_sexual_content(self):
        lyrics = "Baby give me a blowjob in the back of the car."
        res = evaluate_content_rating(lyrics)
        assert res["is_explicit"] is True
        assert res["rating"] == "explicit"
        assert "sexual_content" in res["categories"]

    def test_hate_speech(self):
        lyrics = "Some hateful line with a slur faggot here."
        res = evaluate_content_rating(lyrics)
        assert res["is_explicit"] is True
        assert res["rating"] == "explicit"
        assert "hate_speech" in res["categories"]

    def test_violence(self):
        lyrics = "I will slit your throat in a bloodbath."
        res = evaluate_content_rating(lyrics)
        assert res["is_explicit"] is True
        assert res["rating"] == "explicit"
        assert "violence" in res["categories"]
