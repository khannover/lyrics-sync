"""
Unified Multi-Tier Audio and Genre Analysis Cascade for lyrics-sync.

Cascade:
- Tier 1 (Instant, 0ms): ID3 tags (TCON, TBPM) + Suno/Udio lyrics prompt parsing ([Style: ...])
- Tier 2 (Signal, ~1.5s): librosa acoustic features (exact BPM, energy level, musical key)
- Tier 3 (Neural, ~100ms): Discogs-EffNet ONNX deep audio genre classifier
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from app.lyrics_tag_reader import extract_lyrics_from_mp3
from app.audio_features import extract_audio_features
from app.genre_classifier import classify_audio_genre

logger = logging.getLogger(__name__)


def analyze_audio_profile(
    mp3_path: str,
    include_signal: bool = True,
    force_neural: bool = False,
    top_k_genres: int = 5,
) -> Dict[str, Any]:
    """
    Run multi-tier acoustic and metadata analysis on an MP3 file.

    Parameters:
      mp3_path: Absolute or relative path to MP3 file.
      include_signal: If True, run Tier 2 acoustic signal analysis (BPM, key, energy).
      force_neural: If True, always run Tier 3 neural genre classifier even if
                    Tier 1 found genres in tags or lyrics prompt.
      top_k_genres: Number of top neural genre predictions to include.

    Returns:
      Comprehensive audio profile dictionary.
    """
    total_start = time.monotonic()

    # ── Tier 1: Instant Metadata & Prompt Parsing ────────────────────────────
    tier1 = extract_lyrics_from_mp3(mp3_path)
    t1_genres = list(tier1.get("genres", []))
    style_tags = list(tier1.get("style_tags", []))
    raw_prompt = tier1.get("style_prompt_raw")
    id3_bpm = tier1.get("bpm")

    # ── Tier 2: Acoustic Signal Features ─────────────────────────────────────
    t2_result: Dict[str, Any] = {
        "bpm": None,
        "energy": "unknown",
        "key": None,
        "rms": 0.0,
        "spectral_centroid": 0.0,
        "analysis_time_sec": 0.0,
    }
    if include_signal:
        try:
            t2_result = extract_audio_features(mp3_path)
        except Exception as exc:
            logger.warning("Tier 2 signal extraction failed for %s: %s", mp3_path, exc)

    # ── Tier 3: Neural Genre Classification ──────────────────────────────────
    # Run if explicitly forced, or if Tier 1 yielded no genres
    should_run_neural = force_neural or (len(t1_genres) == 0)
    t3_result: Dict[str, Any] = {
        "ran": False,
        "primary_genre": None,
        "top_genres": [],
        "inference_time_sec": 0.0,
    }

    if should_run_neural:
        try:
            raw_t3 = classify_audio_genre(mp3_path, top_k=top_k_genres)
            t3_result = {
                "ran": True,
                "primary_genre": raw_t3.get("primary_genre"),
                "top_genres": raw_t3.get("top_genres", []),
                "inference_time_sec": raw_t3.get("inference_time_sec", 0.0),
            }
        except Exception as exc:
            logger.warning("Tier 3 neural genre classification failed for %s: %s", mp3_path, exc)

    # ── Synthesize Unified Profile ───────────────────────────────────────────
    # Final BPM: Prefer acoustic BPM if valid, fallback to ID3 TBPM
    final_bpm = t2_result.get("bpm") or id3_bpm

    # Combine & normalize genres
    combined_genres: List[str] = list(t1_genres)
    if t3_result.get("primary_genre"):
        p_gen = t3_result["primary_genre"]
        if p_gen and not any(p_gen.lower() == g.lower() for g in combined_genres):
            combined_genres.append(p_gen)

    for tg in t3_result.get("top_genres", []):
        g_name = tg.get("genre")
        if g_name and not any(g_name.lower() == g.lower() for g in combined_genres):
            # Include top 2 neural genres if list is short
            if len(combined_genres) < 4:
                combined_genres.append(g_name)

    # Primary genre: first Tier 1 genre if present, else Tier 3 neural
    primary_genre = t1_genres[0] if t1_genres else t3_result.get("primary_genre")

    has_plain = bool(tier1.get("plain_lyrics"))
    has_timed = bool(tier1.get("timed_lyrics_lrc"))

    elapsed = round(time.monotonic() - total_start, 3)

    return {
        "primary_genre": primary_genre,
        "genres": combined_genres,
        "style_tags": style_tags,
        "style_prompt_raw": raw_prompt,
        "bpm": final_bpm,
        "key": t2_result.get("key"),
        "energy": t2_result.get("energy", "unknown"),
        "lyrics": {
            "has_embedded": has_plain or has_timed,
            "plain_lyrics": tier1.get("plain_lyrics"),
            "timed_lyrics_lrc": tier1.get("timed_lyrics_lrc"),
        },
        "tier_breakdown": {
            "tier1_metadata": {
                "sources": tier1.get("sources", {}),
                "genres": t1_genres,
                "id3_bpm": id3_bpm,
                "notes": tier1.get("notes"),
            },
            "tier2_signal": t2_result,
            "tier3_neural": t3_result,
        },
        "total_time_sec": elapsed,
    }
