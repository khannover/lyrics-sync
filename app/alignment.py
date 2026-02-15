"""
Forced alignment using faster-whisper (CTranslate2) + DTW.
No torch, no openvino python package, no pkg_resources issues.

faster-whisper uses CTranslate2 which is a C++ inference engine.
It natively provides word-level timestamps via Whisper.
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dtw import dtw

logger = logging.getLogger(__name__)

MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
MODEL_DIR = Path("/app/models")

_model = None


def ensure_model():
    """Pre-download the Whisper model for faster-whisper / CTranslate2."""
    from faster_whisper import WhisperModel

    logger.info("Downloading whisper model '%s' ...", MODEL_SIZE)
    # This downloads and caches the CTranslate2-converted model
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        download_root=str(MODEL_DIR),
    )
    # Quick test
    logger.info("Model '%s' ready.", MODEL_SIZE)
    return model


def _get_model():
    """Get or create the cached Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info("Loading whisper model '%s' ...", MODEL_SIZE)
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=str(MODEL_DIR),
        )
        logger.info("Model loaded.")
    return _model


def _convert_to_wav(mp3_path: str, job_dir: str) -> str:
    """Convert MP3 to 16 kHz mono WAV."""
    wav_path = os.path.join(job_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return wav_path


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _transcribe_with_word_timestamps(wav_path: str) -> List[dict]:
    """
    Transcribe audio with word-level timestamps using faster-whisper.
    Returns: [{"word": "hello", "start": 0.0, "end": 0.52}, ...]
    """
    model = _get_model()

    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        word_timestamps=True,
        language=None,  # auto-detect
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    logger.info(
        "Transcribed %d words (detected language: %s, prob: %.2f)",
        len(words),
        info.language,
        info.language_probability,
    )
    return words


def _align_lines_to_words(
    lyrics_lines: List[str],
    whisper_words: List[dict],
) -> List[Tuple[str, int]]:
    """
    Align user-provided lyrics lines to Whisper word timestamps via DTW.
    Returns: [(line_text, start_time_ms), ...]
    """
    if not whisper_words:
        logger.warning("No words from transcription — returning zero timestamps")
        return [(line, 0) for line in lyrics_lines]

    # Split lyrics into individual words, tracking which line each belongs to
    lyrics_word_entries = []  # (normalized_word, line_index)
    for line_idx, line in enumerate(lyrics_lines):
        line_words = _normalize(line).split()
        for w in line_words:
            if w:
                lyrics_word_entries.append((w, line_idx))

    if not lyrics_word_entries:
        return [(line, 0) for line in lyrics_lines]

    whisper_normalized = [_normalize(w["word"]) for w in whisper_words]

    # Build character vocabulary for vector representation
    all_chars = set()
    for w, _ in lyrics_word_entries:
        all_chars.update(w)
    for w in whisper_normalized:
        all_chars.update(w)
    vocab = {ch: i for i, ch in enumerate(sorted(all_chars))}
    vocab_size = len(vocab)

    def to_vec(word: str) -> np.ndarray:
        vec = np.zeros(vocab_size, dtype=np.float32)
        for ch in word:
            if ch in vocab:
                vec[vocab[ch]] += 1.0
        # Normalize to unit length to reduce bias from word length
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    lyrics_vecs = np.array([to_vec(w) for w, _ in lyrics_word_entries])
    whisper_vecs = np.array([to_vec(w) for w in whisper_normalized])

    # DTW alignment
    alignment = dtw(lyrics_vecs, whisper_vecs, dist_method="euclidean")

    # Map each lyrics word to its best Whisper word match
    lyrics_to_whisper = {}
    for l_idx, w_idx in zip(alignment.index1, alignment.index2):
        if l_idx not in lyrics_to_whisper:
            lyrics_to_whisper[l_idx] = w_idx

    # For each lyrics line, find the timestamp of its first word
    line_timestamps = {}
    for lw_idx, (_, line_idx) in enumerate(lyrics_word_entries):
        if line_idx not in line_timestamps and lw_idx in lyrics_to_whisper:
            w_idx = lyrics_to_whisper[lw_idx]
            start_ms = int(whisper_words[w_idx]["start"] * 1000)
            line_timestamps[line_idx] = start_ms

    # Build final result, carrying forward last known timestamp for unmatched lines
    result = []
    last_ts = 0
    for i, line in enumerate(lyrics_lines):
        ts = line_timestamps.get(i, last_ts)
        result.append((line, ts))
        last_ts = ts

    return result


def align_lyrics_to_audio(
    mp3_path: str,
    lyrics_text: str,
    job_dir: str,
) -> List[Tuple[str, int]]:
    """
    Main entry point for alignment.
    Returns: [(line_text, start_time_ms), ...]
    """
    wav_path = _convert_to_wav(mp3_path, job_dir)

    # Parse lyrics into non-empty lines
    lyrics_lines = [l.strip() for l in lyrics_text.splitlines() if l.strip()]

    if not lyrics_lines:
        raise ValueError("No lyrics lines found")

    # Get word-level timestamps from Whisper
    whisper_words = _transcribe_with_word_timestamps(wav_path)

    # Align user lyrics to whisper timestamps
    synced = _align_lines_to_words(lyrics_lines, whisper_words)

    logger.info("Alignment complete: %d lines", len(synced))
    for line, ts in synced[:5]:
        logger.info("  [%7d ms] %s", ts, line[:60])
    if len(synced) > 5:
        logger.info("  ... and %d more lines", len(synced) - 5)

    return synced