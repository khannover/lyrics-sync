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
import time
from difflib import SequenceMatcher
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dtw import dtw
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _job_log(job_id: Optional[str], event: str, **fields) -> None:
    parts = ["[sync]"]
    if job_id:
        parts.append(f"job={job_id}")
    parts.append(event)
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


@dataclass
class AlignmentResult:
    lines: List[Tuple[str, int]]
    quality: str = "good"
    warnings: List[str] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None


MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))
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
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
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
        from huggingface_hub import model_info
        
        # Determine the model ID based on size (Systran/faster-whisper-{size})
        # This matches what faster-whisper uses internally for default models
        repo_id = f"Systran/faster-whisper-{MODEL_SIZE}"

        logger.info("Loading whisper model '%s' ...", MODEL_SIZE)
        
        # 1. Try to load locally first (strictly)
        try:
            _model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                download_root=str(MODEL_DIR),
                local_files_only=True,
            )
            logger.info("Model loaded from local cache.")
            
            # 2. Check for updates (best effort, don't block or fail if offline)
            try:
                # faster-whisper (via huggingface_hub v0.x) caching structure:
                # models--Systran--faster-whisper-{size}/refs/main
                # which contains the current commit hash.
                
                # Sanitize repo_id for path construction (replace / with --)
                cache_dir_name = f"models--{repo_id.replace('/', '--')}"
                ref_path = MODEL_DIR / cache_dir_name / "refs" / "main"
                
                local_sha = None
                if ref_path.exists():
                    local_sha = ref_path.read_text().strip()
                
                info = model_info(repo_id)
                remote_sha = info.sha
                
                if local_sha and remote_sha != local_sha:
                    logger.warning(
                        "Model update available! Local: %s, Remote: %s. "
                        "Delete the '%s' directory (or the specific model cache) to update.",
                        local_sha, remote_sha, str(MODEL_DIR)
                    )
                elif local_sha:
                    logger.info("Local model is up-to-date (SHA: %s)", local_sha)
                else:
                    logger.info("Remote SHA: %s. (Could not determine local SHA)", remote_sha)
                
            except Exception as e:
                logger.warning("Could not check for model updates: %s", e)

        except Exception:
            # Local loading failed (not present?), so we MUST download.
            logger.info("Local model not found. Downloading from HuggingFace...")
            _model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                download_root=str(MODEL_DIR),
                local_files_only=False,
            )
            logger.info("Model downloaded and loaded.")

    return _model


def _convert_to_wav(mp3_path: str, job_dir: str, job_id: Optional[str] = None) -> str:
    """Convert MP3 to 16 kHz mono WAV."""
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Input audio file not found: {mp3_path}")
    wav_path = os.path.join(job_dir, "audio.wav")
    mp3_name = Path(mp3_path).name
    mp3_bytes = os.path.getsize(mp3_path)
    _job_log(
        job_id,
        "stage=convert_start",
        file=mp3_name,
        mp3_bytes=mp3_bytes,
    )
    started = time.monotonic()
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    elapsed = time.monotonic() - started
    duration_ms = _get_audio_duration_ms(wav_path)
    _job_log(
        job_id,
        "stage=convert_done",
        file=mp3_name,
        elapsed=f"{elapsed:.1f}s",
        duration_ms=duration_ms,
    )
    return wav_path


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _is_marker(text: str) -> bool:
    """Check if a line is just a metadata marker like [Chorus] or Verse 1:"""
    clean = text.strip()
    if not clean:
        return False
    # Lines in brackets: [Verse], (Chorus), etc.
    if (clean.startswith("[") and clean.endswith("]")) or (clean.startswith("(") and clean.endswith(")")):
        return True
    # Lines ending with colon and have 1-2 words: "Verse 1:", "Outro:"
    if clean.endswith(":") and len(clean.split()) <= 2:
        return True
    return False


def _transcribe_with_word_timestamps(
    wav_path: str,
    lyrics_text: str = None,
    job_id: Optional[str] = None,
) -> List[dict]:
    """
    Transcribe audio with word-level timestamps using faster-whisper.
    Returns: [{"word": "hello", "start": 0.0, "end": 0.52}, ...]
    """
    model = _get_model()

    # 1. Detect language hint from lyrics text
    # This prevents Whisper from getting stuck in a wrong language (e.g. Khmer)
    # for songs with instrumental intros or poor quality.
    language = None
    if lyrics_text:
        lower_lyrics = lyrics_text.lower()
        # Common word checks for language hinting
        if any(w in lower_lyrics for w in [" the ", " and ", " you ", " are ", " this "]):
            language = "en"
        elif any(w in lower_lyrics for w in [" que ", " el ", " la ", " con ", " los "]):
            language = "es"
        elif any(w in lower_lyrics for w in [" der ", " die ", " das ", " und ", " ist "]):
            language = "de"
        elif any(w in lower_lyrics for w in [" le ", " la ", " et ", " les ", " une "]):
            language = "fr"
        elif any(w in lower_lyrics for w in [" che ", " il ", " la ", " un ", " non "]):
            language = "it"
        elif any(w in lower_lyrics for w in [" que ", " o ", " a ", " e ", " com "]):
            language = "pt"

    # 2. Create initial prompt from first few lines of lyrics
    # This helps Whisper with context and reinforces the language choice.
    initial_prompt = None
    if lyrics_text:
        prompt_lines = [l.strip() for l in lyrics_text.splitlines() if l.strip() and not _is_marker(l)]
        if prompt_lines:
            initial_prompt = " ".join(prompt_lines[:5])

    _job_log(
        job_id,
        "stage=transcribe_start",
        language_hint=language or "auto",
        prompt=(initial_prompt[:50] + "...") if initial_prompt else "none",
    )
    started = time.monotonic()

    segments, info = model.transcribe(
        wav_path,
        beam_size=1,
        word_timestamps=True,
        language=language,  # Use our hint or None for auto-detect
        initial_prompt=initial_prompt,
        vad_filter=False,
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

    elapsed = time.monotonic() - started
    _job_log(
        job_id,
        "stage=transcribe_done",
        words=len(words),
        language=info.language,
        language_prob=f"{info.language_probability:.2f}",
        elapsed=f"{elapsed:.1f}s",
    )

    return words


def _get_audio_duration_ms(audio_path: str) -> int:
    """Return audio duration in milliseconds via ffprobe, or 0 on failure."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return 0


def _spread_timestamps_by_lyrics(lyrics_lines: List[str], duration_ms: int) -> List[Tuple[str, int]]:
    """
    Deterministic full-song fallback: spread lyric lines across audio duration by word density.
    """
    if duration_ms <= 0:
        return [(line, 0) for line in lyrics_lines]

    non_marker_lines = [
        i for i, line in enumerate(lyrics_lines)
        if not _is_marker(line) and _normalize(line)
    ]
    if not non_marker_lines:
        return [(line, 0) for line in lyrics_lines]

    words_per_line = [max(1, len(_normalize(lyrics_lines[i]).split())) for i in non_marker_lines]
    total_words = sum(words_per_line)

    line_ts = {}
    cumulative = 0
    # Leave a small tail so last line does not hard-hit track end.
    usable_ms = int(duration_ms * 0.96)

    for idx, wcount in zip(non_marker_lines, words_per_line):
        ratio = cumulative / max(1, total_words)
        line_ts[idx] = int(ratio * usable_ms)
        cumulative += wcount

    result = []
    for i, line in enumerate(lyrics_lines):
        result.append([line, line_ts.get(i)])

    for i in range(len(result) - 2, -1, -1):
        if result[i][1] is None and _is_marker(result[i][0]) and result[i + 1][1] is not None:
            result[i][1] = result[i + 1][1]

    last_known_ts = 0
    for i in range(len(result)):
        if result[i][1] is None:
            result[i][1] = last_known_ts
        else:
            last_known_ts = result[i][1]

    return [(line, ts) for line, ts in result]


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
        # Skip markers (metadata) during alignment to avoid matching them to random sounds
        if _is_marker(line):
            continue
            
        line_words = _normalize(line).split()
        for w in line_words:
            if w:
                lyrics_word_entries.append((w, line_idx))

    if len(lyrics_word_entries) == 0:
        logger.warning("No lyrics words to align (only markers?) - returning zero timestamps")
        return [(line, 0) for line in lyrics_lines]

    # Filter whisper words to only those that normalize to something non-empty
    whisper_words = [w for w in whisper_words if _normalize(w["word"])]
    whisper_normalized = [_normalize(w["word"]) for w in whisper_words]

    if len(whisper_normalized) == 0:
         logger.warning("Empty whisper word list (after filtering) - returning zero timestamps")
         return [(line, 0) for line in lyrics_lines]

    # Build character vocabulary for vector representation
    all_chars = set()
    for w, _ in lyrics_word_entries:
        all_chars.update(w)
    for w in whisper_normalized:
        all_chars.update(w)
    
    if not all_chars:
        logger.warning("Vocabulary is empty - returning zero timestamps")
        return [(line, 0) for line in lyrics_lines]

    vocab = {ch: i for i, ch in enumerate(sorted(all_chars))}
    vocab_size = len(vocab)
    
    def to_vec(word: str) -> np.ndarray:
        vec = np.zeros(vocab_size, dtype=np.float32)
        for ch in word:
            if ch in vocab:
                vec[vocab[ch]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    lyrics_vecs = np.array([to_vec(w) for w, _ in lyrics_word_entries], dtype=np.float32)
    whisper_vecs = np.array([to_vec(w) for w in whisper_normalized], dtype=np.float32)

    if lyrics_vecs.ndim == 1: lyrics_vecs = lyrics_vecs.reshape(1, -1)
    if whisper_vecs.ndim == 1: whisper_vecs = whisper_vecs.reshape(1, -1)

    logger.info("Aligning %d lyric words vs %d whisper words (vocab size: %d, shapes: %s, %s)", 
                len(lyrics_vecs), len(whisper_vecs), vocab_size, lyrics_vecs.shape, whisper_vecs.shape)

    try:
        dist_matrix = cdist(lyrics_vecs, whisper_vecs, metric="euclidean")
        alignment = dtw(dist_matrix, step_pattern="symmetric2")
    except ValueError as e:
        logger.error("DTW failed: %s", e)
        return [(line, 0) for line in lyrics_lines]

    # DTW can map one lyric word to multiple whisper words (and vice versa).
    # Use the median whisper index for each lyric word to avoid "first-hit" collapse.
    lyric_word_to_whisper_candidates = {}
    for l_idx, w_idx in zip(alignment.index1, alignment.index2):
        lyric_word_to_whisper_candidates.setdefault(l_idx, []).append(w_idx)

    lyric_word_to_whisper = {}
    for l_idx, candidates in lyric_word_to_whisper_candidates.items():
        lyric_word_to_whisper[l_idx] = int(np.median(candidates))

    # Build per-line candidate whisper indices from all words in that line,
    # then pick the earliest aligned index as the line start.
    line_whisper_idx_candidates = {}
    for lw_idx, (_, line_idx) in enumerate(lyrics_word_entries):
        if lw_idx in lyric_word_to_whisper:
            line_whisper_idx_candidates.setdefault(line_idx, []).append(lyric_word_to_whisper[lw_idx])

    line_timestamps = {}
    for line_idx, idx_candidates in line_whisper_idx_candidates.items():
        if idx_candidates:
            w_idx = min(idx_candidates)
            line_timestamps[line_idx] = int(whisper_words[w_idx]["start"] * 1000)

    # Fallback when alignment collapses into one (or almost one) timestamp.
    non_marker_lines = [
        i for i, line in enumerate(lyrics_lines)
        if not _is_marker(line) and _normalize(line)
    ]
    non_marker_ts = [line_timestamps[i] for i in non_marker_lines if i in line_timestamps]
    if len(non_marker_lines) > 1 and len(set(non_marker_ts)) <= 1:
        logger.warning(
            "Low timestamp diversity detected (%d unique for %d lines). Applying proportional fallback.",
            len(set(non_marker_ts)),
            len(non_marker_lines),
        )

        lyric_words_per_line = [
            len(_normalize(lyrics_lines[i]).split()) for i in non_marker_lines
        ]
        total_lyric_words = sum(lyric_words_per_line)
        max_whisper_idx = len(whisper_words) - 1

        if total_lyric_words > 0 and max_whisper_idx >= 0:
            cumulative = 0
            for i, word_count in zip(non_marker_lines, lyric_words_per_line):
                ratio = cumulative / total_lyric_words
                w_idx = min(max_whisper_idx, int(round(ratio * max_whisper_idx)))
                line_timestamps[i] = int(whisper_words[w_idx]["start"] * 1000)
                cumulative += max(word_count, 1)

    # Build final result, inheriting timestamps intelligently
    result = []
    # 1. Fill in known timestamps
    for i, line in enumerate(lyrics_lines):
        ts = line_timestamps.get(i)
        result.append([line, ts]) # List for mutability
    
    # 2. Backward fill for markers at the start of a section
    # If a marker exists just before a line with a timestamp, it should take that timestamp
    for i in range(len(result) - 2, -1, -1):
        if result[i][1] is None and _is_marker(result[i][0]):
            if result[i+1][1] is not None:
                result[i][1] = result[i+1][1]

    # 3. Forward fill for the rest (default to 0 or previous)
    last_known_ts = 0
    for i in range(len(result)):
        if result[i][1] is None:
            result[i][1] = last_known_ts
        else:
            last_known_ts = result[i][1]

    return [(line, ts) for line, ts in result]


def _align_lines_progressive(
    lyrics_lines: List[str],
    whisper_words: List[dict],
) -> List[Tuple[str, int]]:
    """
    Progressive forward alignment fallback.

    For each non-marker lyric line, find the best matching forward window in
    Whisper words using sequence similarity. This avoids DTW collapse cases
    where many lines get mapped to just a few early timestamps.
    """
    if not whisper_words:
        return [(line, 0) for line in lyrics_lines]

    # Keep normalized tokens and timestamp-bearing words in sync.
    filtered_whisper_words = [w for w in whisper_words if _normalize(w["word"])]
    whisper_norm = [_normalize(w["word"]) for w in filtered_whisper_words]
    if not whisper_norm:
        return [(line, 0) for line in lyrics_lines]

    def _window_score(line_words: List[str], cand_words: List[str]) -> float:
        if not line_words or not cand_words:
            return 0.0

        target = " ".join(line_words)
        cand = " ".join(cand_words)
        seq = SequenceMatcher(None, target, cand).ratio()

        line_set = set(line_words)
        cand_set = set(cand_words)
        overlap = len(line_set.intersection(cand_set)) / max(1, len(line_set))

        # Prefer windows that start near the lyric line's first word.
        first_word_bonus = 0.0
        if cand_words and line_words:
            if cand_words[0] == line_words[0]:
                first_word_bonus += 0.08
            elif line_words[0] in cand_words[:2]:
                first_word_bonus += 0.04

        return (0.62 * seq) + (0.38 * overlap) + first_word_bonus

    line_ts = {}
    cursor = 0
    max_w = len(whisper_norm)

    for line_idx, line in enumerate(lyrics_lines):
        if _is_marker(line):
            continue

        line_words = _normalize(line).split()
        if not line_words:
            continue

        lw_len = len(line_words)

        # Search forward in a bounded window to keep alignment monotonic.
        start_lo = min(cursor, max_w - 1)
        start_hi = min(max_w - 1, start_lo + 240)

        best_start = start_lo
        best_score = -1.0

        for s in range(start_lo, start_hi + 1):
            # Compare with multiple candidate phrase lengths around lyric length.
            for span in range(max(1, lw_len - 3), lw_len + 6):
                e = min(max_w, s + span)
                if e <= s:
                    continue
                score = _window_score(line_words, whisper_norm[s:e])
                # Mildly prefer earlier positions on ties to reduce jumps.
                score_adj = score - ((s - start_lo) * 0.0004)
                if score_adj > best_score:
                    best_score = score_adj
                    best_start = s

        line_ts[line_idx] = int(filtered_whisper_words[best_start]["start"] * 1000)
        cursor = min(max_w - 1, best_start + max(1, lw_len - 1))

    # Build result and keep markers aligned with neighboring lyrics.
    result = []
    for i, line in enumerate(lyrics_lines):
        result.append([line, line_ts.get(i)])

    for i in range(len(result) - 2, -1, -1):
        if result[i][1] is None and _is_marker(result[i][0]) and result[i + 1][1] is not None:
            result[i][1] = result[i + 1][1]

    last_known_ts = 0
    for i in range(len(result)):
        if result[i][1] is None:
            result[i][1] = last_known_ts
        else:
            last_known_ts = result[i][1]

    # Stabilize non-marker line order: avoid repeated identical timestamps.
    prev_non_marker_ts = -1
    for i, (line, ts) in enumerate(result):
        if _is_marker(line):
            continue
        if ts <= prev_non_marker_ts:
            ts = prev_non_marker_ts + 120
            result[i][1] = ts
        prev_non_marker_ts = ts

    return [(line, ts) for line, ts in result]


def align_lyrics_to_audio(
    mp3_path: str,
    lyrics_text: str,
    job_dir: str,
    job_id: Optional[str] = None,
) -> AlignmentResult:
    """
    Main entry point for alignment.
    Returns aligned lines plus sync quality metadata.
    """
    quality = "good"
    warnings: List[str] = []
    job_started = time.monotonic()
    mp3_name = Path(mp3_path).name
    _job_log(
        job_id,
        "stage=alignment_start",
        file=mp3_name,
        model=MODEL_SIZE,
    )

    wav_path = _convert_to_wav(mp3_path, job_dir, job_id=job_id)

    # Parse lyrics into non-empty lines
    lyrics_lines = [l.strip() for l in lyrics_text.splitlines() if l.strip()]

    if not lyrics_lines:
        raise ValueError("No lyrics lines found")

    _job_log(job_id, "stage=lyrics_parsed", lines=len(lyrics_lines))

    # Get word-level timestamps from Whisper
    whisper_words = _transcribe_with_word_timestamps(
        wav_path, lyrics_text=lyrics_text, job_id=job_id
    )

    # Align user lyrics to whisper timestamps
    _job_log(
        job_id,
        "stage=dtw_start",
        lyric_words=sum(len(_normalize(line).split()) for line in lyrics_lines),
        whisper_words=len(whisper_words),
    )
    align_started = time.monotonic()
    synced = _align_lines_to_words(lyrics_lines, whisper_words)
    _job_log(
        job_id,
        "stage=dtw_done",
        elapsed=f"{time.monotonic() - align_started:.1f}s",
        lines=len(synced),
    )

    # If DTW output collapsed to very few timestamps, retry with progressive matcher.
    non_marker_ts = [
        ts for line, ts in synced
        if not _is_marker(line) and _normalize(line)
    ]
    unique_ratio = (len(set(non_marker_ts)) / max(1, len(non_marker_ts))) if non_marker_ts else 0
    if len(non_marker_ts) > 6 and unique_ratio < 0.35:
        logger.warning(
            "Low DTW timestamp diversity (%d unique / %d lines). Falling back to progressive matching.",
            len(set(non_marker_ts)),
            len(non_marker_ts),
        )
        _job_log(job_id, "stage=progressive_fallback")
        quality = "degraded"
        warnings.append("Progressive alignment fallback used after low DTW diversity.")
        synced = _align_lines_progressive(lyrics_lines, whisper_words)

    # Final guardrail: if timestamps still cover only a tiny early segment,
    # spread lines across the full track duration.
    final_non_marker_ts = [
        ts for line, ts in synced
        if not _is_marker(line) and _normalize(line)
    ]
    final_unique_ratio = (
        len(set(final_non_marker_ts)) / max(1, len(final_non_marker_ts))
        if final_non_marker_ts else 0.0
    )
    max_ts = max(final_non_marker_ts) if final_non_marker_ts else 0
    duration_ms = _get_audio_duration_ms(wav_path)

    if duration_ms > 0 and len(final_non_marker_ts) > 6 and (
        final_unique_ratio < 0.45 or max_ts < int(duration_ms * 0.25)
    ):
        logger.warning(
            "Timestamp coverage still low (unique_ratio=%.2f, max_ts=%dms, duration=%dms). "
            "Applying full-song proportional fallback.",
            final_unique_ratio,
            max_ts,
            duration_ms,
        )
        _job_log(
            job_id,
            "stage=proportional_fallback",
            duration_ms=duration_ms,
            unique_ratio=f"{final_unique_ratio:.2f}",
        )
        quality = "fallback"
        warnings.append("Approximate spread timing applied after low timestamp coverage.")
        synced = _spread_timestamps_by_lyrics(lyrics_lines, duration_ms)

    if not whisper_words:
        quality = "fallback"
        warnings.append("No words detected in audio transcription.")

    elapsed = time.monotonic() - job_started
    _job_log(
        job_id,
        "stage=alignment_done",
        file=mp3_name,
        lines=len(synced),
        elapsed=f"{elapsed:.1f}s",
    )
    for line, ts in synced[:3]:
        logger.info("[sync] job=%s preview [%dms] %s", job_id or "-", ts, line[:60])
    if len(synced) > 3:
        logger.info("[sync] job=%s preview ... and %d more lines", job_id or "-", len(synced) - 3)

    report = {
        "quality": quality,
        "warnings": warnings,
        "line_count": len(synced),
        "duration_ms": duration_ms if "duration_ms" in locals() else 0,
        "whisper_word_count": len(whisper_words),
    }
    return AlignmentResult(
        lines=synced,
        quality=quality,
        warnings=warnings,
        report=report,
    )
def transcribe_audio_to_text(mp3_path: str, job_dir: str, job_id: Optional[str] = None) -> str:
    """
    Transcribe audio to plain text lyrics using faster-whisper.
    Used as a fallback when no embedded lyrics are present.
    """
    wav_path = _convert_to_wav(mp3_path, job_dir, job_id)
    model = _get_model()

    _job_log(
        job_id,
        "stage=transcribe_text_start",
        vad_filter=True,
    )
    started = time.monotonic()

    # We use beam_size=5 for better text quality, and VAD filter to skip silence.
    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,
    )

    text_lines = []
    for segment in segments:
        line = segment.text.strip()
        if line:
            text_lines.append(line)

    elapsed = time.monotonic() - started
    _job_log(
        job_id,
        "stage=transcribe_text_done",
        language=info.language,
        language_prob=f"{info.language_probability:.2f}",
        lines=len(text_lines),
        elapsed=f"{elapsed:.1f}s",
    )

    return "\n".join(text_lines)
