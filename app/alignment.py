"""
Forced alignment using faster-whisper (CTranslate2) + DTW.
No torch, no openvino python package, no pkg_resources issues.

faster-whisper uses CTranslate2 which is a C++ inference engine.
It natively provides word-level timestamps via Whisper.
"""

import logging
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
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
class WordTiming:
    line_index: int
    word_index: int
    word: str
    start_ms: int
    end_ms: int
    matched: bool
    confidence: Optional[float] = None


@dataclass
class AlignmentResult:
    lines: List[Tuple[str, int]]
    quality: str = "good"
    warnings: List[str] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    word_timings: List[WordTiming] = field(default_factory=list)
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    transcription_pass: str = "fast"


@dataclass
class StructuredAlignment:
    lines: List[Tuple[str, int]]
    word_timings: List[WordTiming]


@dataclass
class TranscriptionPassResult:
    words: List[dict]
    detected_language: Optional[str]
    language_probability: Optional[float]
    pass_name: str


MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))
MODEL_DIR = Path("/app/models")

_model = None

_JOINER_CHARS = {"'", "’"}
_SCRIPT_HINTS = {
    "hiragana": "ja",
    "katakana": "ja",
    "cjk": "zh",
    "hangul": "ko",
    "thai": "th",
    "khmer": "km",
    "arabic": "ar",
    "cyrillic": "ru",
    "devanagari": "hi",
}
_QUALITY_RANK = {"good": 2, "degraded": 1, "fallback": 0}


def ensure_model():
    """Pre-download the Whisper model for faster-whisper / CTranslate2."""
    from faster_whisper import WhisperModel

    logger.info("Downloading whisper model '%s' ...", MODEL_SIZE)
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        download_root=str(MODEL_DIR),
    )
    logger.info("Model '%s' ready.", MODEL_SIZE)
    return model


def _get_model():
    """Get or create the cached Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        from huggingface_hub import model_info

        repo_id = f"Systran/faster-whisper-{MODEL_SIZE}"
        logger.info("Loading whisper model '%s' ...", MODEL_SIZE)

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
            try:
                cache_dir_name = f"models--{repo_id.replace('/', '--')}"
                ref_path = MODEL_DIR / cache_dir_name / "refs" / "main"
                local_sha = ref_path.read_text().strip() if ref_path.exists() else None
                info = model_info(repo_id)
                remote_sha = info.sha
                if local_sha and remote_sha != local_sha:
                    logger.warning(
                        "Model update available! Local: %s, Remote: %s. Delete '%s' to update.",
                        local_sha,
                        remote_sha,
                        str(MODEL_DIR),
                    )
                elif local_sha:
                    logger.info("Local model is up-to-date (SHA: %s)", local_sha)
                else:
                    logger.info("Remote SHA: %s. (Could not determine local SHA)", remote_sha)
            except Exception as exc:
                logger.warning("Could not check for model updates: %s", exc)
        except Exception:
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
    _job_log(job_id, "stage=convert_start", file=mp3_name, mp3_bytes=mp3_bytes)
    started = time.monotonic()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp3_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
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


def _create_vocal_focus_wav(audio_path: str, job_dir: str, job_id: Optional[str] = None) -> Optional[str]:
    """Create a vocal-emphasized wav for hard retry cases using ffmpeg only."""
    focused_path = os.path.join(job_dir, "audio_vocal_focus.wav")
    _job_log(job_id, "stage=vocal_focus_start", source=Path(audio_path).name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-af",
        "highpass=f=120,lowpass=f=4500,acompressor=threshold=-18dB:ratio=2.5:attack=5:release=80,"
        "alimiter=limit=0.95",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        focused_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.exists(focused_path):
        logger.warning("Vocal focus preprocessing failed: %s", result.stderr[:300])
        _job_log(job_id, "stage=vocal_focus_failed")
        return None
    _job_log(job_id, "stage=vocal_focus_done", file=Path(focused_path).name)
    return focused_path


def _script_family(ch: str) -> Optional[str]:
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "cjk"
    if 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x1780 <= cp <= 0x17FF:
        return "khmer"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if "LATIN" in unicodedata.name(ch, ""):
        return "latin"
    return None


def _normalize_token(token: str) -> str:
    if not token:
        return ""
    normalized = unicodedata.normalize("NFKD", token.casefold())
    chars = []
    for ch in normalized:
        if unicodedata.category(ch).startswith("M"):
            continue
        if ch.isalnum() or _script_family(ch) in _SCRIPT_HINTS or _script_family(ch) == "latin":
            chars.append(ch)
    return "".join(chars).strip()


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            token = "".join(buffer).strip()
            if token:
                tokens.append(token)
            buffer.clear()

    for ch in text:
        family = _script_family(ch)
        category = unicodedata.category(ch)
        if ch.isspace():
            flush()
            continue
        if family in {"hiragana", "katakana", "cjk", "thai", "khmer"}:
            flush()
            tokens.append(ch)
            continue
        if category.startswith(("L", "N")) or family in {"arabic", "cyrillic", "devanagari", "latin"}:
            buffer.append(ch)
            continue
        if ch in _JOINER_CHARS and buffer:
            buffer.append(ch)
            continue
        flush()

    flush()
    return tokens


def _normalize(text: str) -> str:
    return " ".join(token for token in (_normalize_token(t) for t in _tokenize(text)) if token)


def _lyric_word_entries(lyrics_lines: List[str]) -> List[dict]:
    entries = []
    for line_idx, line in enumerate(lyrics_lines):
        if _is_marker(line):
            continue
        word_idx = 0
        for raw in _tokenize(line):
            normalized = _normalize_token(raw)
            if not normalized:
                continue
            entries.append(
                {
                    "normalized": normalized,
                    "word": raw.strip(),
                    "line_index": line_idx,
                    "word_index": word_idx,
                }
            )
            word_idx += 1
    return entries


def _whisper_entries(whisper_words: List[dict]) -> List[dict]:
    entries = []
    for item in whisper_words:
        normalized = _normalize_token(str(item.get("word") or ""))
        if not normalized:
            continue
        entries.append(
            {
                "word": str(item.get("word") or "").strip(),
                "normalized": normalized,
                "start": float(item.get("start") or 0.0),
                "end": float(item.get("end") or item.get("start") or 0.0),
            }
        )
    return entries


def _detect_language_hint(lyrics_text: Optional[str]) -> Optional[str]:
    if not lyrics_text:
        return None

    script_counts: Dict[str, int] = {}
    for ch in lyrics_text:
        if ch.isspace():
            continue
        family = _script_family(ch)
        if family:
            script_counts[family] = script_counts.get(family, 0) + 1

    significant = {fam for fam, count in script_counts.items() if count >= 3}
    non_latin_significant = [fam for fam in significant if fam != "latin"]
    if len(non_latin_significant) > 1:
        return None
    if len(non_latin_significant) == 1:
        return _SCRIPT_HINTS.get(non_latin_significant[0])

    lower = f" {lyrics_text.casefold()} "
    candidates = []
    if any(w in lower for w in [" the ", " and ", " you ", " are ", " this "]):
        candidates.append("en")
    if any(w in lower for w in [" que ", " el ", " la ", " con ", " los ", " para "]):
        candidates.append("es")
    if any(w in lower for w in [" der ", " die ", " das ", " und ", " ist ", " ich "]):
        candidates.append("de")
    if any(w in lower for w in [" le ", " et ", " les ", " une ", " dans "]):
        candidates.append("fr")
    if any(w in lower for w in [" che ", " il ", " un ", " non ", " per "]):
        candidates.append("it")
    if any(w in lower for w in [" não ", " você ", " vocês ", " coração ", " também "]):
        candidates.append("pt")

    candidates = list(dict.fromkeys(candidates))
    return candidates[0] if len(candidates) == 1 else None


def _build_initial_prompt(lyrics_text: Optional[str]) -> Optional[str]:
    if not lyrics_text:
        return None
    prompt_lines = [l.strip() for l in lyrics_text.splitlines() if l.strip() and not _is_marker(l)]
    if not prompt_lines:
        return None
    return " ".join(prompt_lines[:5])[:220]


def _is_marker(text: str) -> bool:
    """Check if a line is just a metadata marker like [Chorus] or Verse 1:"""
    clean = text.strip()
    if not clean:
        return False
    if (clean.startswith("[") and clean.endswith("]")) or (clean.startswith("(") and clean.endswith(")")):
        return True
    if clean.endswith(":") and len(clean.split()) <= 2:
        return True
    return False


def _transcribe_with_word_timestamps(
    wav_path: str,
    lyrics_text: str = None,
    job_id: Optional[str] = None,
    *,
    beam_size: int = 1,
    vad_filter: bool = False,
    pass_name: str = "fast",
) -> TranscriptionPassResult:
    """
    Transcribe audio with word-level timestamps using faster-whisper.
    Returns: [{"word": "hello", "start": 0.0, "end": 0.52}, ...]
    """
    model = _get_model()
    language = _detect_language_hint(lyrics_text)
    initial_prompt = _build_initial_prompt(lyrics_text)

    _job_log(
        job_id,
        "stage=transcribe_start",
        transcription_pass=pass_name,
        language_hint=language or "auto",
        beam_size=beam_size,
        vad_filter=vad_filter,
        prompt=(initial_prompt[:50] + "...") if initial_prompt else "none",
    )
    started = time.monotonic()

    segments, info = model.transcribe(
        wav_path,
        beam_size=beam_size,
        word_timestamps=True,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
    )

    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                    }
                )

    elapsed = time.monotonic() - started
    _job_log(
        job_id,
        "stage=transcribe_done",
        transcription_pass=pass_name,
        words=len(words),
        language=info.language,
        language_prob=f"{info.language_probability:.2f}",
        elapsed=f"{elapsed:.1f}s",
    )
    return TranscriptionPassResult(
        words=words,
        detected_language=info.language,
        language_probability=float(info.language_probability),
        pass_name=pass_name,
    )


def _get_audio_duration_ms(audio_path: str) -> int:
    """Return audio duration in milliseconds via ffprobe, or 0 on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            return 0
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return 0


def _summarize_line_metrics(lines: List[Tuple[str, int]], duration_ms: int) -> Tuple[float, float, int]:
    non_marker_ts = [ts for line, ts in lines if not _is_marker(line) and _normalize(line)]
    unique_ratio = (len(set(non_marker_ts)) / max(1, len(non_marker_ts))) if non_marker_ts else 0.0
    max_ts = max(non_marker_ts) if non_marker_ts else 0
    coverage_ratio = (max_ts / duration_ms) if duration_ms > 0 and non_marker_ts else 0.0
    return unique_ratio, coverage_ratio, max_ts


def _empty_word_timings(lyrics_lines: List[str], matched: bool = False) -> List[WordTiming]:
    timings: List[WordTiming] = []
    for entry in _lyric_word_entries(lyrics_lines):
        timings.append(
            WordTiming(
                line_index=entry["line_index"],
                word_index=entry["word_index"],
                word=entry["word"],
                start_ms=0,
                end_ms=0,
                matched=matched,
                confidence=0.0 if matched else None,
            )
        )
    return timings


def _spread_timestamps_by_lyrics(lyrics_lines: List[str], duration_ms: int) -> List[Tuple[str, int]]:
    """
    Deterministic full-song fallback: spread lyric lines across audio duration by word density.
    """
    if duration_ms <= 0:
        return [(line, 0) for line in lyrics_lines]

    non_marker_lines = [i for i, line in enumerate(lyrics_lines) if not _is_marker(line) and _normalize(line)]
    if not non_marker_lines:
        return [(line, 0) for line in lyrics_lines]

    words_per_line = [max(1, len(_normalize(lyrics_lines[i]).split())) for i in non_marker_lines]
    total_words = sum(words_per_line)

    line_ts = {}
    cumulative = 0
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


def _build_proportional_word_timings(
    lyrics_lines: List[str],
    line_timings: List[Tuple[str, int]],
    duration_ms: int,
) -> List[WordTiming]:
    line_ts = {idx: ts for idx, (_line, ts) in enumerate(line_timings)}
    timings: List[WordTiming] = []
    entries = _lyric_word_entries(lyrics_lines)

    next_line_ts: Dict[int, int] = {}
    known = [(idx, ts) for idx, ts in line_ts.items() if ts is not None]
    for pos, (idx, ts) in enumerate(known):
        if pos + 1 < len(known):
            next_line_ts[idx] = known[pos + 1][1]
        else:
            next_line_ts[idx] = duration_ms if duration_ms > ts else ts + 1200

    grouped: Dict[int, List[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["line_index"], []).append(entry)

    for line_index, group in grouped.items():
        start = int(line_ts.get(line_index, 0) or 0)
        line_end = int(next_line_ts.get(line_index, start + 1200))
        if line_end <= start:
            line_end = start + max(240, len(group) * 180)
        step = max(90, (line_end - start) // max(1, len(group)))
        for idx, entry in enumerate(group):
            word_start = start + idx * step
            word_end = min(line_end, word_start + step)
            timings.append(
                WordTiming(
                    line_index=line_index,
                    word_index=entry["word_index"],
                    word=entry["word"],
                    start_ms=word_start,
                    end_ms=max(word_start + 60, word_end),
                    matched=False,
                    confidence=0.1,
                )
            )
    return timings


def _align_lines_to_words(lyrics_lines: List[str], whisper_words: List[dict]) -> StructuredAlignment:
    """
    Align user-provided lyrics lines to Whisper word timestamps via DTW.
    Returns structured line + word timing data.
    """
    if not whisper_words:
        logger.warning("No words from transcription — returning zero timestamps")
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

    lyrics_word_entries = _lyric_word_entries(lyrics_lines)
    if not lyrics_word_entries:
        logger.warning("No lyrics words to align (only markers?) - returning zero timestamps")
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=[],
        )

    whisper_entries = _whisper_entries(whisper_words)
    if not whisper_entries:
        logger.warning("Empty whisper word list (after filtering) - returning zero timestamps")
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

    all_chars = set()
    for entry in lyrics_word_entries:
        all_chars.update(entry["normalized"])
    for entry in whisper_entries:
        all_chars.update(entry["normalized"])
    if not all_chars:
        logger.warning("Vocabulary is empty - returning zero timestamps")
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

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

    lyrics_vecs = np.array([to_vec(entry["normalized"]) for entry in lyrics_word_entries], dtype=np.float32)
    whisper_vecs = np.array([to_vec(entry["normalized"]) for entry in whisper_entries], dtype=np.float32)
    if lyrics_vecs.ndim == 1:
        lyrics_vecs = lyrics_vecs.reshape(1, -1)
    if whisper_vecs.ndim == 1:
        whisper_vecs = whisper_vecs.reshape(1, -1)

    logger.info(
        "Aligning %d lyric words vs %d whisper words (vocab size: %d, shapes: %s, %s)",
        len(lyrics_vecs),
        len(whisper_vecs),
        vocab_size,
        lyrics_vecs.shape,
        whisper_vecs.shape,
    )

    try:
        dist_matrix = cdist(lyrics_vecs, whisper_vecs, metric="euclidean")
        alignment = dtw(dist_matrix, step_pattern="symmetric2")
    except ValueError as exc:
        logger.error("DTW failed: %s", exc)
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

    lyric_word_to_whisper_candidates = {}
    for l_idx, w_idx in zip(alignment.index1, alignment.index2):
        lyric_word_to_whisper_candidates.setdefault(l_idx, []).append(w_idx)

    lyric_word_to_whisper = {}
    for l_idx, candidates in lyric_word_to_whisper_candidates.items():
        lyric_word_to_whisper[l_idx] = int(np.median(candidates))

    line_whisper_idx_candidates = {}
    word_timings: List[WordTiming] = []
    for lw_idx, entry in enumerate(lyrics_word_entries):
        mapped_idx = lyric_word_to_whisper.get(lw_idx)
        if mapped_idx is None:
            word_timings.append(
                WordTiming(
                    line_index=entry["line_index"],
                    word_index=entry["word_index"],
                    word=entry["word"],
                    start_ms=0,
                    end_ms=0,
                    matched=False,
                    confidence=None,
                )
            )
            continue

        line_whisper_idx_candidates.setdefault(entry["line_index"], []).append(mapped_idx)
        whisper_entry = whisper_entries[mapped_idx]
        dist = float(dist_matrix[lw_idx][mapped_idx])
        confidence = round(max(0.0, 1.0 - (min(2.0, dist) / 2.0)), 3)
        word_timings.append(
            WordTiming(
                line_index=entry["line_index"],
                word_index=entry["word_index"],
                word=entry["word"],
                start_ms=int(whisper_entry["start"] * 1000),
                end_ms=int(whisper_entry["end"] * 1000),
                matched=True,
                confidence=confidence,
            )
        )

    line_timestamps = {}
    for line_idx, idx_candidates in line_whisper_idx_candidates.items():
        if idx_candidates:
            w_idx = min(idx_candidates)
            line_timestamps[line_idx] = int(whisper_entries[w_idx]["start"] * 1000)

    non_marker_lines = [i for i, line in enumerate(lyrics_lines) if not _is_marker(line) and _normalize(line)]
    non_marker_ts = [line_timestamps[i] for i in non_marker_lines if i in line_timestamps]
    if len(non_marker_lines) > 1 and len(set(non_marker_ts)) <= 1:
        logger.warning(
            "Low timestamp diversity detected (%d unique for %d lines). Applying proportional fallback.",
            len(set(non_marker_ts)),
            len(non_marker_lines),
        )
        lyric_words_per_line = [len(_normalize(lyrics_lines[i]).split()) for i in non_marker_lines]
        total_lyric_words = sum(lyric_words_per_line)
        max_whisper_idx = len(whisper_entries) - 1

        if total_lyric_words > 0 and max_whisper_idx >= 0:
            cumulative = 0
            for i, word_count in zip(non_marker_lines, lyric_words_per_line):
                ratio = cumulative / total_lyric_words
                w_idx = min(max_whisper_idx, int(round(ratio * max_whisper_idx)))
                line_timestamps[i] = int(whisper_entries[w_idx]["start"] * 1000)
                cumulative += max(word_count, 1)

    result = []
    for i, line in enumerate(lyrics_lines):
        result.append([line, line_timestamps.get(i)])

    for i in range(len(result) - 2, -1, -1):
        if result[i][1] is None and _is_marker(result[i][0]) and result[i + 1][1] is not None:
            result[i][1] = result[i + 1][1]

    last_known_ts = 0
    for i in range(len(result)):
        if result[i][1] is None:
            result[i][1] = last_known_ts
        else:
            last_known_ts = result[i][1]

    return StructuredAlignment(lines=[(line, ts) for line, ts in result], word_timings=word_timings)


def _align_lines_progressive(lyrics_lines: List[str], whisper_words: List[dict]) -> StructuredAlignment:
    """
    Progressive forward alignment fallback.

    For each non-marker lyric line, find the best matching forward window in
    Whisper words using sequence similarity. This avoids DTW collapse cases.
    """
    if not whisper_words:
        return StructuredAlignment(lines=[(line, 0) for line in lyrics_lines], word_timings=_empty_word_timings(lyrics_lines))

    whisper_entries = _whisper_entries(whisper_words)
    whisper_norm = [entry["normalized"] for entry in whisper_entries]
    if not whisper_norm:
        return StructuredAlignment(lines=[(line, 0) for line in lyrics_lines], word_timings=_empty_word_timings(lyrics_lines))

    def _window_score(line_words: List[str], cand_words: List[str]) -> float:
        if not line_words or not cand_words:
            return 0.0
        target = " ".join(line_words)
        cand = " ".join(cand_words)
        seq = SequenceMatcher(None, target, cand).ratio()
        line_set = set(line_words)
        cand_set = set(cand_words)
        overlap = len(line_set.intersection(cand_set)) / max(1, len(line_set))
        first_word_bonus = 0.0
        if cand_words and line_words:
            if cand_words[0] == line_words[0]:
                first_word_bonus += 0.08
            elif line_words[0] in cand_words[:2]:
                first_word_bonus += 0.04
        return (0.62 * seq) + (0.38 * overlap) + first_word_bonus

    line_ts = {}
    word_timings: List[WordTiming] = []
    cursor = 0
    max_w = len(whisper_norm)
    grouped: Dict[int, List[dict]] = {}
    for entry in _lyric_word_entries(lyrics_lines):
        grouped.setdefault(entry["line_index"], []).append(entry)

    for line_idx, line in enumerate(lyrics_lines):
        if _is_marker(line):
            continue
        line_entries = grouped.get(line_idx, [])
        line_words = [entry["normalized"] for entry in line_entries]
        if not line_words:
            continue

        lw_len = len(line_words)
        start_lo = min(cursor, max_w - 1)
        start_hi = min(max_w - 1, start_lo + 240)
        best_start = start_lo
        best_end = min(max_w, start_lo + max(1, lw_len))
        best_score = -1.0

        for start in range(start_lo, start_hi + 1):
            for span in range(max(1, lw_len - 3), lw_len + 6):
                end = min(max_w, start + span)
                if end <= start:
                    continue
                score = _window_score(line_words, whisper_norm[start:end])
                score_adj = score - ((start - start_lo) * 0.0004)
                if score_adj > best_score:
                    best_score = score_adj
                    best_start = start
                    best_end = end

        line_ts[line_idx] = int(whisper_entries[best_start]["start"] * 1000)
        cursor = min(max_w - 1, best_start + max(1, lw_len - 1))

        span = max(1, best_end - best_start)
        for idx, entry in enumerate(line_entries):
            if span == 1:
                w_idx = best_start
            else:
                offset = int(round(idx * ((span - 1) / max(1, len(line_entries) - 1))))
                w_idx = min(best_end - 1, best_start + offset)
            whisper_entry = whisper_entries[w_idx]
            word_timings.append(
                WordTiming(
                    line_index=line_idx,
                    word_index=entry["word_index"],
                    word=entry["word"],
                    start_ms=int(whisper_entry["start"] * 1000),
                    end_ms=int(whisper_entry["end"] * 1000),
                    matched=True,
                    confidence=round(max(0.0, min(1.0, best_score)), 3),
                )
            )

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

    prev_non_marker_ts = -1
    for i, (line, ts) in enumerate(result):
        if _is_marker(line):
            continue
        if ts <= prev_non_marker_ts:
            delta = (prev_non_marker_ts + 120) - ts
            ts = prev_non_marker_ts + 120
            result[i][1] = ts
            for word_timing in word_timings:
                if word_timing.line_index == i:
                    word_timing.start_ms += delta
                    word_timing.end_ms += delta
        prev_non_marker_ts = ts

    return StructuredAlignment(lines=[(line, ts) for line, ts in result], word_timings=word_timings)


def _build_candidate_result(
    lyrics_lines: List[str],
    whisper_words: List[dict],
    duration_ms: int,
    *,
    detected_language: Optional[str],
    language_probability: Optional[float],
    transcription_pass: str,
) -> AlignmentResult:
    quality = "good"
    warnings: List[str] = []

    structured = _align_lines_to_words(lyrics_lines, whisper_words)
    unique_ratio, coverage_ratio, max_ts = _summarize_line_metrics(structured.lines, duration_ms)
    non_marker_ts = [ts for line, ts in structured.lines if not _is_marker(line) and _normalize(line)]

    if len(non_marker_ts) > 6 and unique_ratio < 0.35:
        logger.warning(
            "Low DTW timestamp diversity (%d unique / %d lines). Falling back to progressive matching.",
            len(set(non_marker_ts)),
            len(non_marker_ts),
        )
        quality = "degraded"
        warnings.append("Progressive alignment fallback used after low DTW diversity.")
        structured = _align_lines_progressive(lyrics_lines, whisper_words)
        unique_ratio, coverage_ratio, max_ts = _summarize_line_metrics(structured.lines, duration_ms)

    if duration_ms > 0 and len(non_marker_ts) > 6 and (unique_ratio < 0.45 or max_ts < int(duration_ms * 0.25)):
        logger.warning(
            "Timestamp coverage still low (unique_ratio=%.2f, max_ts=%dms, duration=%dms). Applying full-song proportional fallback.",
            unique_ratio,
            max_ts,
            duration_ms,
        )
        quality = "fallback"
        warnings.append("Approximate spread timing applied after low timestamp coverage.")
        lines = _spread_timestamps_by_lyrics(lyrics_lines, duration_ms)
        structured = StructuredAlignment(
            lines=lines,
            word_timings=_build_proportional_word_timings(lyrics_lines, lines, duration_ms),
        )
        unique_ratio, coverage_ratio, max_ts = _summarize_line_metrics(structured.lines, duration_ms)

    if not whisper_words:
        quality = "fallback"
        warnings.append("No words detected in audio transcription.")

    total_words = len(structured.word_timings)
    matched_words = sum(1 for timing in structured.word_timings if timing.matched)
    word_match_ratio = matched_words / max(1, total_words)

    report = {
        "quality": quality,
        "warnings": warnings,
        "line_count": len(structured.lines),
        "duration_ms": duration_ms,
        "whisper_word_count": len(whisper_words),
        "coverage_ratio": round(coverage_ratio, 3),
        "unique_ratio": round(unique_ratio, 3),
        "word_match_ratio": round(word_match_ratio, 3),
        "language": detected_language,
        "language_probability": round(language_probability, 3) if language_probability is not None else None,
        "transcription_pass": transcription_pass,
    }
    return AlignmentResult(
        lines=structured.lines,
        quality=quality,
        warnings=warnings,
        report=report,
        word_timings=structured.word_timings,
        detected_language=detected_language,
        language_probability=language_probability,
        transcription_pass=transcription_pass,
    )


def _needs_retry(result: AlignmentResult) -> bool:
    report = result.report or {}
    if result.quality != "good":
        return True
    if int(report.get("whisper_word_count") or 0) < 8:
        return True
    if float(report.get("word_match_ratio") or 0.0) < 0.55:
        return True
    if float(report.get("coverage_ratio") or 0.0) < 0.30:
        return True
    return False


def _candidate_score(result: AlignmentResult) -> Tuple[float, float, float, float, float]:
    report = result.report or {}
    return (
        float(_QUALITY_RANK.get(result.quality, -1)),
        float(report.get("word_match_ratio") or 0.0),
        float(report.get("coverage_ratio") or 0.0),
        float(report.get("unique_ratio") or 0.0),
        float(report.get("whisper_word_count") or 0.0),
    )


def align_lyrics_to_audio(
    mp3_path: str,
    lyrics_text: str,
    job_dir: str,
    job_id: Optional[str] = None,
    *,
    timestamp_mode: str = "line",
) -> AlignmentResult:
    """
    Main entry point for alignment.
    Returns aligned lines plus sync quality metadata.
    """
    job_started = time.monotonic()
    mp3_name = Path(mp3_path).name
    _job_log(job_id, "stage=alignment_start", file=mp3_name, model=MODEL_SIZE, timestamp_mode=timestamp_mode)

    wav_path = _convert_to_wav(mp3_path, job_dir, job_id=job_id)
    duration_ms = _get_audio_duration_ms(wav_path)
    lyrics_lines = [line.strip() for line in lyrics_text.splitlines() if line.strip()]
    if not lyrics_lines:
        raise ValueError("No lyrics lines found")
    _job_log(job_id, "stage=lyrics_parsed", lines=len(lyrics_lines))

    attempts = [
        {"audio_path": wav_path, "beam_size": 1, "vad_filter": False, "pass_name": "fast"},
        {"audio_path": wav_path, "beam_size": 5, "vad_filter": True, "pass_name": "vad_retry"},
    ]

    focused_path = _create_vocal_focus_wav(wav_path, job_dir, job_id=job_id)
    if focused_path:
        attempts.append(
            {"audio_path": focused_path, "beam_size": 5, "vad_filter": True, "pass_name": "vocal_focus"}
        )

    candidates: List[AlignmentResult] = []
    for idx, attempt in enumerate(attempts):
        if idx > 0 and candidates and not _needs_retry(max(candidates, key=_candidate_score)):
            break
        transcription = _transcribe_with_word_timestamps(
            attempt["audio_path"],
            lyrics_text=lyrics_text,
            job_id=job_id,
            beam_size=attempt["beam_size"],
            vad_filter=attempt["vad_filter"],
            pass_name=attempt["pass_name"],
        )
        _job_log(
            job_id,
            "stage=dtw_start",
            transcription_pass=attempt["pass_name"],
            lyric_words=sum(len(_normalize(line).split()) for line in lyrics_lines),
            whisper_words=len(transcription.words),
        )
        align_started = time.monotonic()
        candidate = _build_candidate_result(
            lyrics_lines,
            transcription.words,
            duration_ms,
            detected_language=transcription.detected_language,
            language_probability=transcription.language_probability,
            transcription_pass=attempt["pass_name"],
        )
        _job_log(
            job_id,
            "stage=dtw_done",
            transcription_pass=attempt["pass_name"],
            elapsed=f"{time.monotonic() - align_started:.1f}s",
            lines=len(candidate.lines),
            quality=candidate.quality,
        )
        candidates.append(candidate)

    result = max(candidates, key=_candidate_score) if candidates else _build_candidate_result(
        lyrics_lines,
        [],
        duration_ms,
        detected_language=None,
        language_probability=None,
        transcription_pass="none",
    )
    if result.report is not None:
        result.report["timestamp_mode"] = timestamp_mode

    elapsed = time.monotonic() - job_started
    _job_log(
        job_id,
        "stage=alignment_done",
        file=mp3_name,
        lines=len(result.lines),
        elapsed=f"{elapsed:.1f}s",
        quality=result.quality,
        transcription_pass=result.transcription_pass,
    )
    for line, ts in result.lines[:3]:
        logger.info("[sync] job=%s preview [%dms] %s", job_id or "-", ts, line[:60])
    if len(result.lines) > 3:
        logger.info("[sync] job=%s preview ... and %d more lines", job_id or "-", len(result.lines) - 3)
    return result


def transcribe_audio_to_text(mp3_path: str, job_dir: str, job_id: Optional[str] = None) -> str:
    """
    Transcribe audio to plain text lyrics using faster-whisper.
    Used as a fallback when no embedded lyrics are present.
    """
    wav_path = _convert_to_wav(mp3_path, job_dir, job_id)
    model = _get_model()

    _job_log(job_id, "stage=transcribe_text_start", vad_filter=True)
    started = time.monotonic()
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
        elapsed=f"{elapsed:.1f}s",
    )
    return "\n".join(text_lines).strip()
