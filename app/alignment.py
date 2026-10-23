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
        if family in {"hiragana", "katakana", "cjk"}:
            flush()
            tokens.append(ch)
            continue
        if category.startswith(("L", "N", "M")) or family in {"arabic", "cyrillic", "devanagari", "latin"}:
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


_QUOTED_SPEECH_RE = re.compile(r'["“„]([^"”‟]{2,})["”‟]')
_BRACKET_INLINE_RE = re.compile(r"\[[^\]]*\]|\([^\)]*\)")
_SQUARE_BRACKET_INLINE_RE = re.compile(r"\[[^\]]*\]")
_PAREN_STAGE_CUE_RE = re.compile(
    r"\(\s*(?:"
    r"x\s*\d+|\d+\s*x|"
    r"repeat(?:\s+x?\d+)?|"
    r"instrumental(?:\s+\w+)?|"
    r"(?:guitar|drum|synth|bass|piano|sax|violin|trumpet)?\s*solo|"
    r"chorus|verse(?:\s*\d+)?|bridge|outro|intro|hook|pre-?chorus|refrain|"
    r"fade\s+out|spoken|whispered|ad[- ]?libs?"
    r")\s*\)",
    re.IGNORECASE,
)
_LEADING_SECTION_LABEL_RE = re.compile(
    r"^\s*(?:verse|chorus|bridge|intro|outro|hook|pre-?chorus|refrain|strophe|part)\s*\d*\s*:\s*",
    re.IGNORECASE,
)
_REPEATED_CHAR_RE = re.compile(r"(.)\1+")


def _extract_marker_speech(text: str) -> str:
    """Extract quoted spoken words from bracketed/parenthesized stage cues like [Spoken intro: \"...\"]."""
    clean = text.strip()
    if not ((clean.startswith("[") and clean.endswith("]")) or (clean.startswith("(") and clean.endswith(")"))):
        return ""
    matches = [m.strip() for m in _QUOTED_SPEECH_RE.findall(clean) if m.strip()]
    return " ".join(matches)


def _strip_inline_stage_markers(text: str) -> str:
    """
    Strip inline structural/stage tags like [Verse 1], [x2], (x2), (Guitar Solo), or 'Chorus:'
    while keeping quoted speech inside brackets and sung backing vocals in parentheses.
    """
    if not text:
        return ""
    if _is_marker(text):
        return _extract_marker_speech(text)

    def _replace_square_bracket(match: re.Match) -> str:
        spoken = _extract_marker_speech(match.group(0))
        return f" {spoken} " if spoken else " "

    cleaned = _SQUARE_BRACKET_INLINE_RE.sub(_replace_square_bracket, text)
    cleaned = _PAREN_STAGE_CUE_RE.sub(" ", cleaned)
    cleaned = _LEADING_SECTION_LABEL_RE.sub("", cleaned)
    return cleaned.strip()


def _phonetic_normalize(word: str) -> str:
    """
    Lightweight phonetic normalization for sung words across EN/DE/ES/FR:
    collapses elongated vowels/consonants ('loooove' -> 'love') and unifies common
    grapheme equivalents ('ph' -> 'f', 'ck' -> 'k', 'dt' -> 't', 'th' -> 't', 'sh'/'sch' -> 's').
    """
    if not word:
        return ""
    s = word.casefold()
    s = (
        s.replace("sch", "s")
        .replace("sh", "s")
        .replace("ph", "f")
        .replace("ck", "k")
        .replace("dt", "t")
        .replace("th", "t")
        .replace("tz", "z")
        .replace("qu", "kw")
    )
    if len(s) >= 5 and s.endswith("ing"):
        s = s[:-1]
    s = _REPEATED_CHAR_RE.sub(r"\1", s)
    return s


def _lyric_word_entries(lyrics_lines: List[str]) -> List[dict]:
    entries = []
    for line_idx, line in enumerate(lyrics_lines):
        source_text = _strip_inline_stage_markers(line)
        if not source_text:
            continue
        word_idx = 0
        for raw in _tokenize(source_text):
            normalized = _normalize_token(raw)
            if not normalized:
                continue
            entries.append(
                {
                    "normalized": normalized,
                    "phonetic": _phonetic_normalize(normalized),
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
                "phonetic": _phonetic_normalize(normalized),
                "start": float(item.get("start") or 0.0),
                "end": float(item.get("end") or item.get("start") or 0.0),
            }
        )
    return entries


def _strip_markers_for_language_detection(lyrics_text: str) -> str:
    cleaned_lines = []
    for raw_line in lyrics_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_marker(line):
            spoken = _extract_marker_speech(line)
            if spoken:
                cleaned_lines.append(spoken)
            continue
        line = _BRACKET_INLINE_RE.sub(" ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines) if cleaned_lines else lyrics_text


def _detect_language_hint(lyrics_text: Optional[str]) -> Optional[str]:
    if not lyrics_text:
        return None

    cleaned_text = _strip_markers_for_language_detection(lyrics_text)

    script_counts: Dict[str, int] = {}
    for ch in cleaned_text:
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

    tokens = [t.casefold() for t in _tokenize(cleaned_text)]
    if not tokens:
        return None

    stopwords = {
        "en": {"the", "and", "you", "are", "this", "with", "for", "that", "from", "your"},
        "es": {"que", "el", "la", "con", "los", "para", "por", "una", "del", "las"},
        "de": {"der", "die", "das", "und", "ist", "ich", "nicht", "ein", "mit", "auf", "wir", "du", "dem", "den"},
        "fr": {"le", "et", "les", "une", "dans", "des", "est", "pour", "pas", "sur"},
        "it": {"che", "il", "un", "non", "per", "una", "con", "sono", "della", "nel"},
        "pt": {"não", "você", "vocês", "coração", "também", "uma", "com", "para", "dos"},
    }

    scores: Dict[str, int] = {}
    for lang, words in stopwords.items():
        count = sum(1 for tok in tokens if tok in words)
        if count > 0:
            scores[lang] = count

    if any(ch in cleaned_text.casefold() for ch in ("ä", "ö", "ü", "ß")) and "de" in scores:
        scores["de"] += 2

    if not scores:
        return None
    if len(scores) == 1:
        return next(iter(scores))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_lang, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score >= 3 and best_score >= second_score * 2:
        return best_lang
    return None


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


def _build_contiguous_clip_timestamps(duration_ms: int, window_sec: float = 30.0) -> Optional[List[float]]:
    """Build contiguous [start, end, ...] windows so faster-whisper cannot skip 30s on single_timestamp_ending."""
    if duration_ms <= 0:
        return None
    duration_sec = round(duration_ms / 1000.0, 2)
    if duration_sec <= window_sec:
        return None
    clips: List[float] = []
    pos = 0.0
    while pos + 0.25 < duration_sec:
        end = min(round(pos + window_sec, 2), duration_sec)
        clips.extend([round(pos, 2), end])
        pos = end
    return clips if len(clips) >= 2 else None


def _dedupe_hallucinated_words(words: List[dict], max_consecutive: int = 2, max_gap_sec: float = 1.5) -> List[dict]:
    """Filter out Whisper repetition loops (>max_consecutive identical tokens in a row)."""
    cleaned: List[dict] = []
    streak_norm = ""
    streak_count = 0
    prev_end = 0.0
    for item in words:
        norm = _normalize_token(str(item.get("word") or ""))
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        if norm and norm == streak_norm and (start - prev_end) <= max_gap_sec:
            streak_count += 1
            if streak_count > max_consecutive:
                prev_end = end
                continue
        else:
            streak_norm = norm
            streak_count = 1
        prev_end = end
        cleaned.append(item)
    return cleaned


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
    duration_ms = _get_audio_duration_ms(wav_path) if not vad_filter else 0
    clip_timestamps = _build_contiguous_clip_timestamps(duration_ms) if not vad_filter else None

    _job_log(
        job_id,
        "stage=transcribe_start",
        transcription_pass=pass_name,
        language_hint=language or "auto",
        beam_size=beam_size,
        vad_filter=vad_filter,
        windows=(len(clip_timestamps) // 2) if clip_timestamps else "auto",
    )
    started = time.monotonic()

    transcribe_kwargs: Dict[str, Any] = {
        "beam_size": beam_size,
        "word_timestamps": True,
        "language": language,
        "initial_prompt": None,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "vad_filter": vad_filter,
    }
    if clip_timestamps is not None:
        transcribe_kwargs["clip_timestamps"] = clip_timestamps

    segments, info = model.transcribe(wav_path, **transcribe_kwargs)

    raw_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                raw_words.append(
                    {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                    }
                )

    words = _dedupe_hallucinated_words(raw_words)

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


def _word_lexical_similarity(
    w1: str,
    w2: str,
    p1: Optional[str] = None,
    p2: Optional[str] = None,
) -> float:
    if not w1 or not w2:
        return 0.0
    if w1 == w2:
        return 1.0
    p1 = p1 if p1 is not None else _phonetic_normalize(w1)
    p2 = p2 if p2 is not None else _phonetic_normalize(w2)
    if p1 and p1 == p2 and len(p1) >= 2:
        return 0.94

    seq_raw = SequenceMatcher(None, w1, w2).ratio()
    seq_phon = SequenceMatcher(None, p1, p2).ratio() if (p1 and p2) else seq_raw
    base = max(seq_raw, (0.45 * seq_raw) + (0.55 * seq_phon))

    prefix_bonus = 0.0
    if len(w1) >= 3 and len(w2) >= 3 and (w1[:3] == w2[:3] or (p1 and p2 and p1[:3] == p2[:3])):
        prefix_bonus = 0.08
    elif len(w1) >= 2 and len(w2) >= 2 and (w1[:2] == w2[:2] or (p1 and p2 and p1[:2] == p2[:2])):
        prefix_bonus = 0.04

    if min(len(w1), len(w2)) == 1 and max(len(w1), len(w2)) >= 4:
        base *= 0.65

    return float(min(1.0, base + prefix_bonus))


def _build_word_cost_matrix(
    lyrics_word_entries: List[dict],
    whisper_entries: List[dict],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build (lexical_dist, dtw_cost_matrix, vocab_size) using character unigrams,
    boundary-padded character bigrams, phonetic bigrams, prefix bigrams, and a
    soft relative-position prior to prevent long-range stopword/anagram jumps.
    """
    feature_vocab: Dict[Tuple[str, str], int] = {}

    def _extract_features(entry: dict) -> List[Tuple[Tuple[str, str], float]]:
        w = entry["normalized"]
        p = entry.get("phonetic") or _phonetic_normalize(w)
        feats: List[Tuple[Tuple[str, str], float]] = []
        for ch in w:
            feats.append((("u", ch), 0.45))
        padded_w = f"^{w}$"
        for i in range(len(padded_w) - 1):
            feats.append((("b", padded_w[i : i + 2]), 1.15))
        if p:
            padded_p = f"^{p}$"
            for i in range(len(padded_p) - 1):
                feats.append((("p", padded_p[i : i + 2]), 0.90))
        if len(w) >= 2:
            feats.append((("pre", w[:2]), 0.70))
        if len(w) >= 3:
            feats.append((("pre3", w[:3]), 0.55))
        return feats

    lyric_feats = [_extract_features(entry) for entry in lyrics_word_entries]
    whisper_feats = [_extract_features(entry) for entry in whisper_entries]

    for flist in lyric_feats:
        for key, _weight in flist:
            if key not in feature_vocab:
                feature_vocab[key] = len(feature_vocab)
    for flist in whisper_feats:
        for key, _weight in flist:
            if key not in feature_vocab:
                feature_vocab[key] = len(feature_vocab)

    vocab_size = len(feature_vocab)
    if vocab_size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0

    def _to_matrix(feats_list: List[List[Tuple[Tuple[str, str], float]]]) -> np.ndarray:
        mat = np.zeros((len(feats_list), vocab_size), dtype=np.float32)
        for row_idx, flist in enumerate(feats_list):
            for key, weight in flist:
                mat[row_idx, feature_vocab[key]] += weight
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat /= norms
        return mat

    lyrics_vecs = _to_matrix(lyric_feats)
    whisper_vecs = _to_matrix(whisper_feats)

    sim_matrix = np.clip(lyrics_vecs @ whisper_vecs.T, 0.0, 1.0)

    for l_idx, l_entry in enumerate(lyrics_word_entries):
        lw = l_entry["normalized"]
        lp = l_entry.get("phonetic") or ""
        for w_idx, w_entry in enumerate(whisper_entries):
            ww = w_entry["normalized"]
            wp = w_entry.get("phonetic") or ""
            if lw == ww:
                sim_matrix[l_idx, w_idx] = 1.0
            elif lp and lp == wp and len(lp) >= 2:
                if sim_matrix[l_idx, w_idx] < 0.94:
                    sim_matrix[l_idx, w_idx] = 0.94

    lexical_dist = np.clip(1.0 - sim_matrix, 0.0, 1.0).astype(np.float64)
    cost_matrix = lexical_dist.copy()

    num_l = len(lyrics_word_entries)
    num_w = len(whisper_entries)
    if num_l >= 6 and num_w >= 6:
        l_rel = np.linspace(0.0, 1.0, num_l, dtype=np.float64)
        w_rel = np.linspace(0.0, 1.0, num_w, dtype=np.float64)
        pos_diff = np.abs(l_rel[:, None] - w_rel[None, :])
        short_mask = np.array(
            [1.0 if len(entry["normalized"]) <= 2 else 0.0 for entry in lyrics_word_entries],
            dtype=np.float64,
        )[:, None]
        base_penalty = 0.16 * np.maximum(0.0, pos_diff - 0.15)
        short_penalty = 0.14 * short_mask * np.maximum(0.0, pos_diff - 0.10)
        cost_matrix = cost_matrix + base_penalty + short_penalty

    return lexical_dist, np.ascontiguousarray(cost_matrix, dtype=np.float64), vocab_size


def _interpolate_line_word_timings(
    lyrics_lines: List[str],
    word_timings: List[WordTiming],
    whisper_entries: List[dict],
) -> Tuple[Dict[int, int], List[WordTiming]]:
    """
    Filter intra-line pre-gap/post-gap outliers, interpolate unmatched words smoothly
    between surrounding matched anchor words, and derive coherent line start timestamps.
    """
    by_line: Dict[int, List[WordTiming]] = {}
    for wt in word_timings:
        by_line.setdefault(wt.line_index, []).append(wt)

    line_timestamps: Dict[int, int] = {}
    any_matched_in_song = any(wt.matched for wt in word_timings)
    max_audio_ms = int(whisper_entries[-1]["end"] * 1000) if whisper_entries else 0

    prev_line_end_ms = 0
    for line_idx in sorted(by_line.keys()):
        line_wt = by_line[line_idx]
        if not line_wt:
            continue

        # Pre-gap outlier check: if the first matched word is stranded far before the
        # rest of the line's matched words across an instrumental/noise break, demote it.
        while True:
            m_indices = [i for i, wt in enumerate(line_wt) if wt.matched]
            if len(m_indices) < 2:
                break
            first_i = m_indices[0]
            second_i = m_indices[1]
            gap_ms = line_wt[second_i].start_ms - line_wt[first_i].end_ms
            first_conf = float(line_wt[first_i].confidence or 0.0)
            second_conf = float(line_wt[second_i].confidence or 0.0)
            first_norm_len = len(_normalize_token(line_wt[first_i].word))
            word_step = max(1, second_i - first_i)
            if gap_ms > 2800 * word_step and (
                first_conf + 0.10 < second_conf
                or (gap_ms > 4500 and first_norm_len <= 3 and second_conf >= 0.70)
            ):
                line_wt[first_i].matched = False
                continue
            break

        # Post-gap outlier check at the end of a line
        while True:
            m_indices = [i for i, wt in enumerate(line_wt) if wt.matched]
            if len(m_indices) < 2:
                break
            last_i = m_indices[-1]
            prev_i = m_indices[-2]
            gap_ms = line_wt[last_i].start_ms - line_wt[prev_i].end_ms
            last_conf = float(line_wt[last_i].confidence or 0.0)
            prev_conf = float(line_wt[prev_i].confidence or 0.0)
            word_step = max(1, last_i - prev_i)
            if gap_ms > 3200 * word_step and last_conf + 0.15 < prev_conf:
                line_wt[last_i].matched = False
                continue
            break

        m_indices = [i for i, wt in enumerate(line_wt) if wt.matched]
        if not m_indices:
            if not any_matched_in_song:
                raw_starts = [wt.start_ms for wt in line_wt if wt.start_ms > 0]
                line_timestamps[line_idx] = min(raw_starts) if raw_starts else 0
            continue

        # Enforce monotonicity among matched words within the line
        running_ms = max(0, prev_line_end_ms)
        for idx in m_indices:
            if line_wt[idx].start_ms < running_ms:
                dur = max(60, line_wt[idx].end_ms - line_wt[idx].start_ms)
                line_wt[idx].start_ms = running_ms
                line_wt[idx].end_ms = running_ms + dur
            if line_wt[idx].end_ms < line_wt[idx].start_ms:
                line_wt[idx].end_ms = line_wt[idx].start_ms + 80
            running_ms = line_wt[idx].start_ms

        # Extrapolate leading unmatched words before first_m
        first_m = m_indices[0]
        if first_m > 0:
            anchor_start = line_wt[first_m].start_ms
            est_step = 200
            earliest_allowed = max(0, prev_line_end_ms)
            avail_span = max(0, anchor_start - earliest_allowed)
            step = min(est_step, max(60, avail_span // max(1, first_m))) if avail_span > 0 else 0
            start_base = max(earliest_allowed, anchor_start - (step * first_m))
            for i in range(first_m):
                w_start = start_base + (i * step)
                w_end = min(anchor_start, w_start + max(60, step - 10))
                line_wt[i].start_ms = int(w_start)
                line_wt[i].end_ms = int(max(w_start + 50, w_end))

        # Interpolate internal unmatched words between consecutive matched anchors
        for pair_idx in range(len(m_indices) - 1):
            left_i = m_indices[pair_idx]
            right_i = m_indices[pair_idx + 1]
            gap_count = right_i - left_i - 1
            if gap_count <= 0:
                continue
            left_end = line_wt[left_i].end_ms
            right_start = line_wt[right_i].start_ms
            if right_start <= left_end:
                left_end = line_wt[left_i].start_ms
                right_start = max(left_end + gap_count * 70, line_wt[right_i].end_ms)
            span_ms = max(gap_count * 60, right_start - left_end)
            step = span_ms / float(gap_count + 1)
            for offset in range(1, gap_count + 1):
                cur_i = left_i + offset
                w_start = int(round(left_end + (offset - 0.5) * step))
                w_end = int(round(min(right_start, w_start + max(60.0, step * 0.85))))
                line_wt[cur_i].start_ms = max(line_wt[cur_i - 1].start_ms, w_start)
                line_wt[cur_i].end_ms = max(line_wt[cur_i].start_ms + 50, w_end)

        # Extrapolate trailing unmatched words after last_m
        last_m = m_indices[-1]
        if last_m < len(line_wt) - 1:
            trail_count = (len(line_wt) - 1) - last_m
            anchor_end = line_wt[last_m].end_ms
            step = 180
            for offset in range(1, trail_count + 1):
                cur_i = last_m + offset
                w_start = anchor_end + (offset - 1) * step
                w_end = w_start + step - 20
                if max_audio_ms > 0:
                    w_start = min(max_audio_ms, w_start)
                    w_end = min(max_audio_ms, max(w_start + 50, w_end))
                line_wt[cur_i].start_ms = int(w_start)
                line_wt[cur_i].end_ms = int(w_end)

        line_timestamps[line_idx] = int(line_wt[0].start_ms)
        prev_line_end_ms = int(max(line_wt[-1].start_ms, line_wt[-1].end_ms))

    # Interpolate completely unanchored lines between surrounding anchored lines
    if any_matched_in_song:
        non_marker_lines = [
            i for i, line in enumerate(lyrics_lines) if i in by_line and not _is_marker(line)
        ]
        unanchored = [i for i in non_marker_lines if i not in line_timestamps]
        for u_idx in unanchored:
            prev_anchored = [i for i in non_marker_lines if i < u_idx and i in line_timestamps]
            next_anchored = [i for i in non_marker_lines if i > u_idx and i in line_timestamps]
            start_bound = (
                max(wt.end_ms for wt in by_line[prev_anchored[-1]])
                if prev_anchored
                else 0
            )
            end_bound = (
                line_timestamps[next_anchored[0]]
                if next_anchored
                else max(start_bound + 1500, max_audio_ms)
            )
            # Check how many consecutive unanchored lines sit in this gap
            prev_a = prev_anchored[-1] if prev_anchored else -1
            next_a = next_anchored[0] if next_anchored else len(lyrics_lines)
            block = [i for i in non_marker_lines if prev_a < i < next_a and i not in line_timestamps]
            if not block:
                block = [u_idx]
            span = max(len(block) * 400, end_bound - start_bound)
            block_step = span // max(1, len(block) + 1)
            for b_pos, b_line_idx in enumerate(block):
                l_start = int(start_bound + (b_pos + 1) * block_step)
                l_end = int(min( max(l_start + 300, end_bound), l_start + block_step))
                line_timestamps[b_line_idx] = l_start
                b_words = by_line.get(b_line_idx, [])
                if b_words:
                    w_step = max(70, (l_end - l_start) // max(1, len(b_words)))
                    for w_pos, wt in enumerate(b_words):
                        wt.start_ms = int(l_start + w_pos * w_step)
                        wt.end_ms = int(max(wt.start_ms + 60, min(l_end, wt.start_ms + w_step)))

    return line_timestamps, word_timings


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

    lexical_dist, cost_matrix, vocab_size = _build_word_cost_matrix(
        lyrics_word_entries, whisper_entries
    )
    if vocab_size == 0 or cost_matrix.size == 0:
        logger.warning("Vocabulary is empty - returning zero timestamps")
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

    logger.info(
        "Aligning %d lyric words vs %d whisper words (feature vocab: %d, shape: %s)",
        len(lyrics_word_entries),
        len(whisper_entries),
        vocab_size,
        cost_matrix.shape,
    )

    try:
        alignment = dtw(cost_matrix, step_pattern="symmetric2")
    except ValueError as exc:
        logger.error("DTW failed: %s", exc)
        return StructuredAlignment(
            lines=[(line, 0) for line in lyrics_lines],
            word_timings=_empty_word_timings(lyrics_lines),
        )

    lyric_word_to_whisper_candidates: Dict[int, List[int]] = {}
    for l_idx, w_idx in zip(alignment.index1, alignment.index2):
        lyric_word_to_whisper_candidates.setdefault(int(l_idx), []).append(int(w_idx))

    lyric_word_to_whisper: Dict[int, int] = {}
    num_l = len(lyrics_word_entries)
    for l_idx, candidates in lyric_word_to_whisper_candidates.items():
        best_lex = min(float(lexical_dist[l_idx, w_idx]) for w_idx in candidates)
        near_best = [
            w_idx for w_idx in candidates if float(lexical_dist[l_idx, w_idx]) <= best_lex + 0.02
        ]
        if len(near_best) == 1 or best_lex <= 0.35:
            lyric_word_to_whisper[l_idx] = int(
                min(near_best, key=lambda w_idx: (round(float(cost_matrix[l_idx, w_idx]), 3), w_idx))
            )
        else:
            # Unmatched word stretched horizontally across a gap: if it starts a new line,
            # prefer the end of the gap rather than snapping back to the previous line.
            cur_line = lyrics_word_entries[l_idx]["line_index"]
            prev_line = lyrics_word_entries[l_idx - 1]["line_index"] if l_idx > 0 else cur_line
            next_line = (
                lyrics_word_entries[l_idx + 1]["line_index"] if l_idx + 1 < num_l else cur_line
            )
            if l_idx > 0 and cur_line != prev_line and cur_line == next_line:
                lyric_word_to_whisper[l_idx] = int(near_best[-1])
            else:
                lyric_word_to_whisper[l_idx] = int(near_best[0])

    # Identify vertical pileups (multiple lyric words mapped to the same Whisper token)
    whisper_to_lyrics: Dict[int, List[int]] = {}
    for lw_idx, w_idx in lyric_word_to_whisper.items():
        whisper_to_lyrics.setdefault(w_idx, []).append(lw_idx)

    best_lyric_for_whisper: Dict[int, int] = {}
    for w_idx, lw_list in whisper_to_lyrics.items():
        best_lyric_for_whisper[w_idx] = min(
            lw_list,
            key=lambda lw_idx: (round(float(lexical_dist[lw_idx, w_idx]), 3), lw_idx),
        )

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

        whisper_entry = whisper_entries[mapped_idx]
        lex_d = float(lexical_dist[lw_idx, mapped_idx])
        confidence = round(max(0.0, min(1.0, 1.0 - lex_d)), 3)
        is_primary_owner = best_lyric_for_whisper.get(mapped_idx) == lw_idx
        exact_or_phonetic = (
            entry["normalized"] == whisper_entry["normalized"]
            or (entry.get("phonetic") and entry.get("phonetic") == whisper_entry.get("phonetic"))
        )
        is_matched = bool(
            (is_primary_owner and confidence >= 0.30)
            or (exact_or_phonetic and confidence >= 0.75)
            or (len(lyrics_word_entries) <= 4 and is_primary_owner and confidence >= 0.20)
        )
        word_timings.append(
            WordTiming(
                line_index=entry["line_index"],
                word_index=entry["word_index"],
                word=entry["word"],
                start_ms=int(whisper_entry["start"] * 1000),
                end_ms=int(whisper_entry["end"] * 1000),
                matched=is_matched,
                confidence=confidence,
            )
        )

    line_timestamps, word_timings = _interpolate_line_word_timings(
        lyrics_lines, word_timings, whisper_entries
    )

    non_marker_lines = [i for i, line in enumerate(lyrics_lines) if not _is_marker(line) and _normalize(line)]
    non_marker_ts = [line_timestamps[i] for i in non_marker_lines if i in line_timestamps]
    if len(non_marker_lines) > 1 and len(set(non_marker_ts)) <= 1:
        logger.warning(
            "Low timestamp diversity detected (%d unique for %d lines). Applying proportional fallback.",
            len(set(non_marker_ts)),
            len(non_marker_lines),
        )
        lyric_words_per_line = [
            max(1, len(_normalize(_strip_inline_stage_markers(lyrics_lines[i])).split()))
            for i in non_marker_lines
        ]
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

    # Enforce non-decreasing timestamps across non-marker lines and sync word_timings
    prev_non_marker_ts = -1
    for i, (line, ts) in enumerate(result):
        if _is_marker(line):
            continue
        if ts < prev_non_marker_ts:
            delta = prev_non_marker_ts - ts
            ts = prev_non_marker_ts
            result[i][1] = ts
            for wt in word_timings:
                if wt.line_index == i:
                    wt.start_ms += delta
                    wt.end_ms += delta
        prev_non_marker_ts = ts

    return StructuredAlignment(lines=[(line, ts) for line, ts in result], word_timings=word_timings)


def _align_lines_progressive(lyrics_lines: List[str], whisper_words: List[dict]) -> StructuredAlignment:
    """
    Progressive forward alignment fallback.

    For each non-marker lyric line, find the best matching forward window in
    Whisper words using lexical + phonetic similarity while avoiding runaway
    cursor jumps when individual lines are missing in the transcription.
    """
    if not whisper_words:
        return StructuredAlignment(lines=[(line, 0) for line in lyrics_lines], word_timings=_empty_word_timings(lyrics_lines))

    whisper_entries = _whisper_entries(whisper_words)
    whisper_norm = [entry["normalized"] for entry in whisper_entries]
    whisper_phon = [entry.get("phonetic") or _phonetic_normalize(entry["normalized"]) for entry in whisper_entries]
    if not whisper_norm:
        return StructuredAlignment(lines=[(line, 0) for line in lyrics_lines], word_timings=_empty_word_timings(lyrics_lines))

    def _window_score(
        line_words: List[str],
        line_phon: List[str],
        cand_words: List[str],
        cand_phon: List[str],
    ) -> float:
        if not line_words or not cand_words:
            return 0.0
        # Word-level lexical/phonetic match score (avoids character-level vowel/space inflation)
        matched_sims: List[float] = []
        used_cand: set[int] = set()
        for lw, lp in zip(line_words, line_phon):
            best_s = 0.0
            best_c = -1
            for c_idx, (cw, cp) in enumerate(zip(cand_words, cand_phon)):
                if c_idx in used_cand:
                    continue
                s = _word_lexical_similarity(lw, cw, lp, cp)
                if s > best_s:
                    best_s = s
                    best_c = c_idx
            if best_c >= 0 and best_s >= 0.55:
                used_cand.add(best_c)
                matched_sims.append(best_s)
            else:
                matched_sims.append(0.0)

        word_match_score = sum(matched_sims) / max(1, len(line_words))
        strong_hits = sum(1 for s in matched_sims if s >= 0.70)
        hit_ratio = strong_hits / max(1, len(line_words))

        first_word_bonus = 0.0
        if cand_words and line_words:
            s0 = _word_lexical_similarity(line_words[0], cand_words[0], line_phon[0], cand_phon[0])
            if s0 >= 0.85:
                first_word_bonus += 0.08
            elif len(cand_words) > 1 and _word_lexical_similarity(
                line_words[0], cand_words[1], line_phon[0], cand_phon[1]
            ) >= 0.85:
                first_word_bonus += 0.04

        return (0.60 * word_match_score) + (0.40 * hit_ratio) + first_word_bonus

    line_ts: Dict[int, int] = {}
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
        line_phon = [entry.get("phonetic") or _phonetic_normalize(entry["normalized"]) for entry in line_entries]
        if not line_words:
            continue

        lw_len = len(line_words)
        start_lo = min(cursor, max_w - 1)
        start_hi = min(max_w - 1, start_lo + 160)
        best_start = start_lo
        best_end = min(max_w, start_lo + max(1, lw_len))
        best_raw_score = 0.0
        best_score = -1.0
        base_time_sec = float(whisper_entries[start_lo]["start"])

        for start in range(start_lo, start_hi + 1):
            jump_words = start - start_lo
            time_gap_sec = max(0.0, float(whisper_entries[start]["start"]) - base_time_sec)
            jump_penalty = (jump_words * 0.002) + (max(0.0, time_gap_sec - 15.0) * 0.004)
            for span in range(max(1, lw_len - 2), lw_len + 5):
                end = min(max_w, start + span)
                if end <= start:
                    continue
                raw_score = _window_score(
                    line_words,
                    line_phon,
                    whisper_norm[start:end],
                    whisper_phon[start:end],
                )
                score_adj = raw_score - jump_penalty
                if score_adj > best_score:
                    best_score = score_adj
                    best_raw_score = raw_score
                    best_start = start
                    best_end = end

        # Guard against jumping across the song on an unmappable / missed line
        jump_distance = best_start - start_lo
        time_jump_sec = max(0.0, float(whisper_entries[best_start]["start"]) - base_time_sec)
        is_reliable_window = (
            best_raw_score >= 0.42
            or (jump_distance <= 4 and time_jump_sec <= 10.0 and best_raw_score >= 0.30)
        )
        if not is_reliable_window:
            for entry in line_entries:
                word_timings.append(
                    WordTiming(
                        line_index=line_idx,
                        word_index=entry["word_index"],
                        word=entry["word"],
                        start_ms=0,
                        end_ms=0,
                        matched=False,
                        confidence=round(max(0.0, best_raw_score), 3),
                    )
                )
            continue

        line_ts[line_idx] = int(whisper_entries[best_start]["start"] * 1000)
        cursor = min(max_w - 1, best_start + max(1, (best_end - best_start) - 1))

        # Align words monotonically inside the matched window [best_start, best_end)
        win_cursor = best_start
        span = max(1, best_end - best_start)
        for idx, entry in enumerate(line_entries):
            best_w_idx = min(best_end - 1, win_cursor)
            best_w_sim = -1.0
            remaining_entries = max(1, len(line_entries) - idx)
            search_hi = max(win_cursor + 1, best_end - remaining_entries + 2)
            search_hi = min(best_end, search_hi)
            for cand_w_idx in range(win_cursor, search_hi):
                sim = _word_lexical_similarity(
                    entry["normalized"],
                    whisper_norm[cand_w_idx],
                    entry.get("phonetic"),
                    whisper_phon[cand_w_idx],
                )
                if sim > best_w_sim:
                    best_w_sim = sim
                    best_w_idx = cand_w_idx
            if best_w_sim < 0.28 and span > 1:
                offset = int(round(idx * ((span - 1) / max(1, len(line_entries) - 1))))
                best_w_idx = min(best_end - 1, best_start + offset)
            win_cursor = min(best_end - 1, best_w_idx + (1 if best_w_sim >= 0.45 else 0))
            whisper_entry = whisper_entries[best_w_idx]
            word_conf = round(max(0.0, min(1.0, 0.5 * best_raw_score + 0.5 * max(0.0, best_w_sim))), 3)
            word_timings.append(
                WordTiming(
                    line_index=line_idx,
                    word_index=entry["word_index"],
                    word=entry["word"],
                    start_ms=int(whisper_entry["start"] * 1000),
                    end_ms=int(whisper_entry["end"] * 1000),
                    matched=bool(best_w_sim >= 0.30 or best_raw_score >= 0.50),
                    confidence=word_conf,
                )
            )

    interp_line_ts, word_timings = _interpolate_line_word_timings(
        lyrics_lines, word_timings, whisper_entries
    )
    for k, v in interp_line_ts.items():
        if k not in line_ts:
            line_ts[k] = v

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
        non_marker_ts = [ts for line, ts in structured.lines if not _is_marker(line) and _normalize(line)]

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

    candidates: List[AlignmentResult] = []
    vocal_focus_added = False
    idx = 0
    while idx < len(attempts):
        attempt = attempts[idx]
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
        best_candidate = max(candidates, key=_candidate_score)
        if idx == len(attempts) - 1 and not vocal_focus_added and _needs_retry(best_candidate):
            focused_path = _create_vocal_focus_wav(wav_path, job_dir, job_id=job_id)
            if focused_path:
                attempts.append(
                    {
                        "audio_path": focused_path,
                        "beam_size": 5,
                        "vad_filter": True,
                        "pass_name": "vocal_focus",
                    }
                )
                vocal_focus_added = True
        idx += 1

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
