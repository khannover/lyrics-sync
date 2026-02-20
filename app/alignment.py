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
from scipy.spatial.distance import cdist

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
                compute_type="int8",
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
                compute_type="int8",
                download_root=str(MODEL_DIR),
                local_files_only=False, 
            )
            logger.info("Model downloaded and loaded.")

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


def _transcribe_with_word_timestamps(wav_path: str, lyrics_text: str = None) -> List[dict]:
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

    logger.info("Transcribing audio (language_hint: %s, initial_prompt: %s)", 
                language, (initial_prompt[:50] + "...") if initial_prompt else "None")

    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        word_timestamps=True,
        language=language,  # Use our hint or None for auto-detect
        initial_prompt=initial_prompt,
        vad_filter=True, # Significantly improves robustness to music/silence
        vad_parameters=dict(min_silence_duration_ms=500),
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
    whisper_words = _transcribe_with_word_timestamps(wav_path, lyrics_text=lyrics_text)

    # Align user lyrics to whisper timestamps
    synced = _align_lines_to_words(lyrics_lines, whisper_words)

    logger.info("Alignment complete: %d lines", len(synced))
    for line, ts in synced[:5]:
        logger.info("  [%7d ms] %s", ts, line[:60])
    if len(synced) > 5:
        logger.info("  ... and %d more lines", len(synced) - 5)

    return synced