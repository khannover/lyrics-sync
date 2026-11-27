"""
Neural Vocal Separator for lyrics-sync (ONNX Runtime, CPU-only, no PyTorch).

Uses the public UVR MDX-Net model `Kim_Vocal_2.onnx` (~64 MB) to isolate singing
vocals from dense music mixes before Whisper transcription and forced alignment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("/app/models")
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILENAME = "kuielab_b_vocals.onnx"
MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/kuielab_b_vocals.onnx"
)

# Verified UVR registry parameters for kuielab_b_vocals.onnx (hash 6b31de20e84392859a3d09d43f089515)
SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
N_FFT = 6144
HOP_LENGTH = 1024
DIM_F = 2048
DIM_T = 256
COMPENSATE = 1.035
CHUNK_SIZE = HOP_LENGTH * (DIM_T - 1)  # 261,120 samples (~5.92s at 44.1 kHz)
TRIM = N_FFT // 2  # 3,072 samples trimmed per edge to eliminate STFT boundary artifacts
GEN_SIZE = CHUNK_SIZE - 2 * TRIM  # 254,976 samples (~5.78s clean audio per chunk)

# Thresholds for detecting pure-instrumental tracks after vocal separation.
# On true instrumentals, isolated vocals sit near -65 dBFS (p95_rms ~0.0005, active_ratio ~0.0).
# Even quiet whispered/soft singing exceeds p95_rms >= 0.01 and active_ratio >= 0.01.
INSTRUMENTAL_P95_RMS_THRESHOLD = float(os.environ.get("INSTRUMENTAL_P95_RMS_THRESHOLD", "0.008"))
INSTRUMENTAL_FRAME_RMS_THRESHOLD = float(os.environ.get("INSTRUMENTAL_FRAME_RMS_THRESHOLD", "0.01"))
INSTRUMENTAL_ACTIVE_RATIO_THRESHOLD = float(os.environ.get("INSTRUMENTAL_ACTIVE_RATIO_THRESHOLD", "0.01"))


def assess_vocal_presence(wav_path: str) -> tuple[bool, float, float]:
    """
    Inspect a 16 kHz isolated vocal WAV and return (has_vocals, p95_rms, active_ratio).

    A track is treated as pure instrumental only when BOTH its 95th-percentile
    100ms frame RMS and active-frame ratio fall below conservative thresholds.
    """
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1, dtype=np.float32)
    frame_len = max(1, int(sr * 0.1))
    n_frames = len(data) // frame_len
    if n_frames == 0:
        rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0.0
        return (rms >= INSTRUMENTAL_P95_RMS_THRESHOLD, rms, 1.0 if rms >= INSTRUMENTAL_FRAME_RMS_THRESHOLD else 0.0)
    frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    p95_rms = float(np.percentile(frame_rms, 95))
    active_ratio = float(np.mean(frame_rms >= INSTRUMENTAL_FRAME_RMS_THRESHOLD))
    has_vocals = not (
        p95_rms < INSTRUMENTAL_P95_RMS_THRESHOLD
        and active_ratio < INSTRUMENTAL_ACTIVE_RATIO_THRESHOLD
    )
    return has_vocals, p95_rms, active_ratio


_session = None
_input_name: Optional[str] = None


def is_vocal_separation_enabled() -> bool:
    """Return True unless explicitly disabled via ENABLE_VOCAL_SEPARATION."""
    raw = os.environ.get("ENABLE_VOCAL_SEPARATION", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _find_model_file() -> Optional[Path]:
    for directory in (DEFAULT_MODEL_DIR, LOCAL_MODEL_DIR):
        candidate = directory / MODEL_FILENAME
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
    return None


def ensure_vocal_separator_model(target_dir: Optional[Path] = None) -> Path:
    """Ensure Kim_Vocal_2.onnx is present locally, downloading it if necessary."""
    existing = _find_model_file()
    if existing:
        return existing

    dest_dir = target_dir or (DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else LOCAL_MODEL_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / MODEL_FILENAME
    tmp_path = out_path.with_suffix(".onnx.tmp")

    import urllib.request

    logger.info("Downloading MDX-Net vocal separator model (%s) to %s ...", MODEL_FILENAME, dest_dir)
    try:
        urllib.request.urlretrieve(MODEL_URL, str(tmp_path))
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    logger.info("Vocal separator model ready: %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def _get_session():
    """Lazily initialize and cache the ONNX Runtime session."""
    global _session, _input_name
    if _session is not None:
        return _session, _input_name

    import onnxruntime as ort

    model_path = _find_model_file()
    if not model_path:
        try:
            model_path = ensure_vocal_separator_model()
        except Exception as exc:
            logger.warning("Could not download vocal separator model: %s", exc)
            return None, None

    cpu_threads = max(1, int(os.environ.get("WHISPER_CPU_THREADS", "4")))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = cpu_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("Loading ONNX vocal separator from %s (threads=%d) ...", model_path, cpu_threads)
    _session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def _decode_stereo_44k(audio_path: str) -> np.ndarray:
    """Decode any ffmpeg-supported audio file to float32 stereo (2, N) at 44.1 kHz."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"ffmpeg stereo decode failed: {stderr}")

    raw = np.frombuffer(result.stdout, dtype=np.float32)
    frames = raw.size // 2
    if frames == 0:
        raise ValueError("Decoded audio stream is empty")
    return raw[: frames * 2].reshape(frames, 2).T.copy()


def _stft_chunk(stereo_chunk: np.ndarray) -> np.ndarray:
    """Convert (2, CHUNK_SIZE) stereo float32 chunk to (1, 4, DIM_F, DIM_T) ONNX input."""
    spec_l = librosa.stft(
        stereo_chunk[0],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
    )[:DIM_F, :DIM_T]
    spec_r = librosa.stft(
        stereo_chunk[1],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
    )[:DIM_F, :DIM_T]

    stacked = np.stack(
        [spec_l.real, spec_l.imag, spec_r.real, spec_r.imag],
        axis=0,
    ).astype(np.float32, copy=False)
    return stacked[np.newaxis, :, :, :]


def _istft_chunk(onnx_output: np.ndarray) -> np.ndarray:
    """Convert (1, 4, DIM_F, DIM_T) ONNX output back to (2, CHUNK_SIZE) stereo float32."""
    out = onnx_output[0]
    n_bins = N_FFT // 2 + 1
    full_l = np.zeros((n_bins, DIM_T), dtype=np.complex64)
    full_r = np.zeros((n_bins, DIM_T), dtype=np.complex64)
    full_l[:DIM_F, :] = out[0] + 1j * out[1]
    full_r[:DIM_F, :] = out[2] + 1j * out[3]

    wave_l = librosa.istft(
        full_l,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        length=CHUNK_SIZE,
    )
    wave_r = librosa.istft(
        full_r,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        length=CHUNK_SIZE,
    )
    return np.stack([wave_l, wave_r], axis=0)


def separate_vocals_to_wav(
    audio_path: str,
    output_wav_path: str,
    *,
    job_id: Optional[str] = None,
) -> Optional[str]:
    """
    Separate singing vocals from *audio_path* and write a 16 kHz mono PCM WAV
    to *output_wav_path*. Returns *output_wav_path* on success, or None if
    vocal separation is disabled or fails.
    """
    if not is_vocal_separation_enabled():
        return None

    started = time.monotonic()
    try:
        session, input_name = _get_session()
        if session is None or not input_name:
            return None

        stereo = _decode_stereo_44k(audio_path)
        n_samples = stereo.shape[1]

        pad = GEN_SIZE - (n_samples % GEN_SIZE)
        if pad == GEN_SIZE:
            pad = 0
        padded = np.pad(stereo, ((0, 0), (TRIM, TRIM + pad)), mode="constant")

        vocal_chunks = []
        n_chunks = (n_samples + pad) // GEN_SIZE
        for idx in range(n_chunks):
            start = idx * GEN_SIZE
            chunk = padded[:, start : start + CHUNK_SIZE]
            if chunk.shape[1] < CHUNK_SIZE:
                chunk = np.pad(chunk, ((0, 0), (0, CHUNK_SIZE - chunk.shape[1])), mode="constant")
            inp = _stft_chunk(chunk)
            out = session.run(None, {input_name: inp})[0]
            wave = _istft_chunk(out)
            vocal_chunks.append(wave[:, TRIM : TRIM + GEN_SIZE])

        vocals_stereo = np.concatenate(vocal_chunks, axis=1)[:, :n_samples] * COMPENSATE
        vocals_mono_44k = np.mean(vocals_stereo, axis=0, dtype=np.float32)

        # High-quality polyphase downsampling from 44.1 kHz -> 16 kHz (gcd=100 -> 160/441)
        vocals_16k = resample_poly(vocals_mono_44k, TARGET_SAMPLE_RATE // 100, SAMPLE_RATE // 100)
        vocals_16k = np.clip(vocals_16k, -1.0, 1.0).astype(np.float32, copy=False)

        sf.write(output_wav_path, vocals_16k, TARGET_SAMPLE_RATE, subtype="PCM_16")
        elapsed = time.monotonic() - started
        logger.info(
            "[sync] job=%s stage=vocal_sep_done chunks=%d elapsed=%.1fs file=%s",
            job_id or "-",
            n_chunks,
            elapsed,
            Path(output_wav_path).name,
        )
        return output_wav_path
    except Exception as exc:
        logger.warning(
            "[sync] job=%s stage=vocal_sep_failed error=%s (falling back to non-separated audio)",
            job_id or "-",
            exc,
        )
        return None
