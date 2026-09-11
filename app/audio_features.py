"""
Tier 2 Acoustic Signal Analysis for lyrics-sync.

Extracts musical features directly from audio using librosa:
- BPM (tempo)
- Energy level (low, medium, high) based on RMS loudness and spectral centroid
- Musical Key & Scale (e.g., "A minor", "C major") via chroma analysis
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles for 12 pitch classes
_PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_MAJOR_NORM = (_MAJOR_PROFILE - _MAJOR_PROFILE.mean()) / _MAJOR_PROFILE.std()
_MINOR_NORM = (_MINOR_PROFILE - _MINOR_PROFILE.mean()) / _MINOR_PROFILE.std()


def estimate_musical_key(chroma_mean: np.ndarray) -> Tuple[str, float]:
    """
    Estimate musical key (e.g., 'A minor', 'C major') from 12-bin mean chroma.
    Uses Pearson correlation against Krumhansl-Schmuckler tonality profiles.
    """
    chroma_std = chroma_mean.std()
    if chroma_std > 1e-6:
        chroma_norm = (chroma_mean - chroma_mean.mean()) / chroma_std
    else:
        chroma_norm = chroma_mean

    best_key = "Unknown"
    best_corr = -2.0

    for i in range(12):
        # Major correlation
        corr_maj = float(np.corrcoef(chroma_norm, np.roll(_MAJOR_NORM, i))[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{_PITCH_NAMES[i]} major"

        # Minor correlation
        corr_min = float(np.corrcoef(chroma_norm, np.roll(_MINOR_NORM, i))[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{_PITCH_NAMES[i]} minor"

    return best_key, round(float(best_corr), 3)


def extract_audio_features(
    audio_path: str,
    offset: float = 15.0,
    duration: float = 30.0,
) -> Dict[str, Any]:
    """
    Extract acoustic signal features from an audio file.

    Returns:
      {
        "bpm": int | None,
        "energy": "low" | "medium" | "high",
        "key": str | None,
        "rms": float,
        "spectral_centroid": float,
        "analysis_time_sec": float
      }
    """
    import librosa

    start_time = time.monotonic()
    try:
        # Load up to 30s of audio starting at offset=15s to bypass silent intros
        y, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=duration)
        if len(y) == 0:
            # Fallback to loading from start if offset exceeded duration
            y, sr = librosa.load(audio_path, sr=22050, offset=0.0, duration=duration)

        if len(y) == 0:
            return {
                "bpm": None,
                "energy": "low",
                "key": None,
                "rms": 0.0,
                "spectral_centroid": 0.0,
                "analysis_time_sec": round(time.monotonic() - start_time, 2),
            }

        # 1. BPM / Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        bpm_val = int(round(float(np.atleast_1d(tempo)[0])))
        if bpm_val <= 0 or bpm_val > 300:
            bpm_val = None

        # 2. Energy / Loudness & Brightness
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_cent = float(np.mean(cent))

        if mean_rms > 0.16 or mean_cent > 3200:
            energy_val = "high"
        elif mean_rms > 0.07 or mean_cent > 1800:
            energy_val = "medium"
        else:
            energy_val = "low"

        # 3. Fast Key Detection using chroma_stft (takes ~0.1s on CPU)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_val, key_confidence = estimate_musical_key(chroma_mean)

        elapsed = time.monotonic() - start_time
        return {
            "bpm": bpm_val,
            "energy": energy_val,
            "key": key_val,
            "rms": round(mean_rms, 4),
            "spectral_centroid": round(mean_cent, 1),
            "analysis_time_sec": round(elapsed, 2),
        }

    except Exception as exc:
        logger.warning("Audio feature extraction failed for %s: %s", audio_path, exc)
        return {
            "bpm": None,
            "energy": "medium",
            "key": None,
            "rms": 0.0,
            "spectral_centroid": 0.0,
            "analysis_time_sec": round(time.monotonic() - start_time, 2),
        }
