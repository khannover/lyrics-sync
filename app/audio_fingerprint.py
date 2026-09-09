"""
Chromaprint / AcoustID Audio Copyright Fingerprint Generator for lyrics-sync.

Calls the `fpcalc` binary to produce compact acoustic fingerprints for matching
against commercial databases (AcoustID, MusicBrainz) in < 100ms.
"""

from __future__ import annotations

import json
import shutil
import logging
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_audio_fingerprint(
    audio_path: str,
    timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Generate an AcoustID / Chromaprint acoustic fingerprint from an audio file.

    Parameters:
      audio_path: Path to MP3 or audio file.
      timeout_sec: Maximum execution timeout for fpcalc.

    Returns:
      {
        "acoustid_fingerprint": str | None,
        "duration_sec": float | None,
        "algorithm": "chromaprint"
      }
    """
    fpcalc_bin = shutil.which("fpcalc")
    if not fpcalc_bin:
        logger.warning("fpcalc binary not found in system PATH. Install libchromaprint-tools.")
        return {
            "acoustid_fingerprint": None,
            "duration_sec": None,
            "algorithm": "chromaprint",
        }

    try:
        proc = subprocess.run(
            [fpcalc_bin, "-json", audio_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )

        if proc.returncode != 0:
            logger.warning("fpcalc failed on %s (code %d): %s", audio_path, proc.returncode, proc.stderr.strip())
            return {
                "acoustid_fingerprint": None,
                "duration_sec": None,
                "algorithm": "chromaprint",
            }

        data = json.loads(proc.stdout)
        fingerprint = data.get("fingerprint")
        duration = data.get("duration")

        return {
            "acoustid_fingerprint": fingerprint,
            "duration_sec": round(float(duration), 2) if duration is not None else None,
            "algorithm": "chromaprint",
        }

    except subprocess.TimeoutExpired:
        logger.warning("fpcalc timed out after %s seconds on %s", timeout_sec, audio_path)
        return {
            "acoustid_fingerprint": None,
            "duration_sec": None,
            "algorithm": "chromaprint",
        }
    except Exception as exc:
        logger.warning("Failed to generate fingerprint for %s: %s", audio_path, exc)
        return {
            "acoustid_fingerprint": None,
            "duration_sec": None,
            "algorithm": "chromaprint",
        }
