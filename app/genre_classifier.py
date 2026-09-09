"""
Tier 3 Neural Audio Genre Classifier for lyrics-sync.

Uses an ONNX-runtime Discogs-EffNet model (~18MB) trained on 400 music styles.
Runs on CPU without PyTorch in < 200ms.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("/app/models")
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_MODEL_URL = "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.onnx"
_JSON_URL = "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"

_session = None
_classes = None
_input_name = None


def _find_model_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Locate ONNX model and json labels in /app/models or local models/."""
    candidates = [DEFAULT_MODEL_DIR, LOCAL_MODEL_DIR]
    for d in candidates:
        onnx_path = d / "genre_discogs400.onnx"
        json_path = d / "genre_discogs400.json"
        if onnx_path.exists() and json_path.exists():
            return onnx_path, json_path
    return None, None


def ensure_genre_model(target_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Ensure ONNX genre model and json labels are present, downloading if necessary."""
    onnx_path, json_path = _find_model_files()
    if onnx_path and json_path:
        return onnx_path, json_path

    dest_dir = target_dir or (DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else LOCAL_MODEL_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = dest_dir / "genre_discogs400.onnx"
    out_json = dest_dir / "genre_discogs400.json"

    import urllib.request
    logger.info("Downloading Discogs-EffNet ONNX genre model to %s ...", dest_dir)
    if not out_onnx.exists():
        urllib.request.urlretrieve(_MODEL_URL, str(out_onnx))
    if not out_json.exists():
        urllib.request.urlretrieve(_JSON_URL, str(out_json))
    logger.info("ONNX genre model ready in %s.", dest_dir)
    return out_onnx, out_json


def _get_classifier_session():
    """Get or lazily initialize the ONNX inference session."""
    global _session, _classes, _input_name
    if _session is not None:
        return _session, _classes, _input_name

    import onnxruntime as ort

    onnx_path, json_path = _find_model_files()
    if not onnx_path:
        try:
            onnx_path, json_path = ensure_genre_model()
        except Exception as exc:
            logger.warning("Could not download or find genre model: %s", exc)
            return None, None, None

    if not onnx_path:
        logger.warning("Genre classifier ONNX model not found in %s or %s", DEFAULT_MODEL_DIR, LOCAL_MODEL_DIR)
        return None, None, None

    logger.info("Initializing ONNX genre classifier from %s ...", onnx_path)
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        _classes = meta.get("classes", [])

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        logger.info("ONNX genre classifier ready (%d classes, input=%s).", len(_classes), _input_name)
        return _session, _classes, _input_name
    except Exception as exc:
        logger.exception("Failed to initialize ONNX genre classifier: %s", exc)
        return None, None, None


def classify_audio_genre(
    audio_path: str,
    top_k: int = 5,
    offset: float = 15.0,
    duration: float = 30.0,
) -> Dict[str, Any]:
    """
    Classify audio into Discogs music genres using the EffNet ONNX model.

    Returns:
      {
        "top_genres": [
          {"label": "Electronic---Electro", "genre": "Electro", "category": "Electronic", "confidence": 0.241},
          ...
        ],
        "primary_genre": "Electro",
        "inference_time_sec": float
      }
    """
    import librosa

    start_time = time.monotonic()
    session, classes, input_name = _get_classifier_session()
    if session is None or not classes:
        return {
            "top_genres": [],
            "primary_genre": None,
            "inference_time_sec": 0.0,
        }

    try:
        # 1. Load up to 30s of audio at 16,000 Hz
        y, sr = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)
        if len(y) == 0:
            y, sr = librosa.load(audio_path, sr=16000, offset=0.0, duration=duration)

        if len(y) < 16000:  # Less than 1 second
            return {
                "top_genres": [],
                "primary_genre": None,
                "inference_time_sec": round(time.monotonic() - start_time, 3),
            }

        # 2. Compute 96-band mel-spectrogram (Essentia Discogs-EffNet parameters)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=16000,
            n_fft=512,
            hop_length=256,
            n_mels=96,
            window="hann",
            center=False,
            power=2.0,
        )

        # 3. Essentia log-mel compression: log(10000 * mel + 1)
        log_mel = np.log(10000.0 * mel + 1.0).T  # Shape: (time_frames, 96)

        # 4. Extract 128-frame patches with 64-frame hop
        time_frames = log_mel.shape[0]
        patch_size = 128
        patch_hop = 64
        patches = []

        for start_frame in range(0, time_frames - patch_size + 1, patch_hop):
            patches.append(log_mel[start_frame:start_frame + patch_size, :])

        if not patches:
            padded = np.zeros((patch_size, 96), dtype=np.float32)
            padded[:time_frames, :] = log_mel
            patches.append(padded)

        batch = np.array(patches, dtype=np.float32)

        # 5. ONNX forward pass
        infer_start = time.monotonic()
        outputs = session.run(None, {input_name: batch})
        infer_time = time.monotonic() - infer_start

        # 6. Average predictions across temporal patches
        avg_preds = np.mean(outputs[0], axis=0)
        ranked_indices = np.argsort(avg_preds)[::-1][:top_k]

        top_genres = []
        for idx in ranked_indices:
            full_label = classes[idx]
            conf = float(avg_preds[idx])
            
            # Discogs labels format: "Category---Genre" (e.g., "Electronic---Electro")
            if "---" in full_label:
                cat, gen = full_label.split("---", 1)
            else:
                cat, gen = full_label, full_label

            top_genres.append({
                "label": full_label,
                "genre": gen.strip(),
                "category": cat.strip(),
                "confidence": round(conf, 4),
            })

        primary = top_genres[0]["genre"] if top_genres else None
        elapsed = time.monotonic() - start_time

        return {
            "top_genres": top_genres,
            "primary_genre": primary,
            "inference_time_sec": round(elapsed, 3),
        }

    except Exception as exc:
        logger.warning("Neural genre classification failed for %s: %s", audio_path, exc)
        return {
            "top_genres": [],
            "primary_genre": None,
            "inference_time_sec": round(time.monotonic() - start_time, 3),
        }
