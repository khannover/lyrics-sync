"""Async lyric-sync job queue with webhook callbacks."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from app.sylt_writer import format_lrc

logger = logging.getLogger(__name__)

WORK_DIR = Path("/tmp/lyric-sync")
CALLBACK_SECRET = os.environ.get("LYRIC_SYNC_CALLBACK_SECRET", "").strip()
CALLBACK_RETRIES = max(1, int(os.environ.get("LYRIC_SYNC_CALLBACK_RETRIES", "5")))
CALLBACK_TIMEOUT_SEC = float(os.environ.get("LYRIC_SYNC_CALLBACK_TIMEOUT_SEC", "30"))

JOB_STATUSES = frozenset({"queued", "processing", "completed", "failed"})


@dataclass
class SyncJob:
    job_id: str
    track_id: str
    callback_url: str
    manual: bool
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    quality: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    report: Optional[dict] = None
    lyrics_lrc: Optional[str] = None
    job_dir: Optional[Path] = None


_jobs: dict[str, SyncJob] = {}
_track_jobs: dict[str, str] = {}
_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None


def _job_snapshot(job: SyncJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "status": job.status,
        "manual": job.manual,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "quality": job.quality,
        "warnings": list(job.warnings),
        "report": job.report,
        "has_lyrics_lrc": bool((job.lyrics_lrc or "").strip()),
    }


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    job = _jobs.get(job_id)
    if not job:
        return None
    return _job_snapshot(job)


def queue_stats() -> dict[str, int]:
    counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
    for job in _jobs.values():
        if job.status in counts:
            counts[job.status] += 1
    return counts


def create_job(
    *,
    track_id: str,
    callback_url: str,
    manual: bool,
    mp3_path: Path,
    lyrics_path: Path,
) -> SyncJob:
    track_id = (track_id or "").strip()
    if not track_id:
        raise ValueError("track_id is required")

    existing_id = _track_jobs.get(track_id)
    if existing_id:
        existing = _jobs.get(existing_id)
        if existing and existing.status in {"queued", "processing"}:
            return existing

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    dest_mp3 = job_dir / "input.mp3"
    dest_lyrics = job_dir / "lyrics.txt"
    shutil.copy2(mp3_path, dest_mp3)
    shutil.copy2(lyrics_path, dest_lyrics)

    job = SyncJob(
        job_id=job_id,
        track_id=track_id,
        callback_url=callback_url.strip(),
        manual=manual,
        job_dir=job_dir,
    )
    from app.main import _register_job  # lazy — keeps cleanup aware of active jobs

    _register_job(job_id)
    _jobs[job_id] = job
    _track_jobs[track_id] = job_id
    _queue.put_nowait(job_id)
    logger.info(
        "[async] job=%s track=%s status=queued manual=%s callback=%s",
        job_id,
        track_id,
        manual,
        job.callback_url,
    )
    return job


async def start_worker() -> None:
    global _worker_task
    if _worker_task and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_worker_loop())
    logger.info("[async] job worker started")


async def stop_worker() -> None:
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    _worker_task = None
    logger.info("[async] job worker stopped")


async def _worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        job = _jobs.get(job_id)
        if not job or job.status != "queued":
            _queue.task_done()
            continue
        try:
            await run_job(job)
        except Exception as exc:
            logger.exception("[async] job=%s worker_failed", job_id)
            await mark_job_failed(job, str(exc))
        finally:
            _queue.task_done()


async def run_job(job: SyncJob) -> None:
    """Align lyrics for a queued job and deliver the callback."""
    if not job.job_dir:
        raise RuntimeError("Job directory missing")

    job.status = "processing"
    job.started_at = time.time()
    logger.info("[async] job=%s track=%s status=processing", job.job_id, job.track_id)

    from app.main import _ensure_taggable_mp3, _run_alignment  # lazy — avoids import cycle

    mp3_path = job.job_dir / "input.mp3"
    mp3_path = _ensure_taggable_mp3(mp3_path, job.job_dir)
    lyrics_path = job.job_dir / "lyrics.txt"
    lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
    if not lyrics_text:
        raise ValueError("Lyrics file is empty")

    result = await _run_alignment(job.job_id, mp3_path, lyrics_text, job.job_dir)

    job.lyrics_lrc = format_lrc(result.lines)
    job.quality = result.quality
    job.warnings = list(result.warnings)
    job.report = result.report
    job.status = "completed"
    job.completed_at = time.time()
    logger.info(
        "[async] job=%s track=%s status=completed quality=%s",
        job.job_id,
        job.track_id,
        job.quality,
    )
    await _deliver_callback(job)
    _finish_job(job)


async def mark_job_failed(job: SyncJob, error: str) -> None:
    job.status = "failed"
    job.error = error
    job.completed_at = time.time()
    logger.warning(
        "[async] job=%s track=%s status=failed error=%s",
        job.job_id,
        job.track_id,
        error,
    )
    await _deliver_callback(job)
    _finish_job(job)


def _finish_job(job: SyncJob) -> None:
    from app.main import _unregister_job  # lazy

    _unregister_job(job.job_id)


async def _deliver_callback(job: SyncJob) -> None:
    if not job.callback_url:
        logger.warning("[async] job=%s missing callback_url — skipping webhook", job.job_id)
        return

    payload = {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "status": "completed" if job.status == "completed" else "failed",
        "manual": job.manual,
        "lyrics_lrc": job.lyrics_lrc if job.status == "completed" else None,
        "quality": job.quality or "unknown",
        "warnings": job.warnings,
        "report": job.report,
        "error": job.error,
    }

    headers = {"Content-Type": "application/json"}
    if CALLBACK_SECRET:
        headers["X-Lyrics-Sync-Token"] = CALLBACK_SECRET

    timeout = httpx.Timeout(CALLBACK_TIMEOUT_SEC)
    last_error = "unknown"

    for attempt in range(1, CALLBACK_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(job.callback_url, json=payload, headers=headers)
                if 200 <= response.status_code < 300:
                    logger.info(
                        "[async] job=%s callback_delivered attempt=%s status=%s",
                        job.job_id,
                        attempt,
                        response.status_code,
                    )
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)

        logger.warning(
            "[async] job=%s callback_failed attempt=%s/%s error=%s",
            job.job_id,
            attempt,
            CALLBACK_RETRIES,
            last_error,
        )
        if attempt < CALLBACK_RETRIES:
            await asyncio.sleep(min(30.0, 2.0 ** attempt))

    logger.error(
        "[async] job=%s callback_exhausted track=%s error=%s",
        job.job_id,
        job.track_id,
        last_error,
    )