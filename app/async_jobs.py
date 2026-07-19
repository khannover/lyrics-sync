"""Async lyric-sync job queue with webhook callbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
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
CALLBACK_PENDING = "pending"
CALLBACK_DELIVERED = "delivered"
CALLBACK_ACKED = "acked"
ACK_RETENTION_SECONDS = float(os.environ.get("LYRIC_SYNC_ACK_RETENTION_SECONDS", "86400"))


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
    callback_status: str = CALLBACK_PENDING
    callback_attempts: int = 0
    callback_last_error: Optional[str] = None
    acked_at: Optional[float] = None


_jobs: dict[str, SyncJob] = {}
_track_jobs: dict[str, str] = {}
_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None
_callback_recovery_task: Optional[asyncio.Task] = None


def _save_job(job: SyncJob) -> None:
    """Persist job state atomically on the mounted work volume."""
    if not job.job_dir:
        raise RuntimeError("Job directory missing")
    job.job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job.job_id,
        "track_id": job.track_id,
        "callback_url": job.callback_url,
        "manual": job.manual,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "quality": job.quality,
        "warnings": job.warnings,
        "report": job.report,
        "lyrics_lrc": job.lyrics_lrc,
        "callback_status": job.callback_status,
        "callback_attempts": job.callback_attempts,
        "callback_last_error": job.callback_last_error,
        "acked_at": job.acked_at,
    }
    path = job.job_dir / "job.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_persisted_jobs() -> int:
    """Restore durable jobs after a container/process restart."""
    restored = 0
    if not WORK_DIR.exists():
        return restored
    for metadata in WORK_DIR.glob("*/job.json"):
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            job = SyncJob(
                job_id=str(data["job_id"]),
                track_id=str(data["track_id"]),
                callback_url=str(data.get("callback_url") or ""),
                manual=bool(data.get("manual")),
                status=str(data.get("status") or "queued"),
                created_at=float(data.get("created_at") or time.time()),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                quality=data.get("quality"),
                warnings=list(data.get("warnings") or []),
                report=data.get("report"),
                lyrics_lrc=data.get("lyrics_lrc"),
                job_dir=metadata.parent,
                callback_status=str(data.get("callback_status") or CALLBACK_PENDING),
                callback_attempts=int(data.get("callback_attempts") or 0),
                callback_last_error=data.get("callback_last_error"),
                acked_at=data.get("acked_at"),
            )
            if job.status not in JOB_STATUSES:
                continue
            if job.status == "processing":
                job.status = "queued"
                job.started_at = None
                _save_job(job)
            _jobs[job.job_id] = job
            _track_jobs[job.track_id] = job.job_id
            if job.status in {"queued", "processing"}:
                from app.main import _register_job

                _register_job(job.job_id)
            if job.status == "queued":
                _queue.put_nowait(job.job_id)
            restored += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("[async] could not restore job metadata %s: %s", metadata, exc)
    if restored:
        logger.info("[async] restored %s persisted job(s)", restored)
    return restored


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
        "lyrics_lrc": job.lyrics_lrc,
        "has_lyrics_lrc": bool((job.lyrics_lrc or "").strip()),
        "callback_status": job.callback_status,
        "callback_attempts": job.callback_attempts,
        "callback_last_error": job.callback_last_error,
        "acked_at": job.acked_at,
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
    _save_job(job)
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
    global _worker_task, _callback_recovery_task
    if _worker_task and not _worker_task.done():
        return
    load_persisted_jobs()
    _worker_task = asyncio.create_task(_worker_loop())
    _callback_recovery_task = asyncio.create_task(_callback_recovery_loop())
    logger.info("[async] job worker started")


async def stop_worker() -> None:
    global _worker_task, _callback_recovery_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    _worker_task = None
    if _callback_recovery_task and not _callback_recovery_task.done():
        _callback_recovery_task.cancel()
        await asyncio.gather(_callback_recovery_task, return_exceptions=True)
    _callback_recovery_task = None
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


async def _callback_recovery_loop() -> None:
    while True:
        try:
            await asyncio.sleep(30)
            for job in list(_jobs.values()):
                if job.status in {"completed", "failed"} and job.callback_status == CALLBACK_PENDING:
                    if await _deliver_callback(job):
                        _finish_job(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[async] callback recovery loop failed")


async def run_job(job: SyncJob) -> None:
    """Align lyrics for a queued job and deliver the callback."""
    if not job.job_dir:
        raise RuntimeError("Job directory missing")

    job.status = "processing"
    job.started_at = time.time()
    _save_job(job)
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
    _save_job(job)
    logger.info(
        "[async] job=%s track=%s status=completed quality=%s",
        job.job_id,
        job.track_id,
        job.quality,
    )
    if await _deliver_callback(job):
        _finish_job(job)
    else:
        from app.main import _unregister_job

        _unregister_job(job.job_id)


async def mark_job_failed(job: SyncJob, error: str) -> None:
    job.status = "failed"
    job.error = error
    job.completed_at = time.time()
    _save_job(job)
    logger.warning(
        "[async] job=%s track=%s status=failed error=%s",
        job.job_id,
        job.track_id,
        error,
    )
    if await _deliver_callback(job):
        _finish_job(job)
    else:
        from app.main import _unregister_job

        _unregister_job(job.job_id)


def _finish_job(job: SyncJob) -> None:
    from app.main import _unregister_job  # lazy

    _unregister_job(job.job_id)


async def _deliver_callback(job: SyncJob) -> bool:
    if not job.callback_url:
        logger.warning("[async] job=%s missing callback_url — skipping webhook", job.job_id)
        job.callback_status = CALLBACK_DELIVERED
        _save_job(job)
        return True

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
        job.callback_attempts += 1
        _save_job(job)
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
                    job.callback_status = CALLBACK_DELIVERED
                    job.callback_last_error = None
                    _save_job(job)
                    return True
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)

        job.callback_last_error = last_error
        _save_job(job)
        logger.warning(
            "[async] job=%s callback_failed attempt=%s/%s error=%s",
            job.job_id,
            attempt,
            CALLBACK_RETRIES,
            last_error,
        )
        if attempt < CALLBACK_RETRIES:
            await asyncio.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0.0, 1.0))

    logger.error(
        "[async] job=%s callback_exhausted track=%s error=%s",
        job.job_id,
        job.track_id,
        last_error,
    )
    job.callback_status = CALLBACK_PENDING
    job.callback_last_error = last_error
    _save_job(job)
    return False


def cleanup_protected_names(now: Optional[float] = None) -> set[str]:
    now = now or time.time()
    protected = set()
    for job in _jobs.values():
        if job.callback_status != CALLBACK_ACKED or not job.acked_at or (now - job.acked_at) < ACK_RETENTION_SECONDS:
            protected.add(job.job_id)
    return protected


def ack_job(job_id: str) -> Optional[dict[str, Any]]:
    job = _jobs.get(job_id)
    if not job:
        return None
    job.callback_status = CALLBACK_ACKED
    job.acked_at = time.time()
    _save_job(job)
    return _job_snapshot(job)