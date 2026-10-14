"""Unit tests for temp work-dir cleanup (no Whisper / alignment)."""

from __future__ import annotations

import os
import time

from app.main import _cleanup_stale_work_items


def _touch(path, age_seconds: float) -> None:
    ts = time.time() - age_seconds
    os.utime(path, (ts, ts))


def test_cleanup_removes_old_job_folder(tmp_path):
    work = tmp_path / "lyric-sync"
    work.mkdir()
    stale = work / "job-old"
    stale.mkdir()
    _touch(stale, age_seconds=7200)

    removed = _cleanup_stale_work_items(
        work,
        now=time.time(),
        max_age_seconds=3600,
        protected_names=set(),
    )

    assert removed == ["job-old"]
    assert not stale.exists()


def test_cleanup_skips_recent_and_protected_items(tmp_path):
    work = tmp_path / "lyric-sync"
    work.mkdir()

    recent = work / "job-fresh"
    recent.mkdir()
    _touch(recent, age_seconds=60)

    active = work / "job-active"
    active.mkdir()
    _touch(active, age_seconds=7200)

    stale_file = work / "orphan.tmp"
    stale_file.write_text("x", encoding="utf-8")
    _touch(stale_file, age_seconds=7200)

    removed = _cleanup_stale_work_items(
        work,
        now=time.time(),
        max_age_seconds=3600,
        protected_names={"job-active"},
    )

    assert removed == ["orphan.tmp"]
    assert recent.exists()
    assert active.exists()
    assert not stale_file.exists()


def test_cleanup_noop_when_work_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert not missing.exists()
    removed = _cleanup_stale_work_items(
        missing,
        now=time.time(),
        max_age_seconds=3600,
        protected_names=set(),
    )
    assert removed == []


def test_cleanup_removes_expired_delivered_jobs_and_prunes_memory(tmp_path, monkeypatch):
    import app.async_jobs as aj

    work = tmp_path / "lyric-sync"
    work.mkdir()
    monkeypatch.setattr(aj, "WORK_DIR", work)

    aj._jobs.clear()
    aj._track_jobs.clear()

    now = time.time()

    # 1. Expired delivered job: folder > 3600s, delivered > 3600s
    expired_id = "job-deliv-expired"
    expired_dir = work / expired_id
    expired_dir.mkdir()
    _touch(expired_dir, age_seconds=7200)
    job_expired = aj.SyncJob(
        job_id=expired_id,
        track_id="track-expired",
        callback_url="https://example.com/cb",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 8000,
        completed_at=now - 7200,
        delivered_at=now - 7200,
        job_dir=expired_dir,
    )

    # 2. Fresh delivered job: folder > 3600s, but delivered 600s ago (< 3600s grace period)
    recent_deliv_id = "job-deliv-recent"
    recent_deliv_dir = work / recent_deliv_id
    recent_deliv_dir.mkdir()
    _touch(recent_deliv_dir, age_seconds=7200)
    job_recent_deliv = aj.SyncJob(
        job_id=recent_deliv_id,
        track_id="track-recent",
        callback_url="https://example.com/cb",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 7200,
        completed_at=now - 600,
        delivered_at=now - 600,
        job_dir=recent_deliv_dir,
    )

    # 3. Active processing job: folder > 3600s, but status="processing"
    active_id = "job-active"
    active_dir = work / active_id
    active_dir.mkdir()
    _touch(active_dir, age_seconds=7200)
    job_active = aj.SyncJob(
        job_id=active_id,
        track_id="track-active",
        callback_url="https://example.com/cb",
        manual=False,
        status="processing",
        created_at=now - 7200,
        job_dir=active_dir,
    )

    # 4. Pending retry job: folder > 3600s, status="completed", callback_status="pending", within timeout
    pending_id = "job-pending"
    pending_dir = work / pending_id
    pending_dir.mkdir()
    _touch(pending_dir, age_seconds=7200)
    job_pending = aj.SyncJob(
        job_id=pending_id,
        track_id="track-pending",
        callback_url="https://example.com/cb",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_PENDING,
        created_at=now - 7200,
        completed_at=now - 3600,
        job_dir=pending_dir,
    )

    aj._jobs[expired_id] = job_expired
    aj._track_jobs["track-expired"] = expired_id

    aj._jobs[recent_deliv_id] = job_recent_deliv
    aj._track_jobs["track-recent"] = recent_deliv_id

    aj._jobs[active_id] = job_active
    aj._track_jobs["track-active"] = active_id

    aj._jobs[pending_id] = job_pending
    aj._track_jobs["track-pending"] = pending_id

    # Verify protected names
    protected = aj.cleanup_protected_names(now=now)
    assert expired_id not in protected
    assert recent_deliv_id in protected
    assert active_id in protected
    assert pending_id in protected

    # Run cleanup
    removed = _cleanup_stale_work_items(
        work,
        now=now,
        max_age_seconds=3600,
        protected_names=protected,
    )

    assert removed == [expired_id]
    assert not expired_dir.exists()
    assert recent_deliv_dir.exists()
    assert active_dir.exists()
    assert pending_dir.exists()

    # Call prune_jobs with removed
    pruned = aj.prune_jobs(removed)
    assert pruned == 1
    assert expired_id not in aj._jobs
    assert "track-expired" not in aj._track_jobs
    assert recent_deliv_id in aj._jobs
    assert active_id in aj._jobs
    assert pending_id in aj._jobs

    aj._jobs.clear()
    aj._track_jobs.clear()