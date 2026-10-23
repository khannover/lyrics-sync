"""Unit tests for app.async_jobs (no Whisper / alignment)."""

from __future__ import annotations

import pytest

import app.async_jobs as aj
import app.main as main


@pytest.fixture(autouse=True)
def _isolated_async_job_state(monkeypatch, tmp_path):
    """Keep in-memory job state and work dir local to each test."""
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "lyric-sync")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    aj._jobs.clear()
    aj._track_jobs.clear()
    while not aj._queue.empty():
        aj._queue.get_nowait()

    main.active_job_ids.clear()
    yield

    aj._jobs.clear()
    aj._track_jobs.clear()
    while not aj._queue.empty():
        aj._queue.get_nowait()
    main.active_job_ids.clear()


def _write_inputs(tmp_path, mp3_name: str = "song.mp3", lyrics: str = "line one\nline two"):
    mp3_path = tmp_path / mp3_name
    mp3_path.write_bytes(b"not-a-real-mp3-but-enough-for-copy")
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text(lyrics, encoding="utf-8")
    return mp3_path, lyrics_path


def test_create_job_requires_track_id(tmp_path):
    mp3_path, lyrics_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="track_id"):
        aj.create_job(
            track_id="   ",
            callback_url="https://example.com/hook",
            manual=False,
            mp3_path=mp3_path,
            lyrics_path=lyrics_path,
        )


def test_create_job_queues_and_registers_files(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path, lyrics_path = _write_inputs(tmp_path)
    job = aj.create_job(
        track_id="track-42",
        callback_url="https://example.com/callback",
        manual=True,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )

    assert job.status == "queued"
    assert job.track_id == "track-42"
    assert job.manual is True
    assert job.job_dir is not None
    assert (job.job_dir / "input.mp3").read_bytes() == mp3_path.read_bytes()
    assert (job.job_dir / "lyrics.txt").read_text(encoding="utf-8") == "line one\nline two"
    assert job.job_id in main.active_job_ids

    snapshot = aj.get_job(job.job_id)
    assert snapshot is not None
    assert snapshot["status"] == "queued"
    assert snapshot["track_id"] == "track-42"
    assert snapshot["manual"] is True
    assert aj._queue.qsize() == 1


def test_create_job_idempotent_while_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path, lyrics_path = _write_inputs(tmp_path)
    first = aj.create_job(
        track_id="dup-track",
        callback_url="https://example.com/a",
        manual=False,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )
    second = aj.create_job(
        track_id="dup-track",
        callback_url="https://example.com/b",
        manual=False,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )

    assert second.job_id == first.job_id
    assert len(aj._jobs) == 1
    assert aj._queue.qsize() == 1


def test_persisted_queued_job_is_restored_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)
    mp3_path, lyrics_path = _write_inputs(tmp_path)
    first = aj.create_job(
        track_id="restart-track",
        callback_url="https://example.com/callback",
        manual=False,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )
    assert (first.job_dir / "job.json").exists()

    aj._jobs.clear()
    aj._track_jobs.clear()
    while not aj._queue.empty():
        aj._queue.get_nowait()
    restored = aj.load_persisted_jobs()

    assert restored == 1
    snapshot = aj.get_job(first.job_id)
    assert snapshot["status"] == "queued"
    assert snapshot["track_id"] == "restart-track"
    assert aj._queue.qsize() == 1


def test_queue_stats_reflects_job_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path, lyrics_path = _write_inputs(tmp_path)
    job = aj.create_job(
        track_id="stats-track",
        callback_url="https://example.com/cb",
        manual=False,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )
    assert aj.queue_stats() == {
        "queued": 1,
        "processing": 0,
        "completed": 0,
        "failed": 0,
    }

    job.status = "processing"
    assert aj.queue_stats()["processing"] == 1
    assert aj.queue_stats()["queued"] == 0


def test_get_job_unknown_returns_none():
    assert aj.get_job("00000000-0000-0000-0000-000000000000") is None


def test_cleanup_protected_names_in_flight_and_pending_remain_protected():
    now = 100000.0
    job_queued = aj.SyncJob(
        job_id="job-queued",
        track_id="t-queued",
        callback_url="",
        manual=False,
        status="queued",
        created_at=now - 50000,
    )
    job_processing = aj.SyncJob(
        job_id="job-proc",
        track_id="t-proc",
        callback_url="",
        manual=False,
        status="processing",
        created_at=now - 50000,
    )
    job_pending_fresh = aj.SyncJob(
        job_id="job-pending-fresh",
        track_id="t-pending-fresh",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_PENDING,
        created_at=now - 1000,
        completed_at=now - 500,
    )
    job_pending_timed_out = aj.SyncJob(
        job_id="job-pending-timed-out",
        track_id="t-pending-timed-out",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_PENDING,
        created_at=now - 100000,
        completed_at=now - 90000,
    )

    aj._jobs[job_queued.job_id] = job_queued
    aj._jobs[job_processing.job_id] = job_processing
    aj._jobs[job_pending_fresh.job_id] = job_pending_fresh
    aj._jobs[job_pending_timed_out.job_id] = job_pending_timed_out

    protected = aj.cleanup_protected_names(now=now)
    assert "job-queued" in protected
    assert "job-proc" in protected
    assert "job-pending-fresh" in protected
    assert "job-pending-timed-out" not in protected


def test_cleanup_protected_names_delivered_retention():
    now = 100000.0
    job_delivered_fresh = aj.SyncJob(
        job_id="job-deliv-fresh",
        track_id="t-deliv-fresh",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 2000,
        completed_at=now - 1800,
        delivered_at=now - 1800,  # 30 min ago (< 3600s grace period)
    )
    job_delivered_old = aj.SyncJob(
        job_id="job-deliv-old",
        track_id="t-deliv-old",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 10000,
        completed_at=now - 7200,
        delivered_at=now - 7200,  # 2 hours ago (> 3600s grace period)
    )
    job_delivered_legacy_old = aj.SyncJob(
        job_id="job-deliv-legacy-old",
        track_id="t-deliv-legacy-old",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 10000,
        completed_at=now - 5000,  # > 3600s
        delivered_at=None,
    )
    job_delivered_legacy_fresh = aj.SyncJob(
        job_id="job-deliv-legacy-fresh",
        track_id="t-deliv-legacy-fresh",
        callback_url="https://example.com/hook",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_DELIVERED,
        created_at=now - 1000,
        completed_at=now - 500,  # < 3600s
        delivered_at=None,
    )

    aj._jobs[job_delivered_fresh.job_id] = job_delivered_fresh
    aj._jobs[job_delivered_old.job_id] = job_delivered_old
    aj._jobs[job_delivered_legacy_old.job_id] = job_delivered_legacy_old
    aj._jobs[job_delivered_legacy_fresh.job_id] = job_delivered_legacy_fresh

    protected = aj.cleanup_protected_names(now=now)
    assert "job-deliv-fresh" in protected
    assert "job-deliv-legacy-fresh" in protected
    assert "job-deliv-old" not in protected
    assert "job-deliv-legacy-old" not in protected


def test_cleanup_protected_names_acked_retention():
    now = 100000.0
    job_acked_fresh = aj.SyncJob(
        job_id="job-acked-fresh",
        track_id="t-acked-fresh",
        callback_url="",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_ACKED,
        acked_at=now - 3600,
    )
    job_acked_old = aj.SyncJob(
        job_id="job-acked-old",
        track_id="t-acked-old",
        callback_url="",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_ACKED,
        acked_at=now - 90000,
    )
    job_acked_no_ts = aj.SyncJob(
        job_id="job-acked-no-ts",
        track_id="t-acked-no-ts",
        callback_url="",
        manual=False,
        status="completed",
        callback_status=aj.CALLBACK_ACKED,
        acked_at=None,
    )

    aj._jobs[job_acked_fresh.job_id] = job_acked_fresh
    aj._jobs[job_acked_old.job_id] = job_acked_old
    aj._jobs[job_acked_no_ts.job_id] = job_acked_no_ts

    protected = aj.cleanup_protected_names(now=now)
    assert "job-acked-fresh" in protected
    assert "job-acked-old" not in protected
    assert "job-acked-no-ts" not in protected


def test_prune_jobs_and_evict_job():
    job1 = aj.SyncJob(
        job_id="j1",
        track_id="t1",
        callback_url="",
        manual=False,
    )
    job2 = aj.SyncJob(
        job_id="j2",
        track_id="t2",
        callback_url="",
        manual=False,
    )
    aj._jobs["j1"] = job1
    aj._jobs["j2"] = job2
    aj._track_jobs["t1"] = "j1"
    aj._track_jobs["t2"] = "j2"

    assert len(aj._jobs) == 2
    assert len(aj._track_jobs) == 2

    # Prune non-existent does not fail
    assert aj.prune_jobs(["ghost", "orphan.tmp"]) == 0

    # Prune j1
    assert aj.prune_jobs(["j1"]) == 1
    assert "j1" not in aj._jobs
    assert "t1" not in aj._track_jobs
    assert "j2" in aj._jobs
    assert aj._track_jobs["t2"] == "j2"

    # Evict j2
    assert aj.evict_job("j2") is True
    assert "j2" not in aj._jobs
    assert "t2" not in aj._track_jobs
    assert aj.evict_job("j2") is False


@pytest.mark.asyncio
async def test_deliver_callback_retries_with_backoff_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(aj, "CALLBACK_RETRIES", 3)
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    sleep_calls = []

    async def _fast_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(aj.asyncio, "sleep", _fast_sleep)

    attempts = []

    class _MockResponse:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            attempts.append((url, json, headers))
            if len(attempts) < 3:
                return _MockResponse(502, "bad gateway")
            return _MockResponse(200, "ok")

    monkeypatch.setattr(aj.httpx, "AsyncClient", _MockAsyncClient)

    mp3_path, lyrics_path = _write_inputs(tmp_path)
    job = aj.create_job(
        track_id="retry-track",
        callback_url="https://example.com/callback",
        manual=False,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )
    job.status = "completed"
    job.lyrics_lrc = "[00:01.00]line one"
    job.quality = "good"

    ok = await aj._deliver_callback(job)
    assert ok is True
    assert len(attempts) == 3
    assert len(sleep_calls) == 2
    assert job.callback_status == aj.CALLBACK_DELIVERED
    assert job.callback_attempts == 3
    assert job.delivered_at is not None


def test_ack_job_persists_state_across_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "WORK_DIR", tmp_path / "work")
    aj.WORK_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path, lyrics_path = _write_inputs(tmp_path)
    job = aj.create_job(
        track_id="ack-track",
        callback_url="https://example.com/callback",
        manual=True,
        mp3_path=mp3_path,
        lyrics_path=lyrics_path,
    )
    job.status = "completed"
    job.lyrics_lrc = "[00:02.00]acked line"
    job.quality = "good"
    aj._save_job(job)

    acked = aj.ack_job(job.job_id)
    assert acked is not None
    assert acked["callback_status"] == aj.CALLBACK_ACKED
    assert acked["acked_at"] is not None

    # Simulate service restart
    aj._jobs.clear()
    aj._track_jobs.clear()
    restored = aj.load_persisted_jobs()
    assert restored == 1

    reloaded = aj.get_job(job.job_id)
    assert reloaded is not None
    assert reloaded["callback_status"] == aj.CALLBACK_ACKED
    assert reloaded["lyrics_lrc"] == "[00:02.00]acked line"
