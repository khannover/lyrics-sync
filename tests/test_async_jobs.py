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