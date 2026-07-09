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