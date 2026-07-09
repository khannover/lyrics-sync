"""Unit tests for sync_quality helpers (no Whisper)."""

from tests.sync_quality import (
    check_monotonic_timestamps_ms,
    evaluate_sync_report,
    parse_lrc_timestamps_ms,
)


def test_parse_lrc_timestamps_ms():
    lrc = "[00:01.00]Hello\n[00:05.50]World\n"
    assert parse_lrc_timestamps_ms(lrc) == [1000, 5500]


def test_monotonic_detects_regression():
    ok, err = check_monotonic_timestamps_ms([1000, 2000, 1500])
    assert not ok
    assert "regression" in (err or "")


def test_evaluate_sync_report_good():
    report = {
        "quality": "good",
        "warnings": [],
        "line_count": 4,
        "duration_ms": 120_000,
        "whisper_word_count": 40,
    }
    lrc = "\n".join(
        f"[00:{30 + i * 15:02d}.00]line {i}" for i in range(4)
    )
    v = evaluate_sync_report(report, lrc_text=lrc)
    assert v.ok
    assert v.quality == "good"


def test_evaluate_sync_report_accepts_degraded():
    report = {
        "quality": "degraded",
        "warnings": ["sparse alignment in bridge"],
        "line_count": 4,
        "duration_ms": 120_000,
        "whisper_word_count": 40,
    }
    lrc = "\n".join(
        f"[00:{30 + i * 15:02d}.00]line {i}" for i in range(4)
    )
    v = evaluate_sync_report(report, lrc_text=lrc)
    assert v.ok
    assert v.quality == "degraded"
    assert v.warnings == ["sparse alignment in bridge"]


def test_evaluate_sync_report_rejects_fallback():
    report = {
        "quality": "fallback",
        "warnings": ["x"],
        "line_count": 4,
        "duration_ms": 120_000,
        "whisper_word_count": 0,
    }
    lrc = "[00:10.00]a\n[00:20.00]b\n[00:30.00]c\n"
    v = evaluate_sync_report(report, lrc_text=lrc, allow_fallback=False)
    assert not v.ok
    assert any("fallback" in i for i in v.issues)


def test_evaluate_sync_report_flags_timestamp_regression():
    report = {
        "quality": "good",
        "warnings": [],
        "line_count": 3,
        "duration_ms": 120_000,
        "whisper_word_count": 40,
    }
    lrc = "[00:10.00]a\n[00:30.00]b\n[00:20.00]c\n"
    v = evaluate_sync_report(report, lrc_text=lrc)
    assert not v.ok
    assert any("regression" in i for i in v.issues)


def test_evaluate_sync_report_flags_low_coverage():
    report = {
        "quality": "good",
        "warnings": [],
        "line_count": 4,
        "duration_ms": 300_000,
        "whisper_word_count": 40,
    }
    lrc = "\n".join(f"[00:{i * 5:02d}.00]line {i}" for i in range(4))
    v = evaluate_sync_report(report, lrc_text=lrc, min_coverage_ratio=0.35)
    assert not v.ok
    assert any("coverage" in i for i in v.issues)


def test_wait_for_service_ready_retries_502(monkeypatch):
    from unittest.mock import MagicMock

    import scripts.live_sync_quality_check as live

    calls = {"n": 0}

    def fake_get(url, timeout=10.0):
        calls["n"] += 1
        resp = MagicMock()
        if calls["n"] < 3:
            resp.status_code = 502
            return resp
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(live.httpx, "get", fake_get)
    monkeypatch.setattr(live.time, "sleep", lambda _s: None)

    live._wait_for_service_ready("http://127.0.0.1:8005", timeout_s=30, interval_s=0)
    assert calls["n"] == 3


def test_evaluate_sync_report_legacy_zip_ok_without_whisper_count():
    report = {
        "quality": "good",
        "warnings": [],
        "line_count": 4,
        "duration_ms": 0,
        "whisper_word_count": -1,
        "legacy_zip": True,
    }
    lrc = "\n".join(f"[01:{i * 20:02d}.00]line {i}" for i in range(4))
    v = evaluate_sync_report(report, lrc_text=lrc)
    assert v.ok


def test_evaluate_sync_report_legacy_zip_fails_short_span():
    report = {
        "quality": "good",
        "warnings": [],
        "line_count": 3,
        "duration_ms": 0,
        "whisper_word_count": -1,
        "legacy_zip": True,
    }
    lrc = "[00:10.00]a\n[00:20.00]b\n[00:30.00]c\n"
    v = evaluate_sync_report(report, lrc_text=lrc)
    assert not v.ok
    assert any("legacy ZIP" in i for i in v.issues)


def test_wait_for_service_ready_times_out(monkeypatch):
    from unittest.mock import MagicMock

    import pytest

    import scripts.live_sync_quality_check as live

    def always_502(url, timeout=10.0):
        resp = MagicMock()
        resp.status_code = 502
        return resp

    monkeypatch.setattr(live.httpx, "get", always_502)
    monkeypatch.setattr(live.time, "monotonic", lambda: 0.0)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # Advance past deadline on second sleep.
        monkeypatch.setattr(live.time, "monotonic", lambda: 100.0)

    monkeypatch.setattr(live.time, "sleep", fake_sleep)

    with pytest.raises(SystemExit) as exc:
        live._wait_for_service_ready("http://127.0.0.1:8005", timeout_s=5, interval_s=1)
    assert "not ready" in str(exc.value)