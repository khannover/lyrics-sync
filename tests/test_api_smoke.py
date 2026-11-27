"""Lightweight FastAPI smoke tests (no Whisper / alignment)."""

import io
import json
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

import app.async_jobs as async_jobs
import app.main as main
from app.main import app, limiter


@pytest.fixture(autouse=True)
def _clean_test_env(monkeypatch):
    monkeypatch.delenv("LYRIC_SYNC_CALLBACK_SECRET", raising=False)


def test_health_returns_disk_stats():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == app.version
    assert body["max_concurrent_jobs"] == main.MAX_CONCURRENT_JOBS
    alignment = body["alignment"]
    assert alignment["waiting_jobs"] >= 0
    assert alignment["active_jobs"] >= 0
    assert alignment["total_slots"] == main.MAX_CONCURRENT_JOBS
    async_jobs = body["async_jobs"]
    assert set(async_jobs.keys()) == {"queued", "processing", "completed", "failed"}
    disk = body["disk"]
    for key in ("total_gb", "used_gb", "free_gb"):
        assert key in disk
        assert isinstance(disk[key], (int, float))


def test_health_disk_gb_rounded_to_two_decimals():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    disk = response.json()["disk"]
    for key in ("total_gb", "used_gb", "free_gb"):
        value = disk[key]
        assert round(value, 2) == value, f"{key}={value!r} not rounded to 2 decimals"


def test_queue_returns_semaphore_and_async_stats():
    with TestClient(app) as client:
        response = client.get("/queue")
    assert response.status_code == 200
    body = response.json()
    assert body["waiting_jobs"] >= 0
    assert body["total_slots"] >= 1
    assert body["active_jobs"] >= 0
    async_jobs = body["async_jobs"]
    assert set(async_jobs.keys()) == {"queued", "processing", "completed", "failed"}


def test_sync_returns_429_when_hourly_rate_limit_exceeded():
    """slowapi counts rejected validation requests; 61st POST /sync should 429."""
    limiter.reset()
    payload = {
        "mp3": ("track.flac", b"fake-audio", "audio/flac"),
        "lyrics": ("lyrics.txt", b"hello", "text/plain"),
    }
    try:
        with TestClient(app) as client:
            for _ in range(60):
                response = client.post("/sync", files=payload)
                assert response.status_code == 400
            response = client.post("/sync", files=payload)
        assert response.status_code == 429
        assert "rate" in response.text.lower() or response.json().get("error")
    finally:
        limiter.reset()


def test_sync_mp3_only_returns_429_when_hourly_rate_limit_exceeded():
    """slowapi counts rejected validation requests; 61st POST /sync/mp3-only should 429."""
    limiter.reset()
    payload = {
        "mp3": ("track.flac", b"fake-audio", "audio/flac"),
        "lyrics": ("lyrics.txt", b"hello", "text/plain"),
    }
    try:
        with TestClient(app) as client:
            for _ in range(60):
                response = client.post("/sync/mp3-only", files=payload)
                assert response.status_code == 400
            response = client.post("/sync/mp3-only", files=payload)
        assert response.status_code == 429
        assert "rate" in response.text.lower() or response.json().get("error")
    finally:
        limiter.reset()


def test_enqueue_sync_job_returns_429_when_hourly_rate_limit_exceeded(monkeypatch):
    """slowapi counts rejected validation requests; 61st POST /sync/jobs should 429."""
    import importlib

    monkeypatch.setenv("SYNC_JOBS_RATE_LIMIT", "60/hour")
    importlib.reload(main)
    jobs_app = main.app
    jobs_limiter = main.limiter
    jobs_limiter.reset()
    try:
        with TestClient(jobs_app) as client:
            for _ in range(60):
                response = client.post(
                    "/sync/jobs",
                    data=_async_job_form(),
                    files={
                        "mp3": ("track.flac", b"fake-audio", "audio/flac"),
                        "lyrics": ("lyrics.txt", b"hello", "text/plain"),
                    },
                )
                assert response.status_code == 400
            response = client.post(
                "/sync/jobs",
                data=_async_job_form(),
                files={
                    "mp3": ("track.flac", b"fake-audio", "audio/flac"),
                    "lyrics": ("lyrics.txt", b"hello", "text/plain"),
                },
            )
        assert response.status_code == 429
        assert "rate" in response.text.lower() or response.json().get("error")
    finally:
        jobs_limiter.reset()
        monkeypatch.delenv("SYNC_JOBS_RATE_LIMIT", raising=False)
        importlib.reload(main)


def test_get_sync_job_unknown_returns_404():
    with TestClient(app) as client:
        response = client.get("/sync/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_sync_rejects_non_mp3_extension():
    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.wav", b"fake-audio", "audio/wav"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert ".mp3" in response.json()["detail"]


def test_sync_rejects_invalid_embed_mode():
    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", b"fake-audio", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
            data={"embed_mode": "banana"},
        )
    assert response.status_code == 400
    assert "embed_mode" in response.json()["detail"]


def test_sync_rejects_invalid_timestamp_mode():
    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", b"fake-audio", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
            data={"timestamp_mode": "banana"},
        )
    assert response.status_code == 400
    assert "timestamp_mode" in response.json()["detail"]


def test_sync_mp3_only_rejects_non_mp3_extension():
    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.wav", b"fake-audio", "audio/wav"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert ".mp3" in response.json()["detail"]


def test_sync_mp3_only_rejects_invalid_embed_mode():
    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.mp3", b"fake-audio", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
            data={"embed_mode": "banana"},
        )
    assert response.status_code == 400
    assert "embed_mode" in response.json()["detail"]


def test_sync_mp3_only_rejects_invalid_timestamp_mode():
    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.mp3", b"fake-audio", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
            data={"timestamp_mode": "banana"},
        )
    assert response.status_code == 400
    assert "timestamp_mode" in response.json()["detail"]


def test_sync_mp3_only_rejects_empty_mp3():
    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.mp3", b"", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded MP3 file is empty."


def test_sync_mp3_only_rejects_empty_lyrics():
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"  \n\t", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Lyrics file is empty."


def test_sync_rejects_empty_mp3():
    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", b"", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded MP3 file is empty."


def test_sync_rejects_empty_lyrics():
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"  \n\t", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Lyrics file is empty."


def test_get_client_ip_prefers_x_forwarded_for():
    from unittest.mock import MagicMock

    from app.main import _get_client_ip

    request = MagicMock()
    request.headers.get.return_value = "203.0.113.7, 10.0.0.1"
    assert _get_client_ip(request) == "203.0.113.7"


def test_get_client_ip_falls_back_to_remote_address(monkeypatch):
    from unittest.mock import MagicMock

    import app.main as main

    monkeypatch.setattr(main, "get_remote_address", lambda _request: "198.51.100.42")

    request = MagicMock()
    request.headers.get.return_value = None
    assert main._get_client_ip(request) == "198.51.100.42"


def test_synced_mp3_download_name_case_insensitive():
    from app.main import _mp3_upload_stem, _synced_mp3_download_name

    assert _synced_mp3_download_name("My Song.MP3") == "My Song_synced.mp3"
    assert _synced_mp3_download_name("track.mp3") == "track_synced.mp3"
    assert _mp3_upload_stem("") == "track"
    assert _mp3_upload_stem("weird.MP3") == "weird"


def test_sync_returns_sync_quality_headers(monkeypatch):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    import app.main as main
    from app.alignment import AlignmentResult

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir, timestamp_mode="line"):
        return AlignmentResult(
            lines=[("hello", 0)],
            quality="good",
            warnings=[],
            report={"quality": "good", "line_count": 1},
        )

    monkeypatch.setattr(main, "_run_alignment", _fake_alignment)
    monkeypatch.setattr(main, "write_sylt_tag", lambda *args, **kwargs: None)

    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"hello world", "text/plain"),
            },
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/zip")
    assert response.headers.get("X-Sync-Quality") == "good"
    assert "X-Sync-Warning" not in response.headers


def test_sync_mp3_only_returns_sync_quality_headers(monkeypatch):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    import app.main as main
    from app.alignment import AlignmentResult

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir, timestamp_mode="line"):
        return AlignmentResult(
            lines=[("hello", 0)],
            quality="degraded",
            warnings=["sparse bridge"],
            report={"quality": "degraded", "line_count": 1},
        )

    monkeypatch.setattr(main, "_run_alignment", _fake_alignment)
    monkeypatch.setattr(main, "write_sylt_tag", lambda *args, **kwargs: None)

    with TestClient(app) as client:
        response = client.post(
            "/sync/mp3-only",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"hello world", "text/plain"),
            },
        )

    assert response.status_code == 200
    assert response.headers.get("X-Sync-Quality") == "degraded"
    assert response.headers.get("X-Sync-Warning") == "sparse bridge"


def test_sync_word_mode_writes_word_sidecar(monkeypatch):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    import app.main as main
    from app.alignment import AlignmentResult, WordTiming

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir, timestamp_mode="line"):
        assert timestamp_mode == "word"
        return AlignmentResult(
            lines=[("hello world", 1200)],
            quality="good",
            warnings=[],
            report={
                "quality": "good",
                "line_count": 1,
                "duration_ms": 10000,
                "whisper_word_count": 2,
                "timestamp_mode": "word",
            },
            word_timings=[
                WordTiming(0, 0, "hello", 1200, 1500, True, 0.91),
                WordTiming(0, 1, "world", 1520, 1800, True, 0.88),
            ],
            detected_language="en",
            transcription_pass="vad_retry",
        )

    monkeypatch.setattr(main, "_run_alignment", _fake_alignment)
    monkeypatch.setattr(main, "write_sylt_tag", lambda *args, **kwargs: None)

    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"hello world", "text/plain"),
            },
            data={"timestamp_mode": "word"},
        )

    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    report_name = next(name for name in zf.namelist() if name.endswith("_sync_report.json"))
    words_name = next(name for name in zf.namelist() if name.endswith(".words.json"))
    report = json.loads(zf.read(report_name).decode("utf-8"))
    words = json.loads(zf.read(words_name).decode("utf-8"))
    assert report["timestamp_mode"] == "word"
    assert words["timestamp_mode"] == "word"
    assert words["language"] == "en"
    assert words["transcription_pass"] == "vad_retry"
    assert words["lines"][0]["words"][0]["word"] == "hello"


def test_content_disposition_attachment_ascii_and_utf8():
    from app.main import _content_disposition_attachment

    header = _content_disposition_attachment("song_synced.zip")
    assert 'filename="song_synced.zip"' in header
    assert "filename*=UTF-8''" in header

    header_unicode = _content_disposition_attachment("Müller_synced.zip")
    assert "filename=" in header_unicode
    assert "Müller" in header_unicode or "M%C3%BCller" in header_unicode


def test_lyrics_from_mp3_rejects_non_mp3_extension():
    with TestClient(app) as client:
        response = client.post(
            "/lyrics/from-mp3",
            files={"mp3": ("track.flac", b"fake-audio", "audio/flac")},
        )
    assert response.status_code == 400
    assert ".mp3" in response.json()["detail"]


def test_lyrics_from_mp3_rejects_empty_mp3():
    with TestClient(app) as client:
        response = client.post(
            "/lyrics/from-mp3",
            files={"mp3": ("track.mp3", b"", "audio/mpeg")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded MP3 file is empty."


def test_lyrics_from_mp3_returns_sources_json_for_tagless_mp3():
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    with TestClient(app) as client:
        response = client.post(
            "/lyrics/from-mp3",
            files={"mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"plain_lyrics", "timed_lyrics_lrc", "sources"}
    assert body["sources"] == {
        "uslt": False,
        "txxx_lyrics": False,
        "sylt": False,
        "tcon": False,
        "tbpm": False,
    }


def _async_job_form(**overrides):
    base = {
        "track_id": "cron-tick-track",
        "callback_url": "https://example.com/lyrics-sync-hook",
        "manual": "false",
    }
    base.update(overrides)
    return base


def _async_job_files(mp3_bytes=b"fake-audio", lyrics_text=b"line one"):
    return {
        "mp3": ("track.mp3", mp3_bytes, "audio/mpeg"),
        "lyrics": ("lyrics.txt", lyrics_text, "text/plain"),
    }


def test_enqueue_sync_job_rejects_non_mp3():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(),
            files={
                "mp3": ("track.wav", b"fake", "audio/wav"),
                "lyrics": ("lyrics.txt", b"line", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert ".mp3" in response.json()["detail"]


def test_enqueue_sync_job_rejects_blank_track_id():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(track_id="   "),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "track_id is required."


def test_enqueue_sync_job_rejects_blank_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="\t"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url is required."


def test_enqueue_sync_job_rejects_non_http_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="ftp://example.com/hook"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert "http" in response.json()["detail"]


def test_enqueue_sync_job_rejects_javascript_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="javascript:alert(1)"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url must be a valid http or https URL."


def test_enqueue_sync_job_rejects_file_scheme_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="file:///etc/passwd"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url must be a valid http or https URL."


def test_enqueue_sync_job_rejects_protocol_relative_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="//evil.example/hook"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url must be a valid http or https URL."


def test_enqueue_sync_job_rejects_data_scheme_callback_url():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url="data:text/html,<script>alert(1)</script>"),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url must be a valid http or https URL."


@pytest.mark.parametrize(
    "callback_url",
    [
        "http://",
        "http:///hook",
        "http:evil.example/hook",
    ],
)
def test_enqueue_sync_job_rejects_http_url_without_host(callback_url):
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(callback_url=callback_url),
            files=_async_job_files(),
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "callback_url must be a valid http or https URL."


@pytest.fixture
def stub_async_run_job(monkeypatch):
    async def _complete_immediately(job):
        job.status = "completed"
        job.completed_at = 1.0

    monkeypatch.setattr(async_jobs, "run_job", _complete_immediately)


def test_enqueue_sync_job_accepts_https_returns_202(stub_async_run_job):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    track_id = f"smoke-{uuid.uuid4()}"
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(track_id=track_id),
            files=_async_job_files(_SILENT_MP3_BYTES),
        )
    assert response.status_code == 202
    body = response.json()
    assert body["track_id"] == track_id
    assert body["status"] == "queued"
    assert body["manual"] is False
    assert body["job_id"]


def test_get_sync_job_after_enqueue_returns_snapshot(stub_async_run_job):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    track_id = f"poll-{uuid.uuid4()}"
    with TestClient(app) as client:
        post_resp = client.post(
            "/sync/jobs",
            data=_async_job_form(track_id=track_id),
            files=_async_job_files(_SILENT_MP3_BYTES),
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        get_resp = client.get(f"/sync/jobs/{job_id}")

    assert get_resp.status_code == 200
    snap = get_resp.json()
    assert snap["job_id"] == job_id
    assert snap["track_id"] == track_id
    assert snap["manual"] is False
    assert snap["status"] in {"queued", "processing", "completed"}
    assert "has_lyrics_lrc" in snap
    assert "created_at" in snap


def test_normalize_callback_url_strips_whitespace():
    from app.main import _normalize_callback_url

    assert (
        _normalize_callback_url("  https://example.com/lyrics-sync-hook  ")
        == "https://example.com/lyrics-sync-hook"
    )


def test_enqueue_sync_job_rejects_empty_lyrics():
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(),
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"  \n", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Lyrics file is empty."


def test_enqueue_sync_job_rejects_empty_mp3():
    with TestClient(app) as client:
        response = client.post(
            "/sync/jobs",
            data=_async_job_form(),
            files={
                "mp3": ("track.mp3", b"", "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"line one", "text/plain"),
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded MP3 file is empty."


def test_get_sync_job_requires_token_when_secret_configured(monkeypatch):
    monkeypatch.setenv("LYRIC_SYNC_CALLBACK_SECRET", "super-secret-token")
    with TestClient(app) as client:
        resp_missing = client.get("/sync/jobs/00000000-0000-0000-0000-000000000000")
        assert resp_missing.status_code == 401

        resp_wrong = client.get(
            "/sync/jobs/00000000-0000-0000-0000-000000000000",
            headers={"X-Lyrics-Sync-Token": "wrong-token"},
        )
        assert resp_wrong.status_code == 401


def test_sync_removes_job_dir_immediately(monkeypatch, tmp_path):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES
    from app.alignment import AlignmentResult

    work = tmp_path / "sync-work"
    work.mkdir()
    monkeypatch.setattr(main, "WORK_DIR", work)

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir, timestamp_mode="line"):
        return AlignmentResult(
            lines=[("hello", 0)],
            quality="good",
            warnings=[],
            report={"quality": "good", "line_count": 1},
        )

    monkeypatch.setattr(main, "_run_alignment", _fake_alignment)
    monkeypatch.setattr(main, "write_sylt_tag", lambda *args, **kwargs: None)

    with TestClient(app) as client:
        response = client.post(
            "/sync",
            files={
                "mp3": ("track.mp3", _SILENT_MP3_BYTES, "audio/mpeg"),
                "lyrics": ("lyrics.txt", b"hello world", "text/plain"),
            },
        )
    assert response.status_code == 200
    assert list(work.iterdir()) == []


def test_lyrics_extract_returns_multilingual_metadata_and_forwards_language_hint(monkeypatch, tmp_path):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES
    from app.alignment import TextTranscriptionResult

    work = tmp_path / "extract-work"
    work.mkdir()
    monkeypatch.setattr(main, "WORK_DIR", work)
    monkeypatch.setattr(
        main,
        "extract_lyrics_from_mp3",
        lambda _path: {
            "plain_lyrics": None,
            "timed_lyrics_lrc": None,
            "genres": [],
            "style_tags": [],
            "style_prompt_raw": None,
            "bpm": None,
            "notes": None,
        },
    )

    seen_languages = []

    async def _fake_transcription(job_id, mp3_path, job_dir, *, language=None):
        seen_languages.append(language)
        return TextTranscriptionResult(
            text="Je marche dans la nuit\nHold me closer now",
            detected_language=language or "fr",
            language_probability=0.8812,
            multilingual=(language is None),
            used_vocal_focus=True,
        )

    monkeypatch.setattr(main, "_run_transcription", _fake_transcription)

    with TestClient(app) as client:
        resp_auto = client.post(
            "/lyrics/extract",
            files={"mp3": ("chanson.mp3", _SILENT_MP3_BYTES, "audio/mpeg")},
        )
        assert resp_auto.status_code == 200
        body_auto = resp_auto.json()
        assert body_auto["source"] == "transcription"
        assert body_auto["detected_language"] == "fr"
        assert body_auto["language_probability"] == 0.881
        assert body_auto["multilingual"] is True

        resp_hint = client.post(
            "/lyrics/extract",
            files={"mp3": ("chanson.mp3", _SILENT_MP3_BYTES, "audio/mpeg")},
            data={"language": "FR"},
        )
        assert resp_hint.status_code == 200
        body_hint = resp_hint.json()
        assert body_hint["detected_language"] == "fr"
        assert body_hint["multilingual"] is False

        resp_bad = client.post(
            "/lyrics/extract",
            files={"mp3": ("chanson.mp3", _SILENT_MP3_BYTES, "audio/mpeg")},
            data={"language": "french-invalid"},
        )
        assert resp_bad.status_code == 400

    assert seen_languages == [None, "fr"]


def test_ensure_taggable_mp3_normalizes_unsupported_id3v20_header(tmp_path):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES

    # Prepend a synthetic ID3v2.0 header (version 2.0, flags 0, size 0) to valid MPEG frames.
    # mutagen raises ID3UnsupportedVersionError on ID3v2.0; _ensure_taggable_mp3 must normalize via ffmpeg.
    bad_mp3 = tmp_path / "bad_id3v20.mp3"
    bad_mp3.write_bytes(b"ID3\x00\x00\x00\x00\x00\x00\x00" + _SILENT_MP3_BYTES)

    normalized = main._ensure_taggable_mp3(bad_mp3, tmp_path)
    assert normalized.exists()
    assert normalized.name == "input_normalized.mp3"


def test_lyrics_extract_ignores_placeholder_embedded_tag_and_transcribes_audio(monkeypatch, tmp_path):
    from tests.test_sylt_writer import _SILENT_MP3_BYTES
    from app.alignment import TextTranscriptionResult

    work = tmp_path / "extract-placeholder-work"
    work.mkdir()
    monkeypatch.setattr(main, "WORK_DIR", work)
    monkeypatch.setattr(
        main,
        "extract_lyrics_from_mp3",
        lambda _path: {
            "plain_lyrics": "Instrumental",
            "timed_lyrics_lrc": None,
            "genres": [],
            "style_tags": [],
            "style_prompt_raw": None,
            "bpm": None,
            "notes": None,
        },
    )

    async def _fake_transcription(job_id, mp3_path, job_dir, *, language=None):
        return TextTranscriptionResult(
            text="Fuck this fake instrumental tag",
            detected_language="en",
            language_probability=0.95,
            multilingual=True,
            used_vocal_focus=True,
        )

    monkeypatch.setattr(main, "_run_transcription", _fake_transcription)

    with TestClient(app) as client:
        resp = client.post(
            "/lyrics/extract",
            files={"mp3": ("sneaky.mp3", _SILENT_MP3_BYTES, "audio/mpeg")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "transcription"
    assert body["plain_lyrics"] == "Fuck this fake instrumental tag"
    assert body["content_rating"]["rating"] == "explicit"
