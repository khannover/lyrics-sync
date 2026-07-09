"""Lightweight FastAPI smoke tests (no Whisper / alignment)."""

import uuid

import pytest
from fastapi.testclient import TestClient

import app.async_jobs as async_jobs
from app.main import app


def test_health_returns_disk_stats():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
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
    assert body["waiting_jobs"] == 0
    assert body["total_slots"] >= 1
    assert body["active_jobs"] == 0
    async_jobs = body["async_jobs"]
    assert set(async_jobs.keys()) == {"queued", "processing", "completed", "failed"}


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

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir):
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

    async def _fake_alignment(job_id, mp3_path, lyrics_text, job_dir):
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
    assert body["sources"] == {"uslt": False, "txxx_lyrics": False, "sylt": False}


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