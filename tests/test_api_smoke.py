"""Lightweight FastAPI smoke tests (no Whisper / alignment)."""

from fastapi.testclient import TestClient

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


def test_queue_returns_semaphore_and_async_stats():
    with TestClient(app) as client:
        response = client.get("/queue")
    assert response.status_code == 200
    body = response.json()
    assert body["waiting_jobs"] == 0
    assert body["total_slots"] >= 1
    assert "active_jobs" in body
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


def test_content_disposition_attachment_ascii_and_utf8():
    from app.main import _content_disposition_attachment

    header = _content_disposition_attachment("song_synced.zip")
    assert 'filename="song_synced.zip"' in header
    assert "filename*=UTF-8''" in header

    header_unicode = _content_disposition_attachment("Müller_synced.zip")
    assert "filename=" in header_unicode
    assert "M%C3%BCller" in header_unicode or "Müller" in header_unicode