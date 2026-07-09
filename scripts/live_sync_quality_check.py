#!/usr/bin/env python3
"""
POST a real MP3 + lyrics.txt to a running lyrics-sync service and validate output.

Env:
  LYRICS_SYNC_BASE_URL  default http://127.0.0.1:8005
  LIVE_SYNC_MP3         path to .mp3
  LIVE_SYNC_LYRICS      path to .txt (or set LIVE_SYNC_TRACK_ID + BANCAMP_DB)
  LIVE_SYNC_TRACK_ID    bancamp track UUID (loads mp3 + lyrics from dev DB)
  BANCAMP_DB            default /mnt/c/projects/bancamp/backend/data/bancamp.db
  ALLOW_FALLBACK        if 1, accept quality=fallback

Exit 0 when quality gates pass; 2 on gate failure; 1 on setup/HTTP errors.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sync_quality import evaluate_sync_report  # noqa: E402


def _load_bancamp_fixture(track_id: str, db_path: Path) -> tuple[Path, Path, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, title, lyrics FROM tracks WHERE id=? AND status='approved'",
            (track_id,),
        ).fetchone()
        if not row:
            raise SystemExit(f"track not found or not approved: {track_id}")
        _id, title, lyrics = row
        mp3 = Path(f"/mnt/c/projects/bancamp/backend/data/music/{_id}.mp3")
        if not mp3.is_file():
            raise SystemExit(f"mp3 missing: {mp3}")
        if not (lyrics or "").strip():
            raise SystemExit(f"track has no lyrics: {track_id}")
        tmp = Path(tempfile.mkdtemp(prefix="lyrics-sync-live-"))
        lyrics_path = tmp / "lyrics.txt"
        lyrics_path.write_text(lyrics.strip() + "\n", encoding="utf-8")
        return mp3, lyrics_path, title
    finally:
        conn.close()


def _wait_for_service_ready(base: str, *, timeout_s: float = 90.0, interval_s: float = 3.0) -> None:
    """Poll /health through nginx until upstream is up (avoids 502 right after container restart)."""
    deadline = time.monotonic() + timeout_s
    last_status: int | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base}/health", timeout=10.0)
            last_status = resp.status_code
            if resp.status_code == 200:
                return
            if resp.status_code not in (502, 503, 504):
                resp.raise_for_status()
        except httpx.HTTPError:
            pass
        time.sleep(interval_s)
    raise SystemExit(
        f"service not ready after {timeout_s:.0f}s (last HTTP {last_status}); "
        "is lyric-sync up? docker compose ps"
    )


def main() -> int:
    base = os.environ.get("LYRICS_SYNC_BASE_URL", "http://127.0.0.1:8005").rstrip("/")
    allow_fallback = os.environ.get("ALLOW_FALLBACK", "").strip() in ("1", "true", "yes")

    mp3_path: Path | None = None
    lyrics_path: Path | None = None
    label = "live"

    track_id = os.environ.get("LIVE_SYNC_TRACK_ID", "").strip()
    if track_id:
        db = Path(
            os.environ.get(
                "BANCAMP_DB",
                "/mnt/c/projects/bancamp/backend/data/bancamp.db",
            )
        )
        mp3_path, lyrics_path, label = _load_bancamp_fixture(track_id, db)
    else:
        mp3_env = os.environ.get("LIVE_SYNC_MP3", "").strip()
        lyr_env = os.environ.get("LIVE_SYNC_LYRICS", "").strip()
        if not mp3_env or not lyr_env:
            print(
                "Set LIVE_SYNC_MP3 + LIVE_SYNC_LYRICS or LIVE_SYNC_TRACK_ID",
                file=sys.stderr,
            )
            return 1
        mp3_path = Path(mp3_env)
        lyrics_path = Path(lyr_env)
        label = mp3_path.stem

    if not mp3_path.is_file() or not lyrics_path.is_file():
        print("mp3 or lyrics path missing", file=sys.stderr)
        return 1

    _wait_for_service_ready(base)

    with mp3_path.open("rb") as mp3_f, lyrics_path.open("rb") as lyr_f:
        files = {
            "mp3": (mp3_path.name, mp3_f, "audio/mpeg"),
            "lyrics": ("lyrics.txt", lyr_f, "text/plain"),
        }
        print(f"POST {base}/sync track={label} mp3={mp3_path.name} …", flush=True)
        resp = httpx.post(f"{base}/sync", files=files, timeout=600.0)

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        return 1

    header_quality = resp.headers.get("X-Sync-Quality", "")
    header_warn = resp.headers.get("X-Sync-Warning", "")
    print(f"X-Sync-Quality: {header_quality}")
    if header_warn:
        print(f"X-Sync-Warning: {header_warn}")

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    report_name = next((n for n in zf.namelist() if n.endswith("_sync_report.json")), None)
    lrc_name = next((n for n in zf.namelist() if n.endswith("_synced.lrc")), None)
    if not lrc_name:
        print(f"zip missing synced LRC: {zf.namelist()}", file=sys.stderr)
        return 1

    lrc_text = zf.read(lrc_name).decode("utf-8", errors="replace")

    if report_name:
        report = json.loads(zf.read(report_name).decode("utf-8"))
    else:
        # Older lyric-sync images shipped ZIP without *_sync_report.json.
        report = {
            "quality": header_quality or "unknown",
            "warnings": [w.strip() for w in header_warn.split(";") if w.strip()],
            "line_count": sum(1 for ln in lrc_text.splitlines() if ln.strip()),
            "duration_ms": 0,
            "whisper_word_count": -1,
            "legacy_zip": True,
        }
        print(
            "note: ZIP has no *_sync_report.json (rebuild lyric-sync image); "
            "using header + LRC-only gates",
            file=sys.stderr,
        )
        if not header_quality:
            print(
                "error: rebuild required — docker compose build lyric-sync && "
                "docker compose up -d lyric-sync",
                file=sys.stderr,
            )
            return 1

    if header_quality and report.get("quality") and header_quality != report.get("quality"):
        print(
            f"warning: header quality {header_quality!r} != report {report.get('quality')!r}",
            file=sys.stderr,
        )

    verdict = evaluate_sync_report(
        report,
        lrc_text=lrc_text,
        allow_fallback=allow_fallback,
    )
    print(verdict.summary())
    return 0 if verdict.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())