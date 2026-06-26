import json
import shutil
import uuid
import logging
import asyncio
import time
import re
import subprocess
import os
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from mutagen.mp3 import MP3, HeaderNotFoundError

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address  # kept for potential future use / tests
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.alignment import AlignmentResult, align_lyrics_to_audio
from app.async_jobs import (
    create_job,
    get_job,
    queue_stats,
    start_worker,
    stop_worker,
)
from app.sylt_writer import write_sylt_tag, write_lrc_file
from app.lyrics_tag_reader import extract_lyrics_from_mp3

logger = logging.getLogger(__name__)

WORK_DIR = Path("/tmp/lyric-sync")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CLEANUP_INTERVAL_SECONDS = 600
MAX_TEMP_AGE_SECONDS = 3600

def _get_client_ip(request: Request) -> str:
    """Return the original client IP.

    Trusts X-Forwarded-For / X-Real-IP set by nginx. Falls back to the
    direct peer only when running without a proxy (dev, tests).
    """
    # X-Forwarded-For: "client, proxy1, ..."
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()

    # Single-value header commonly set by nginx
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()

    # Direct TCP peer (no proxy in front)
    if request.client and request.client.host:
        return request.client.host

    return "unknown"


limiter = Limiter(key_func=_get_client_ip)
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
waiting_jobs = 0
active_job_ids: set[str] = set()
SYNC_RATE_LIMIT = os.environ.get("SYNC_RATE_LIMIT", "60/hour")
SYNC_MP3_ONLY_RATE_LIMIT = os.environ.get("SYNC_MP3_ONLY_RATE_LIMIT", SYNC_RATE_LIMIT)
# Async job enqueue is the path used by Kai's library sync client. It is
# intentionally cheap (just queues work) and has track_id idempotency.
# Give it a generous limit so large music libraries can be submitted.
SYNC_JOBS_RATE_LIMIT = os.environ.get("SYNC_JOBS_RATE_LIMIT", "1000/hour")


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _job_log(job_id: str, event: str, **fields) -> None:
    parts = [f"[sync] job={job_id}", event]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


def _register_job(job_id: str) -> None:
    active_job_ids.add(job_id)


def _unregister_job(job_id: str) -> None:
    active_job_ids.discard(job_id)


def _touch_job_dir(job_dir: Path) -> None:
    try:
        os.utime(job_dir, None)
    except OSError:
        pass


def _assert_audio_ready(job_id: str, mp3_path: Path, job_dir: Path) -> None:
    if mp3_path.exists():
        return

    try:
        listing = ", ".join(sorted(p.name for p in job_dir.iterdir()))
    except OSError as exc:
        listing = f"<unreadable: {exc}>"

    raise FileNotFoundError(
        f"Input audio file not found before alignment: {mp3_path} "
        f"(job_dir listing: {listing or 'empty'})"
    )


async def _run_alignment(
    job_id: str,
    mp3_path: Path,
    lyrics_text: str,
    job_dir: Path,
) -> AlignmentResult:
    global waiting_jobs
    line_count = len([line for line in lyrics_text.splitlines() if line.strip()])
    _job_log(
        job_id,
        "stage=queued",
        file=mp3_path.name,
        lines=line_count,
        waiting=waiting_jobs,
        slots=MAX_CONCURRENT_JOBS,
    )

    waiting_jobs += 1
    started = time.monotonic()
    try:
        async with semaphore:
            waiting_jobs -= 1
            _touch_job_dir(job_dir)
            _assert_audio_ready(job_id, mp3_path, job_dir)
            mp3_bytes = mp3_path.stat().st_size
            _job_log(
                job_id,
                "stage=align_start",
                file=mp3_path.name,
                mp3_bytes=mp3_bytes,
            )
            synced = await asyncio.to_thread(
                align_lyrics_to_audio,
                str(mp3_path),
                lyrics_text,
                job_dir=str(job_dir),
                job_id=job_id,
            )
            _job_log(
                job_id,
                "stage=align_finished",
                file=mp3_path.name,
                elapsed=f"{time.monotonic() - started:.1f}s",
                lines=len(synced.lines),
            )
            return synced
    except Exception:
        _job_log(
            job_id,
            "stage=align_failed",
            file=mp3_path.name,
            elapsed=f"{time.monotonic() - started:.1f}s",
        )
        raise


def _content_disposition_attachment(filename: str) -> str:
    """Build a Content-Disposition value that is safe for non-ASCII filenames."""
    # Header fallback for legacy clients: keep only ASCII printable chars.
    ascii_fallback = re.sub(r"[^\x20-\x7E]", "_", filename).replace('"', "")
    if not ascii_fallback.strip():
        ascii_fallback = "download.zip"

    # RFC 5987 encoding for full UTF-8 filename support.
    utf8_encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{utf8_encoded}"


def _ensure_taggable_mp3(input_path: Path, job_dir: Path) -> Path:
    """
    Return a valid MP3 path that mutagen can tag.

    Some generators/exporters produce files with .mp3 extension that are not
    actual MPEG-frame MP3 streams. In that case we transcode once via ffmpeg.
    """
    try:
        MP3(str(input_path))
        return input_path
    except HeaderNotFoundError:
        logger.warning("Input is not a taggable MP3 stream, transcoding via ffmpeg: %s", input_path.name)

    normalized_path = job_dir / "input_normalized.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(normalized_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not normalized_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid MP3 stream and could not be converted.",
        )

    try:
        MP3(str(normalized_path))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio could not be normalized into a valid MP3.",
        )

    return normalized_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    logger.info("Lyrics sync service started")

    # Background cleanup task
    async def cleanup_loop():
        logger.info("Starting background cleanup task...")
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                now = time.time()
                for item in WORK_DIR.iterdir():
                    try:
                        mtime = item.stat().st_mtime
                    except FileNotFoundError:
                        # Item may disappear while iterating.
                        continue

                    if (now - mtime) <= MAX_TEMP_AGE_SECONDS:
                        continue

                    if item.name in active_job_ids:
                        continue

                    if item.is_dir():
                        logger.info("Cleaning up old job folder: %s", item.name)
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        logger.info("Cleaning up old temp file: %s", item.name)
                        item.unlink(missing_ok=True)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup task")

    cleanup_task = asyncio.create_task(cleanup_loop())
    await start_worker()
    yield
    await stop_worker()
    cleanup_task.cancel()
    await asyncio.gather(cleanup_task, return_exceptions=True)

app = FastAPI(
    title="Lyrics Sync Service",
    description="Synchronize lyrics to MP3 audio using Whisper forced alignment.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ─── Serve index.html at root ───
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("/app/app/static/index.html")


# ─── All your existing endpoints below (unchanged) ───

@app.post("/sync", summary="Upload MP3 + lyrics, get back ZIP with synced MP3 + LRC")
@limiter.limit(SYNC_RATE_LIMIT)
async def sync_lyrics(
    request: Request,
    mp3: UploadFile = File(..., description="MP3 audio file"),
    lyrics: UploadFile = File(..., description="Plain-text lyrics file (UTF-8)"),
    embed_mode: str = Form(
        default="overwrite",
        description='Embedding mode: "overwrite" overwrites USLT/TXXX with LRC-timestamped text; "sylt_only" adds only the SYLT frame without touching plain lyrics.',
    ),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")
    if embed_mode not in ("overwrite", "sylt_only"):
        raise HTTPException(status_code=400, detail='embed_mode must be "overwrite" or "sylt_only".')

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _register_job(job_id)
    request_started = time.monotonic()
    client_ip = _get_client_ip(request)

    try:
        _job_log(
            job_id,
            "stage=request_start",
            endpoint="/sync",
            client=client_ip,
            file=mp3.filename,
            embed_mode=embed_mode,
        )

        mp3_path = job_dir / "input.mp3"
        lyrics_path = job_dir / "lyrics.txt"

        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)

        if mp3_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded MP3 file is empty.")

        mp3_path = _ensure_taggable_mp3(mp3_path, job_dir)

        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)

        lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics_text:
            raise HTTPException(status_code=400, detail="Lyrics file is empty.")

        alignment = await _run_alignment(job_id, mp3_path, lyrics_text, job_dir)
        synced = alignment.lines

        base_name = Path(mp3.filename).stem
        output_mp3 = job_dir / f"{base_name}_synced.mp3"
        output_lrc = job_dir / f"{base_name}_synced.lrc"

        _job_log(job_id, "stage=tagging_start", file=mp3_path.name)
        shutil.copy2(mp3_path, output_mp3)
        write_sylt_tag(str(output_mp3), synced, embed_mode=embed_mode)
        write_lrc_file(str(output_lrc), synced)
        response_headers = {
            "Content-Disposition": _content_disposition_attachment(f"{base_name}_synced.zip"),
            "X-Sync-Quality": alignment.quality,
        }
        if alignment.warnings:
            response_headers["X-Sync-Warning"] = "; ".join(alignment.warnings)
        _job_log(job_id, "stage=tagging_done", file=mp3_path.name)

        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            zf.write(output_mp3, f"{base_name}_synced.mp3")
            zf.write(output_lrc, f"{base_name}_synced.lrc")
            if alignment.report:
                zf.writestr(
                    f"{base_name}_sync_report.json",
                    json.dumps(alignment.report, ensure_ascii=False),
                )
        zip_buffer.seek(0)

        _job_log(
            job_id,
            "stage=request_done",
            endpoint="/sync",
            file=mp3.filename,
            elapsed=f"{time.monotonic() - request_started:.1f}s",
            zip_bytes=zip_buffer.getbuffer().nbytes,
        )

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers=response_headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[sync] job=%s stage=request_failed", job_id)
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")
    finally:
        _unregister_job(job_id)


@app.post(
    "/sync/jobs",
    summary="Queue async lyric sync; results POST to callback_url when done",
    status_code=202,
)
@limiter.limit(SYNC_JOBS_RATE_LIMIT)
async def enqueue_sync_job(
    request: Request,
    mp3: UploadFile = File(..., description="MP3 audio file"),
    lyrics: UploadFile = File(..., description="Plain-text lyrics file (UTF-8)"),
    track_id: str = Form(..., description="Client track identifier for idempotency"),
    callback_url: str = Form(..., description="Webhook URL for completion payload"),
    manual: bool = Form(default=False, description="Manual sync — save fallback-quality LRC"),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")
    if not (track_id or "").strip():
        raise HTTPException(status_code=400, detail="track_id is required.")
    if not (callback_url or "").strip():
        raise HTTPException(status_code=400, detail="callback_url is required.")

    upload_dir = WORK_DIR / f"upload-{uuid.uuid4()}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = upload_dir / "input.mp3"
    lyrics_path = upload_dir / "lyrics.txt"

    try:
        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)
        if mp3_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded MP3 file is empty.")

        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)
        if not lyrics_path.read_text(encoding="utf-8").strip():
            raise HTTPException(status_code=400, detail="Lyrics file is empty.")

        job = create_job(
            track_id=track_id.strip(),
            callback_url=callback_url.strip(),
            manual=bool(manual),
            mp3_path=mp3_path,
            lyrics_path=lyrics_path,
        )
        payload = {
            "job_id": job.job_id,
            "track_id": job.track_id,
            "status": job.status,
            "manual": job.manual,
        }
        return JSONResponse(status_code=202, content=payload)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("[async] enqueue_failed track=%s", track_id)
        raise HTTPException(status_code=500, detail=f"Failed to queue sync job: {exc}")
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.get("/sync/jobs/{job_id}", summary="Poll async lyric-sync job status")
async def get_sync_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/sync/mp3-only", summary="Upload MP3 + lyrics, get back only the tagged MP3")
@limiter.limit(SYNC_MP3_ONLY_RATE_LIMIT)
async def sync_lyrics_mp3_only(
    request: Request,
    mp3: UploadFile = File(..., description="MP3 audio file"),
    lyrics: UploadFile = File(..., description="Plain-text lyrics file (UTF-8)"),
    embed_mode: str = Form(
        default="overwrite",
        description='Embedding mode: "overwrite" overwrites USLT/TXXX with LRC-timestamped text; "sylt_only" adds only the SYLT frame without touching plain lyrics.',
    ),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")
    if embed_mode not in ("overwrite", "sylt_only"):
        raise HTTPException(status_code=400, detail='embed_mode must be "overwrite" or "sylt_only".')

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _register_job(job_id)
    request_started = time.monotonic()
    client_ip = _get_client_ip(request)

    try:
        _job_log(
            job_id,
            "stage=request_start",
            endpoint="/sync/mp3-only",
            client=client_ip,
            file=mp3.filename,
            embed_mode=embed_mode,
        )

        mp3_path = job_dir / "input.mp3"
        lyrics_path = job_dir / "lyrics.txt"

        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)

        if mp3_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded MP3 file is empty.")

        mp3_path = _ensure_taggable_mp3(mp3_path, job_dir)

        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)

        lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics_text:
            raise HTTPException(status_code=400, detail="Lyrics file is empty.")

        alignment = await _run_alignment(job_id, mp3_path, lyrics_text, job_dir)
        synced = alignment.lines

        output_path = job_dir / "output.mp3"
        _job_log(job_id, "stage=tagging_start", file=mp3_path.name)
        shutil.copy2(mp3_path, output_path)
        write_sylt_tag(str(output_path), synced, embed_mode=embed_mode)
        _job_log(job_id, "stage=tagging_done", file=mp3_path.name)

        _job_log(
            job_id,
            "stage=request_done",
            endpoint="/sync/mp3-only",
            file=mp3.filename,
            elapsed=f"{time.monotonic() - request_started:.1f}s",
            mp3_bytes=output_path.stat().st_size,
        )

        return FileResponse(
            path=str(output_path),
            media_type="audio/mpeg",
            filename=re.sub(r"[^\x20-\x7E]", "_", mp3.filename.replace(".mp3", "_synced.mp3")),
            headers={
                "Content-Disposition": _content_disposition_attachment(mp3.filename.replace(".mp3", "_synced.mp3"))
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[sync] job=%s stage=request_failed", job_id)
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")
    finally:
        _unregister_job(job_id)


@app.post(
    "/lyrics/from-mp3",
    summary="Extract embedded lyrics from an MP3 file",
)
async def lyrics_from_mp3(
    mp3: UploadFile = File(..., description="MP3 audio file"),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.monotonic()

    try:
        _job_log(
            job_id,
            "stage=request_start",
            endpoint="/lyrics/from-mp3",
            file=mp3.filename,
        )

        mp3_path = job_dir / "input.mp3"
        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)

        _job_log(
            job_id,
            "stage=extract_start",
            file=mp3.filename,
            mp3_bytes=mp3_path.stat().st_size,
        )
        result = await asyncio.to_thread(extract_lyrics_from_mp3, str(mp3_path))
        _job_log(
            job_id,
            "stage=request_done",
            endpoint="/lyrics/from-mp3",
            file=mp3.filename,
            elapsed=f"{time.monotonic() - request_started:.1f}s",
            found=",".join(k for k, v in result.get("sources", {}).items() if v) or "none",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[sync] job=%s stage=request_failed", job_id)
        raise HTTPException(status_code=500, detail=f"Lyrics extraction failed: {e}")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/queue", summary="Returns current number of waiting jobs")
async def get_queue():
    async_stats = queue_stats()
    return {
        "waiting_jobs": waiting_jobs,
        "total_slots": MAX_CONCURRENT_JOBS,
        "active_jobs": MAX_CONCURRENT_JOBS - semaphore._value,
        "async_jobs": async_stats,
    }


@app.get("/health")
async def health():
    usage = shutil.disk_usage(WORK_DIR)
    return {
        "status": "ok",
        "disk": {
            "total_gb": round(usage.total / (2**30), 2),
            "used_gb": round(usage.used / (2**30), 2),
            "free_gb": round(usage.free / (2**30), 2),
        }
    }