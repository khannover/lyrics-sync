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
from fastapi.responses import FileResponse, StreamingResponse
from mutagen.mp3 import MP3, HeaderNotFoundError

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.alignment import align_lyrics_to_audio
from app.sylt_writer import write_sylt_tag, write_lrc_file
from app.lyrics_tag_reader import extract_lyrics_from_mp3

WORK_DIR = Path("/tmp/lyric-sync")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CLEANUP_INTERVAL_SECONDS = 600
MAX_TEMP_AGE_SECONDS = 3600

def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    return get_remote_address(request)


limiter = Limiter(key_func=_get_client_ip)
semaphore = asyncio.Semaphore(2)
waiting_jobs = 0
SYNC_RATE_LIMIT = os.environ.get("SYNC_RATE_LIMIT", "60/hour")
SYNC_MP3_ONLY_RATE_LIMIT = os.environ.get("SYNC_MP3_ONLY_RATE_LIMIT", SYNC_RATE_LIMIT)


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
        logging.warning("Input is not a taggable MP3 stream, transcoding via ffmpeg: %s", input_path.name)

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
    # Background cleanup task
    async def cleanup_loop():
        logging.info("Starting background cleanup task...")
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

                    if item.is_dir():
                        logging.info("Cleaning up old job folder: %s", item.name)
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        logging.info("Cleaning up old temp file: %s", item.name)
                        item.unlink(missing_ok=True)
            except asyncio.CancelledError:
                break
            except Exception:
                logging.exception("Error in cleanup task")

    cleanup_task = asyncio.create_task(cleanup_loop())
    yield
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

    try:
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

        global waiting_jobs
        waiting_jobs += 1
        async with semaphore:
            waiting_jobs -= 1
            synced = await asyncio.to_thread(
                align_lyrics_to_audio, str(mp3_path), lyrics_text, job_dir=str(job_dir)
            )

        base_name = Path(mp3.filename).stem
        output_mp3 = job_dir / f"{base_name}_synced.mp3"
        output_lrc = job_dir / f"{base_name}_synced.lrc"

        shutil.copy2(mp3_path, output_mp3)
        write_sylt_tag(str(output_mp3), synced, embed_mode=embed_mode)
        write_lrc_file(str(output_lrc), synced)

        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            zf.write(output_mp3, f"{base_name}_synced.mp3")
            zf.write(output_lrc, f"{base_name}_synced.lrc")
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": _content_disposition_attachment(f"{base_name}_synced.zip")
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Alignment failed")
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")


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

    try:
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

        global waiting_jobs
        waiting_jobs += 1
        async with semaphore:
            waiting_jobs -= 1
            synced = await asyncio.to_thread(
                align_lyrics_to_audio, str(mp3_path), lyrics_text, job_dir=str(job_dir)
            )

        output_path = job_dir / "output.mp3"
        shutil.copy2(mp3_path, output_path)
        write_sylt_tag(str(output_path), synced, embed_mode=embed_mode)

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
        logging.exception("Alignment failed")
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")


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

    try:
        mp3_path = job_dir / "input.mp3"
        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)

        result = await asyncio.to_thread(extract_lyrics_from_mp3, str(mp3_path))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Lyrics extraction failed")
        raise HTTPException(status_code=500, detail=f"Lyrics extraction failed: {e}")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/queue", summary="Returns current number of waiting jobs")
async def get_queue():
    return {
        "waiting_jobs": waiting_jobs,
        "total_slots": 2,
        "active_jobs": 2 - semaphore._value
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