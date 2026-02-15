"""
FastAPI application – accepts MP3 + lyrics, returns MP3 with synced lyrics.
"""

import shutil
import uuid
import logging
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles  # ← add this

from app.alignment import align_lyrics_to_audio
from app.sylt_writer import write_sylt_tag, write_lrc_file

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Lyrics Sync Service",
    description="Synchronize lyrics to MP3 audio using Whisper forced alignment.",
    version="1.0.0",
)

WORK_DIR = Path("/tmp/lyric-sync")
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ─── Serve index.html at root ───
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("/app/app/static/index.html")


# ─── All your existing endpoints below (unchanged) ───

@app.post("/sync", summary="Upload MP3 + lyrics, get back ZIP with synced MP3 + LRC")
async def sync_lyrics(
    mp3: UploadFile = File(..., description="MP3 audio file"),
    lyrics: UploadFile = File(..., description="Plain-text lyrics file (UTF-8)"),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp3_path = job_dir / "input.mp3"
        lyrics_path = job_dir / "lyrics.txt"

        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)
        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)

        lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics_text:
            raise HTTPException(status_code=400, detail="Lyrics file is empty.")

        synced = align_lyrics_to_audio(str(mp3_path), lyrics_text, job_dir=str(job_dir))

        base_name = Path(mp3.filename).stem
        output_mp3 = job_dir / f"{base_name}_synced.mp3"
        output_lrc = job_dir / f"{base_name}_synced.lrc"

        shutil.copy2(mp3_path, output_mp3)
        write_sylt_tag(str(output_mp3), synced)
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
                "Content-Disposition": f'attachment; filename="{base_name}_synced.zip"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Alignment failed")
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")


@app.post("/sync/mp3-only", summary="Upload MP3 + lyrics, get back only the tagged MP3")
async def sync_lyrics_mp3_only(
    mp3: UploadFile = File(..., description="MP3 audio file"),
    lyrics: UploadFile = File(..., description="Plain-text lyrics file (UTF-8)"),
):
    if not mp3.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Please upload an .mp3 file.")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp3_path = job_dir / "input.mp3"
        lyrics_path = job_dir / "lyrics.txt"

        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(mp3.file, f)
        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)

        lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics_text:
            raise HTTPException(status_code=400, detail="Lyrics file is empty.")

        synced = align_lyrics_to_audio(str(mp3_path), lyrics_text, job_dir=str(job_dir))

        output_path = job_dir / "output.mp3"
        shutil.copy2(mp3_path, output_path)
        write_sylt_tag(str(output_path), synced)

        return FileResponse(
            path=str(output_path),
            media_type="audio/mpeg",
            filename=mp3.filename.replace(".mp3", "_synced.mp3"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Alignment failed")
        raise HTTPException(status_code=500, detail=f"Alignment failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}