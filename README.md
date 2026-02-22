# Lyrics Sync Service

A FastAPI-based service that synchronizes plain-text lyrics to MP3 audio files using **faster-whisper** for forced alignment and **DTW** for timing refinement.

## Tech Stack
- **FastAPI**: Web framework for the API.
- **faster-whisper**: High-performance Whisper implementation for transcription and alignment.
- **DTW (Dynamic Time Warping)**: Used to align text tokens with audio segments.
- **ffmpeg**: Handles audio processing and conversion.
- **mutagen**: Writes SYLT tags to MP3 files.

## Running the Service

### Using Docker Compose
The easiest way to run the service is using Docker Compose:

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`.

## API Endpoints

### 1. `POST /sync`
Upload an MP3 and a lyrics file to get a ZIP containing the synced MP3 and an LRC file.

**Example Curl:**
```bash
curl -X POST "http://localhost:8000/sync" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  --output synced_lyrics.zip
```

### 2. `POST /sync/mp3-only`
Upload an MP3 and a lyrics file to get back a single MP3 file with embedded SYLT lyrics.

**Example Curl:**
```bash
curl -X POST "http://localhost:8000/sync/mp3-only" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  --output song_synced.mp3
```

### 3. `GET /queue`
Returns the status of the job queue.

### 4. `GET /health`
Returns health status and disk usage statistics.

## Web UI
A simple web interface is available at the root URL: `http://localhost:8000/`.
