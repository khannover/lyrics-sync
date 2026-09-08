# Lyrics Sync Service

A FastAPI-based service that synchronizes plain-text lyrics to MP3 audio files using **faster-whisper** for forced alignment and **DTW (Dynamic Time Warping)** for timing refinement.

## Tech Stack
- **FastAPI**: Web framework for the API.
- **faster-whisper**: High-performance Whisper implementation for transcription and alignment.
- **DTW (Dynamic Time Warping)**: Aligns user lyric tokens with Whisper word-level audio segments.
- **ffmpeg**: Audio normalization, format verification, and conversion.
- **mutagen**: Reads and writes ID3 tags (SYLT synchronized lyrics, USLT unsynchronized lyrics, and TXXX frames).

---

## Running the Service

### Using Docker Compose
The recommended way to run the service is using Docker Compose:

```bash
docker compose up --build
```

The service will be available at `http://localhost:8005`.

Docker Compose orchestrates three services:
- `nginx`: Public HTTP entrypoint on port `8005`.
- `lyric-sync`: Internal FastAPI backend on internal network port `8000`.
- `tarpit`: Internal sink for scanner/bot traffic.

> **Note:** This setup is HTTP-only by default and requires no domain name or TLS certificates to run locally. In production, it sits behind reverse proxy HTTPS (e.g. `https://lyricsync.bancamp.de`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNC_RATE_LIMIT` | `60/hour` | Rate limit for synchronous `/sync` and `/lyrics/extract` |
| `SYNC_MP3_ONLY_RATE_LIMIT` | same as `SYNC_RATE_LIMIT` | Rate limit for `/sync/mp3-only` |
| `SYNC_JOBS_RATE_LIMIT` | `1000/hour` | Rate limit for async job enqueuing (`POST /sync/jobs`) |
| `MAX_CONCURRENT_JOBS` | `1` | Max concurrent Whisper alignment jobs in semaphore (`GET /queue` -> `total_slots`) |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LYRIC_SYNC_CALLBACK_SECRET` | *(unset)* | If set, incoming webhook calls & status endpoints verify `X-Lyrics-Sync-Token` |
| `LYRIC_SYNC_CALLBACK_RETRIES` | `5` | Retry attempts for async webhook delivery |
| `LYRIC_SYNC_CALLBACK_TIMEOUT_SEC` | `30` | Timeout per webhook delivery attempt in seconds |
| `LYRIC_SYNC_ACK_RETENTION_SECONDS` | `86400` | Retention window (24h) for acknowledged async jobs |

---

## Bot Mitigation

Nginx sits in front of the application and routes known probe/scanner traffic to a slow tarpit sidecar before it reaches FastAPI.

Paths intercepted and tarpitted include:
- `/_next`, `/actuator/*`, `/geoserver/*`, `/admin/*`, `/manage/*`
- `/models/edit/nuclei_rce_test`, `/chat/completions`, `/SDK/webLanguage`
- Suspicious probe queries (e.g., `?XDEBUG_SESSION_START=...`)

Legitimate traffic (browsers with standard user agents, API clients, and `python-httpx`) flows straight through to the FastAPI application.

---

## API Endpoints

### 1. `POST /sync`
Upload an MP3 and a lyrics file to perform forced alignment. Returns a `.zip` archive containing:
- `<track>_synced.mp3`: MP3 tagged with ID3 SYLT synchronized lyrics.
- `<track>_synced.lrc`: Standard LRC timed lyrics file.
- `<track>_sync_report.json`: Alignment quality metrics, line count, warnings, and duration.

**Parameters (multipart/form-data):**
- `mp3` (file, required): MP3 audio file.
- `lyrics` (file, required): Plain-text lyrics file (UTF-8).
- `embed_mode` (string, optional, default: `"overwrite"`):
  - `"overwrite"`: Overwrites existing SYLT, USLT, and TXXX:LYRICS frames with LRC-timestamped text.
  - `"sylt_only"`: Adds/updates only the SYLT frame without touching existing plain USLT/TXXX lyrics.

**Response Headers:**
- `X-Sync-Quality`: `"good"` | `"degraded"` | `"fallback"`
- `X-Sync-Warning`: Semicolon-delimited warning strings (if any).

**Example:**
```bash
curl -X POST "http://localhost:8005/sync" \
  -F "mp3=@song.mp3" \
  -F "lyrics=@lyrics.txt" \
  -F "embed_mode=overwrite" \
  --output "synced_bundle.zip"
```

---

### 2. `POST /sync/mp3-only`
Same alignment process as `/sync`, but returns the synchronized MP3 file directly instead of a ZIP archive.

**Parameters (multipart/form-data):**
- `mp3` (file, required): MP3 audio file.
- `lyrics` (file, required): Plain-text lyrics file (UTF-8).
- `embed_mode` (string, optional, default: `"overwrite"`): `"overwrite"` or `"sylt_only"`.

**Response Headers:**
- `X-Sync-Quality`: `"good"` | `"degraded"` | `"fallback"`
- `X-Sync-Warning`: Semicolon-delimited warning strings.

**Example:**
```bash
curl -X POST "http://localhost:8005/sync/mp3-only" \
  -F "mp3=@song.mp3" \
  -F "lyrics=@lyrics.txt" \
  -F "embed_mode=sylt_only" \
  --output "song_synced.mp3"
```

---

### 3. `POST /sync/jobs` (Async Alignment Queue)
Enqueues an asynchronous forced-alignment job with durable SQLite/filesystem persistence and webhook callback delivery. Perfect for batch pipelines and long tracks.

**Parameters (multipart/form-data):**
- `mp3` (file, required): MP3 audio file.
- `lyrics` (file, required): Plain-text lyrics file (UTF-8).
- `track_id` (string, required): Client track ID for idempotency (returns existing job if already queued/processing).
- `callback_url` (string, required): Webhook URL (`http` or `https`) receiving the completion payload.
- `manual` (boolean, optional, default: `false`): Mark as manual sync (saves fallback-quality LRC).

**Response (HTTP 202 Accepted):**
```json
{
  "job_id": "8bb38cb5-...",
  "track_id": "track-101",
  "status": "queued",
  "manual": false
}
```

**Webhook Callback Payload (POST to `callback_url`):**
```json
{
  "job_id": "8bb38cb5-...",
  "track_id": "track-101",
  "status": "completed",
  "manual": false,
  "lyrics_lrc": "[00:12.34] First line\n[00:16.78] Second line",
  "quality": "good",
  "warnings": [],
  "report": {
    "quality": "good",
    "warnings": [],
    "line_count": 24,
    "duration_ms": 182000,
    "whisper_word_count": 142
  },
  "error": null
}
```
*Note: If `LYRIC_SYNC_CALLBACK_SECRET` is set on the server, the webhook will include the header `X-Lyrics-Sync-Token` matching the secret.*

---

### 4. `GET /sync/jobs/{job_id}`
Polls the status of an asynchronous alignment job.

**Example:**
```bash
curl "http://localhost:8005/sync/jobs/8bb38cb5-..."
```

---

### 5. `POST /sync/jobs/{job_id}/ack`
Acknowledges durable consumption of a completed job's result.

**Example:**
```bash
curl -X POST "http://localhost:8005/sync/jobs/8bb38cb5-.../ack"
```

---

### 6. `POST /lyrics/extract`
Smart lyrics extractor. Extracts embedded lyrics from ID3 tags, or automatically transcribes the audio using faster-whisper if no embedded lyrics are found.

**Parameters (multipart/form-data):**
- `mp3` (file, required): MP3 audio file.

**JSON Response:**
```json
{
  "source": "embedded",
  "plain_lyrics": "First line\nSecond line",
  "timed_lyrics_lrc": "[00:12.34] First line\n[00:16.78] Second line",
  "notes": "Found embedded SYLT and USLT tags"
}
```
*(If no embedded lyrics exist, `"source"` will be `"transcription"`, `"timed_lyrics_lrc"` will be `null`, and `"plain_lyrics"` will contain the AI transcription).*

**Example:**
```bash
curl -X POST "http://localhost:8005/lyrics/extract" \
  -F "mp3=@song.mp3"
```

---

### 7. `POST /lyrics/from-mp3`
Fast extraction of existing embedded ID3 lyric tags (`USLT`, `TXXX:LYRICS`, `SYLT`). Does **not** run Whisper AI transcription.

**Parameters (multipart/form-data):**
- `mp3` (file, required): MP3 audio file.

**JSON Response:**
```json
{
  "plain_lyrics": "Lyrics text...",
  "timed_lyrics_lrc": "[00:10.50] Lyrics line...",
  "sources": {
    "uslt": true,
    "txxx_lyrics": false,
    "sylt": true
  },
  "notes": null
}
```

**Example:**
```bash
curl -X POST "http://localhost:8005/lyrics/from-mp3" \
  -F "mp3=@song.mp3"
```

---

### 8. `GET /queue`
Returns current concurrency semaphore and background job counts.

**JSON Response:**
```json
{
  "waiting_jobs": 0,
  "total_slots": 1,
  "active_jobs": 0,
  "async_jobs": {
    "queued": 0,
    "processing": 0,
    "completed": 12,
    "failed": 0
  }
}
```

**Example:**
```bash
curl "http://localhost:8005/queue"
```

---

### 9. `GET /health`
Returns system health, version, Whisper semaphore metrics, async job stats, and disk usage for `/tmp/lyric-sync`.

**Example:**
```bash
curl "http://localhost:8005/health"
```

---

## Web UI
A built-in web interface is accessible at `http://localhost:8005/`.
Selecting an MP3 file automatically inspects the track for embedded lyrics via `POST /lyrics/from-mp3`, populating the text area with extracted lyrics and showing read-only synced LRC lines when present.

---

## Agent & MCP Integration Prompt

Want your AI coding assistant or autonomous agent (Cursor, Windsurf, Claude Code, Grok, ChatGPT, custom MCP clients) to interact with this service?

Copy and paste the instruction prompt below into your agent's system prompt, `.cursorrules`, `.windsurfrules`, or custom prompt configuration:

```markdown
## Lyrics Sync Service — Agent Capabilities & Instructions

You have access to a Lyrics Sync Service running at `http://localhost:8005` (or production `https://lyricsync.bancamp.de`).
The service synchronizes plain lyrics to MP3 audio using faster-whisper forced alignment and DTW, extracts and embeds ID3 SYLT/USLT tags, and provides AI audio transcription.

### When to Call Which Endpoint:
1. **Extract lyrics from MP3 (smart tag extract + Whisper fallback):**
   `POST /lyrics/extract` with form file `mp3=@track.mp3`.
   - Returns JSON: `source` ("embedded" or "transcription"), `plain_lyrics`, `timed_lyrics_lrc`.
2. **Read existing embedded tags only (zero AI overhead):**
   `POST /lyrics/from-mp3` with form file `mp3=@track.mp3`.
   - Returns JSON: `plain_lyrics`, `timed_lyrics_lrc`, and `sources` dict.
3. **Align MP3 + lyrics into synced bundle (.zip):**
   `POST /sync` with form files `mp3=@track.mp3`, `lyrics=@lyrics.txt`, and optional `embed_mode` ("overwrite" or "sylt_only").
   - Returns ZIP containing `_synced.mp3`, `_synced.lrc`, and `_sync_report.json`.
4. **Align MP3 + lyrics into direct tagged MP3:**
   `POST /sync/mp3-only` with `mp3=@track.mp3`, `lyrics=@lyrics.txt`, and optional `embed_mode`.
   - Returns tagged MP3 directly (`audio/mpeg`).
5. **Background / asynchronous alignment for long audio or batch jobs:**
   `POST /sync/jobs` with `mp3=@...`, `lyrics=@...`, `track_id=...`, `callback_url=...`.
   - Poll with `GET /sync/jobs/{job_id}` and acknowledge with `POST /sync/jobs/{job_id}/ack`.
6. **Check system capacity / Whisper load:**
   `GET /queue` or `GET /health` to inspect `waiting_jobs` and `active_jobs`.

### Agent Guidelines:
- Uploaded files for `mp3` must end with `.mp3`.
- Inspect the response header `X-Sync-Quality` (`good`, `degraded`, or `fallback`).
- Set `embed_mode=sylt_only` if you want to keep existing plain unsynchronized lyrics intact.
```

A standalone copy-pasteable version is also available in [AGENT_PROMPT.md](AGENT_PROMPT.md).

### MCP Tool Schema Definitions
For agents configured with MCP (Model Context Protocol) or function calling, you can define these tools:

```json
[
  {
    "name": "extract_or_transcribe_lyrics",
    "description": "Extract embedded lyrics from an MP3 file, or transcribe audio using Whisper if no embedded lyrics exist.",
    "parameters": {
      "type": "object",
      "properties": {
        "mp3_path": { "type": "string", "description": "Local path to the .mp3 file" }
      },
      "required": ["mp3_path"]
    }
  },
  {
    "name": "extract_embedded_lyrics",
    "description": "Quickly extract existing embedded ID3 lyric tags (USLT, SYLT) from an MP3 without AI transcription.",
    "parameters": {
      "type": "object",
      "properties": {
        "mp3_path": { "type": "string", "description": "Local path to the .mp3 file" }
      },
      "required": ["mp3_path"]
    }
  },
  {
    "name": "sync_lyrics",
    "description": "Align plain-text lyrics to an MP3 file and download the synced MP3 + LRC bundle.",
    "parameters": {
      "type": "object",
      "properties": {
        "mp3_path": { "type": "string", "description": "Path to input MP3" },
        "lyrics_path": { "type": "string", "description": "Path to UTF-8 text file containing lyrics" },
        "embed_mode": { "type": "string", "enum": ["overwrite", "sylt_only"], "default": "overwrite" },
        "output_zip_path": { "type": "string", "description": "Destination path for the output ZIP archive" }
      },
      "required": ["mp3_path", "lyrics_path", "output_zip_path"]
    }
  },
  {
    "name": "get_lyrics_sync_queue",
    "description": "Check Whisper alignment queue status and concurrency limits.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  }
]
```

---

## Live Quality Check & Testing

To run the automated test suite and live alignment validation on a real audio track:

```bash
cd /mnt/c/projects/lyrics-sync
./scripts/quality_gate.sh
```

The quality gate executes:
1. `GET /health` sanity check.
2. Fast pytest bundle: `test_sync_quality_unit.py`, `test_api_smoke.py`, `test_cleanup.py`, and `test_async_jobs.py`.
3. Live MP3 sync test via `scripts/live_sync_quality_check.py` (checks monotonic LRC timestamps, coverage ratio, and `X-Sync-Quality`).
