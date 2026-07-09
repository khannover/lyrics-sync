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
docker compose up --build
```

The service will be available at `http://localhost:8005`.

Docker Compose now starts three services:
- `nginx`: public HTTP entrypoint on port `8005`
- `lyric-sync`: internal FastAPI app on Docker-network port `8000`
- `tarpit`: internal slow sink for obvious bot and scanner traffic

This setup stays HTTP-only. It does not require a domain name, TLS certificates, or any HTTPS configuration.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SYNC_RATE_LIMIT` | `60/hour` | Rate limit for `POST /sync` and `POST /sync/jobs` (slowapi; excess → HTTP 429) |
| `SYNC_MP3_ONLY_RATE_LIMIT` | same as `SYNC_RATE_LIMIT` | Rate limit for `POST /sync/mp3-only` |
| `MAX_CONCURRENT_JOBS` | `1` | Concurrent Whisper alignment jobs (`GET /queue` → `total_slots`) |
| `LOG_LEVEL` | `INFO` | Application log verbosity |
| `LYRIC_SYNC_CALLBACK_SECRET` | *(unset)* | If set, async job webhooks must send matching `X-Lyrics-Sync-Token` |

## Bot Mitigation

Nginx sits in front of the app and diverts obvious scan traffic to a tarpit service before those requests reach FastAPI. The tarpit intentionally responds slowly so common probes spend time outside the main application.

Examples of traffic diverted by Nginx include requests for paths such as:
- `/_next`
- `/actuator/*`
- `/geoserver/*`
- `/admin/*`
- `/manage/*`
- `/models/edit/nuclei_rce_test`
- `/chat/completions`
- `/SDK/webLanguage`
- suspicious query probes like `?XDEBUG_SESSION_START=...`

The app still keeps endpoint rate limiting enabled as a backup control for requests that reach FastAPI.

## API Endpoints

### 1. `POST /sync`
Upload an MP3 and a lyrics file to get a ZIP containing the synced MP3 and an LRC file.

**Parameters:**
- `mp3` (file): MP3 audio file
- `lyrics` (file): Plain-text lyrics file (UTF-8)
- `embed_mode` (string, optional, default `"overwrite"`): Embedding mode.
  - `"overwrite"`: Removes existing SYLT, USLT, and TXXX:LYRICS frames, then writes all of them with LRC-timestamped text.
  - `"sylt_only"`: Removes only existing SYLT frames and writes only the SYLT frame — existing plain (unsynced) USLT/TXXX:LYRICS lyrics are preserved.

**Response headers:** `X-Sync-Quality` and optional `X-Sync-Warning`. The ZIP includes `*_synced.mp3`, `*_synced.lrc`, and `*_sync_report.json`.

**Example Curl:**
```bash
curl -X POST "http://localhost:8005/sync" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  --output synced_lyrics.zip
```

Use `embed_mode=sylt_only` to keep existing plain lyrics intact:
```bash
curl -X POST "http://localhost:8005/sync" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  -F "embed_mode=sylt_only" \
  --output synced_lyrics.zip
```

### 2. `POST /sync/mp3-only`
Upload an MP3 and a lyrics file to get back a single MP3 file with embedded SYLT lyrics.

**Parameters:**
- `mp3` (file): MP3 audio file
- `lyrics` (file): Plain-text lyrics file (UTF-8)
- `embed_mode` (string, optional, default `"overwrite"`): Embedding mode (same options as `/sync`).

**Response headers:** `X-Sync-Quality` and optional `X-Sync-Warning` (same semantics as `POST /sync`).

**Example Curl:**
```bash
curl -X POST "http://localhost:8005/sync/mp3-only" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  --output song_synced.mp3
```

Use `embed_mode=sylt_only` to keep existing plain lyrics intact:
```bash
curl -X POST "http://localhost:8005/sync/mp3-only" \
  -F "mp3=@path/to/song.mp3" \
  -F "lyrics=@path/to/lyrics.txt" \
  -F "embed_mode=sylt_only" \
  --output song_synced.mp3
```

### 3. `POST /lyrics/from-mp3`
Extract any embedded lyrics (USLT, TXXX:LYRICS, SYLT) from an MP3 without re-syncing.

Returns JSON with:
- `plain_lyrics`: lyrics with timestamps stripped, or `null`
- `timed_lyrics_lrc`: LRC-format timed lyrics, or `null`
- `sources`: `{ "uslt": bool, "txxx_lyrics": bool, "sylt": bool }`
- `notes`: optional diagnostic string

**Example Curl:**
```bash
curl -X POST "http://localhost:8005/lyrics/from-mp3" \
  -F "mp3=@path/to/song.mp3"
```

### 4. `GET /queue`
Returns alignment semaphore and async job counts:

- `waiting_jobs`: requests blocked on the Whisper semaphore
- `total_slots`: `MAX_CONCURRENT_JOBS`
- `active_jobs`: alignments currently running
- `async_jobs`: `{ "queued", "processing", "completed", "failed" }` counts for `POST /sync/jobs`

### 5. `GET /health`
Returns health status and disk usage statistics.

## Live quality check (real MP3)

Smoke tests in `tests/test_api_smoke.py` do not run Whisper. To validate alignment on a real track against a running service:

```bash
cd /mnt/c/projects/lyrics-sync
./scripts/quality_gate.sh
```

The gate runs `curl /health`, fast pytest (`test_sync_quality_unit` + `test_api_smoke` + `test_cleanup`), then the live MP3 check. Set `SKIP_LIVE_SYNC=1` to skip Whisper. Do **not** pipe `live_sync_quality_check.py` through `tee` — that breaks exit-code checks.

Extended local pytest (cleanup + async jobs):

```bash
pytest tests/test_sync_quality_unit.py tests/test_api_smoke.py tests/test_cleanup.py tests/test_async_jobs.py -q
LIVE_SYNC_TRACK_ID=c7721ca1-e8d2-4045-8a5b-e53cfb29e7d2 \
  python3 scripts/live_sync_quality_check.py
```

Uses a Bancamp dev track (MP3 + DB lyrics) by default when `LIVE_SYNC_TRACK_ID` is set. The script waits up to 90s for `/health` through nginx (502/503 after container restart). Gates: `quality` is `good` or `degraded` (not `fallback`), monotonic LRC timestamps, minimum coverage and `whisper_word_count`. Response headers `X-Sync-Quality` / `X-Sync-Warning` and `*_sync_report.json` in the ZIP are checked.

### Async sync (`POST /sync/jobs`, `GET /sync/jobs/{job_id}`)
Queue alignment in the background and receive a JSON webhook at `callback_url` when done (`callback_url` must be `http` or `https`). Optional header `X-Lyrics-Sync-Token` if `LYRIC_SYNC_CALLBACK_SECRET` is set. Poll status with `GET /sync/jobs/{job_id}`.

## Web UI
A simple web interface is available at the root URL: `http://localhost:8005/`.

When you select an MP3 file in the Auto-Sync tab, the UI automatically calls `POST /lyrics/from-mp3` to detect any embedded lyrics. If found, the **Timed Lyrics (LRC)** are shown in a read-only collapsible panel, and the **plain lyrics** (timestamps stripped) are pre-populated in the lyrics textarea so you can review or edit them before clicking Sync.
