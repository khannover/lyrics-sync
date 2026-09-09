# Lyrics Sync Service — Agent & MCP Integration Prompt

Copy and paste this prompt into your agent's system prompt, `.cursorrules`, `.windsurfrules`, custom instructions, or MCP client configuration to teach your agent how to use the Lyrics Sync Service.

---

```markdown
## Service Role & Capabilities: Lyrics Sync Service

You have access to a local or remote **Lyrics Sync Service** (default: `http://localhost:8005`, production: `https://lyricsync.bancamp.de`).
The service synchronizes plain-text lyrics with MP3 audio files using **faster-whisper** (forced alignment) and **Dynamic Time Warping (DTW)**, manages ID3 metadata (SYLT synchronized lyrics frames, USLT, TCON, TBPM), provides **multi-tier musical profile & genre analysis** (ID3 + AI prompts, acoustic BPM/key/energy via librosa, and zero-PyTorch neural classification via ONNX-runtime), and provides **NSFW/explicit content filtering**, **Chromaprint copyright fingerprinting**, and **AI watermark & provenance detection**.

### Base URL Configuration
- Local instance: `http://localhost:8005`
- Remote / Production instance: `https://lyricsync.bancamp.de` (or the URL set in your environment as `LYRICS_SYNC_URL`)

Always use the configured base URL when issuing requests.

---

### Decision Matrix: Which Endpoint Should You Call?

1. **You want to extract musical profile, genres, BPM, key, energy, copyright fingerprint, AI provenance, and content rating:**
   → Use `POST /audio/analyze`
   - Runs a 3-tier cascade:
     1. Instant ID3 tags (`TCON`, `TBPM`) + Suno/Udio prompt parsing (`[Style: ...]`).
     2. Acoustic signal extraction (exact BPM, energy: `low`/`medium`/`high`, musical key: `B minor`).
     3. Neural genre classification via Discogs-EffNet ONNX model (runs in < 150ms on CPU).
   - Fast, offline, zero-PyTorch.
   - Includes:
     - `content_rating`: `clean`, `mild`, `explicit` with flagged terms
     - `copyright_fingerprint`: Chromaprint audio fingerprint string & duration for AcoustID lookup
     - `ai_provenance`: Suno/Udio/AI metadata markers, ultrasonic roll-off analysis, confidence score
   - Options: `force_neural=true`, `include_fingerprint=true`, `include_ai_provenance=true`.

2. **You want to get lyrics from an MP3 file (smart fallback + style tags):**
   → Use `POST /lyrics/extract`
   - Checks for existing embedded tags (`USLT`, `TXXX:LYRICS`, `SYLT`), genres, and style prompt preambles.
   - Automatically strips prompt preambles (`[Style: ...]`) from plain lyrics.
   - If tags exist, extracts and returns them immediately without AI overhead.
   - If no embedded lyrics exist, automatically runs Whisper transcription on the audio and returns the transcribed text.

3. **You ONLY want to read existing embedded ID3 tags from an MP3 (no Whisper):**
   → Use `POST /lyrics/from-mp3`
   - Fast, zero Whisper GPU/CPU cost. Returns plain lyrics (preambles stripped), LRC timed lyrics, normalized genres, style tags, and source flags.

3. **You have an MP3 and plain lyrics, and want both a synced MP3 and an LRC file:**
   → Use `POST /sync`
   - Returns a `.zip` archive containing:
     - `<name>_synced.mp3` (with embedded SYLT tag)
     - `<name>_synced.lrc` (standard LRC timed lyrics)
     - `<name>_sync_report.json` (alignment quality metadata & diagnostics)
   - Inspect response headers `X-Sync-Quality` (`good` | `degraded` | `fallback`) and `X-Sync-Warning`.

4. **You have an MP3 and plain lyrics, and ONLY need the tagged MP3 file directly:**
   → Use `POST /sync/mp3-only`
   - Returns the modified `.mp3` directly (Content-Type: `audio/mpeg`).

5. **Batch processing or long audio (background / async alignment):**
   → Use `POST /sync/jobs`
   - Enqueues job with HTTP 202 Accepted. Returns `{"job_id": "...", "status": "queued"}`.
   - Dispatches a webhook POST to `callback_url` upon completion.
   - Alternatively, poll status with `GET /sync/jobs/{job_id}`.
   - Optionally acknowledge completion with `POST /sync/jobs/{job_id}/ack`.

6. **Check server load or Whisper queue availability:**
   → Use `GET /queue` or `GET /health`
   - `waiting_jobs`: Requests waiting for the Whisper semaphore.
   - `total_slots`: Max concurrent Whisper alignments (usually 1 or 2).
   - `active_jobs`: Alignments currently in flight.

---

### Endpoint Reference & cURL Recipes

#### 1. Analyze Musical Profile, Safety, Fingerprint & Provenance
```bash
curl -s -X POST "http://localhost:8005/audio/analyze" \
  -F "mp3=@/path/to/track.mp3"
```
**JSON Response:**
```json
{
  "primary_genre": "High-Energy Techno",
  "genres": ["High-Energy Techno", "Ebm"],
  "style_tags": ["driving bass", "female vocals"],
  "style_prompt_raw": "High-energy Techno, EBM, driving bass, female vocals, NO SLOP",
  "bpm": 123,
  "key": "B minor",
  "energy": "high",
  "content_rating": {
    "rating": "clean",
    "matched_categories": [],
    "flagged_terms": []
  },
  "copyright_fingerprint": {
    "fingerprint": "AQAAZEqSpEkSRYmSZUkU...",
    "duration_sec": 182.62,
    "algorithm": "chromaprint"
  },
  "ai_provenance": {
    "is_synthetic": true,
    "confidence": 0.99,
    "detected_source": "suno",
    "indicators": {
      "has_suno_url": true,
      "has_suno_comment": true,
      "has_style_prompt": true,
      "has_synthetic_brickwall": true,
      "spectral_rolloff_hz": 16345.2,
      "ultrasonic_ratio": 0.0001
    }
  },
  "lyrics": {
    "has_embedded": true,
    "plain_lyrics": "[Chorus]\n...",
    "timed_lyrics_lrc": null
  },
  "tier_breakdown": {
    "tier1_metadata": { "genres": ["High-Energy Techno", "Ebm"], "id3_bpm": null },
    "tier2_signal": { "bpm": 123, "energy": "high", "key": "B minor" },
    "tier3_neural": { "ran": false, "top_genres": [] }
  }
}
```

#### 2. Extract or Transcribe Lyrics (Smart)
```bash
curl -s -X POST "http://localhost:8005/lyrics/extract" \
  -F "mp3=@/path/to/track.mp3"
```
**JSON Response:**
```json
{
  "source": "embedded", // or "transcription"
  "plain_lyrics": "Line 1\nLine 2\n...",
  "timed_lyrics_lrc": "[00:12.34] Line 1\n[00:16.78] Line 2", // null if transcribed
  "genres": ["High-Energy Techno", "Ebm"],
  "style_tags": ["driving bass", "female vocals"],
  "style_prompt_raw": "High-energy Techno, EBM, driving bass, female vocals, NO SLOP",
  "bpm": 123,
  "content_rating": {
    "rating": "clean",
    "matched_categories": [],
    "flagged_terms": []
  },
  "notes": "..."
}
```

#### 3. Extract Embedded Lyrics Only
```bash
curl -s -X POST "http://localhost:8005/lyrics/from-mp3" \
  -F "mp3=@/path/to/track.mp3"
```
**JSON Response:**
```json
{
  "plain_lyrics": "Line 1\nLine 2",
  "timed_lyrics_lrc": "[00:12.34] Line 1\n[00:16.78] Line 2",
  "genres": ["High-Energy Techno", "Ebm"],
  "style_tags": ["driving bass", "female vocals"],
  "style_prompt_raw": "High-energy Techno, EBM, driving bass, female vocals, NO SLOP",
  "bpm": 123,
  "content_rating": {
    "rating": "clean",
    "matched_categories": [],
    "flagged_terms": []
  },
  "sources": { "uslt": true, "txxx_lyrics": false, "sylt": true },
  "notes": "Found embedded SYLT and USLT tags"
}
```

#### 4. Synchronize Lyrics (Download ZIP)
```bash
curl -s -X POST "http://localhost:8005/sync" \
  -F "mp3=@/path/to/track.mp3" \
  -F "lyrics=@/path/to/lyrics.txt" \
  -F "embed_mode=overwrite" \
  --output "synced_bundle.zip"
```
*Note on `embed_mode`:*
- `"overwrite"` (default): Replaces existing SYLT and USLT/TXXX frames with timestamped LRC text.
- `"sylt_only"`: Adds/updates only the synchronized SYLT frame, leaving existing plain USLT/TXXX lyrics untouched.

#### 5. Synchronize Lyrics (Download MP3 Only)
```bash
curl -s -X POST "http://localhost:8005/sync/mp3-only" \
  -F "mp3=@/path/to/track.mp3" \
  -F "lyrics=@/path/to/lyrics.txt" \
  -F "embed_mode=sylt_only" \
  --output "track_synced.mp3"
```

#### 6. Enqueue Async Job
```bash
curl -s -X POST "http://localhost:8005/sync/jobs" \
  -F "mp3=@/path/to/track.mp3" \
  -F "lyrics=@/path/to/lyrics.txt" \
  -F "track_id=my-track-123" \
  -F "callback_url=https://my-app.example.com/api/lyrics-webhook" \
  -F "manual=false"
```
**Webhook Payload delivered to `callback_url`:**
```json
{
  "job_id": "9f5e...",
  "track_id": "my-track-123",
  "status": "completed", // or "failed"
  "manual": false,
  "lyrics_lrc": "[00:10.50] Hello world\n...",
  "quality": "good", // "good" | "degraded" | "fallback"
  "warnings": [],
  "report": { ... },
  "error": null
}
```
If `LYRIC_SYNC_CALLBACK_SECRET` is configured on the server, verify the `X-Lyrics-Sync-Token` request header.

#### 7. Check Job Status & Acknowledge
```bash
# Poll status
curl -s "http://localhost:8005/sync/jobs/{job_id}"

# Acknowledge completion
curl -s -X POST "http://localhost:8005/sync/jobs/{job_id}/ack"
```

#### 8. Queue & Health Monitoring
```bash
# Check queue
curl -s "http://localhost:8005/queue"

# Check health & disk space
curl -s "http://localhost:8005/health"
```

---

### Operational Rules for Agents
- **Audio requirement:** Files uploaded to `mp3` must have a `.mp3` extension. The server automatically repairs non-MPEG streams via ffmpeg normalization if needed.
- **Lyrics formatting:** Plain-text lyrics should be UTF-8 encoded. Line breaks separate phrases.
- **Handling rate limits:** Respect HTTP 429 responses. Check `/queue` before queueing heavy synchronous alignments.
- **Quality check:** Inspect the `X-Sync-Quality` header or `report.quality`. `"good"` means high DTW confidence. `"degraded"` or `"fallback"` indicates Whisper had low token overlap or applied progressive/proportional spreading.
```
