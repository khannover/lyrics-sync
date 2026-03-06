# Word Precision Sync Plan

## Goal
Add a second synchronization mode with word-precision timestamps while keeping current line-precision behavior as default and fully backward compatible.

## Scope
- Add `timestamp_mode` parameter to API endpoints.
- Keep `line` mode as default.
- Introduce `word` mode for higher precision.
- Preserve current outputs for existing clients.

## API Design
### New Parameter
- Name: `timestamp_mode`
- Type: form string
- Allowed values:
  - `line` (default)
  - `word`

### Endpoints
- `POST /sync`
- `POST /sync/mp3-only`

### Validation
- If `timestamp_mode` is missing: use `line`.
- If invalid: return HTTP 400 with clear message.

## Output Contract
### Line Mode (existing)
- ZIP contains:
  - `*_synced.mp3`
  - `*_synced.lrc`

### Word Mode (new)
- ZIP contains:
  - `*_synced.mp3`
  - `*_synced.lrc` (line fallback for compatibility)
  - `*_synced.words.json` (canonical word-level timings)

Rationale: players consume line-LRC reliably; advanced clients can consume JSON for true word timing.

## Internal Data Model
Introduce explicit internal timing structures:

- `LineTiming`
  - `line_index: int`
  - `text: str`
  - `start_ms: int`

- `WordTiming`
  - `line_index: int`
  - `word_index: int`
  - `word: str`
  - `start_ms: int`
  - `end_ms: int`
  - `matched: bool`
  - `confidence: float | None`

## Alignment Pipeline
### 1. Transcription
Reuse current Whisper word timestamp extraction (including no-VAD retry and coverage guardrails).

### 2. Word Alignment (new)
Align lyrics words to Whisper words with monotonic forward constraints.

Scoring signal per candidate window should combine:
- token equality/overlap
- sequence similarity
- positional penalty for large forward jumps
- local context bonus (neighbor words)

Requirements:
- monotonic timestamps
- robust repeated word handling
- marker lines excluded from word matching

### 3. Line Timing Derivation
In `word` mode, derive line starts from first matched word per line.
If a line has no matched words:
- interpolate from nearest matched lines
- or use conservative carry-forward fallback

### 4. Fallback Strategy
If matching coverage is poor:
- keep current progressive/guardrail logic
- ensure output still generated
- annotate logs with fallback reason

## Tag Writing Strategy
### SYLT
- In `word` mode, write SYLT entries at word granularity.
- In `line` mode, keep current line granularity.

### USLT / TXXX:LYRICS
- Keep line-LRC fallback for compatibility in both modes.
- Continue UTF-safe encoding strategy already implemented.

## JSON Sidecar Schema
File: `*_synced.words.json`

Suggested schema:

```json
{
  "version": 1,
  "timestamp_mode": "word",
  "language": "en",
  "lines": [
    {
      "line_index": 0,
      "text": "You program patterns but I break the code",
      "start_ms": 12340,
      "words": [
        {"word_index": 0, "word": "You", "start_ms": 12340, "end_ms": 12580, "matched": true, "confidence": 0.93}
      ]
    }
  ]
}
```

Notes:
- Keep schema minimal at first.
- Add optional fields only when justified.

## UI Plan
Add a new control in the sync panel:
- Label: `Timestamp Precision`
- Options:
  - `Line (recommended)`
  - `Word (high precision, experimental)`

Behavior:
- default to `line`
- send selected value as `timestamp_mode` in FormData
- keep existing controls unchanged

## Logging and Diagnostics
Add structured logs for each job:
- selected `timestamp_mode`
- transcription word count
- word match coverage ratio
- unique timestamp ratio
- whether fallback path was used

This will make tuning and debugging straightforward.

## Testing Plan
### Unit Tests
- word matcher monotonicity
- repeated word disambiguation
- marker handling
- interpolation/fallback behavior

### Integration Tests
- `/sync` default mode output unchanged
- `/sync` with `timestamp_mode=word` includes `.words.json`
- `SYLT` written in word granularity for word mode
- emoji/unicode handling remains stable

### Regression Tests
- keep a few known song+lyrics fixtures
- compare output quality metrics (coverage, order, monotonicity)

## Rollout Strategy
1. Backend support + tests.
2. Add UI toggle (default remains `line`).
3. Validate on real tracks.
4. Document mode behavior and caveats.

## Non-Goals (initial iteration)
- No immediate switch of default mode to `word`.
- No dependency on player support for inline word LRC.
- No major architectural rewrite.

## Success Criteria
- Line mode remains stable and unchanged for existing users.
- Word mode produces monotonic, high-coverage word timestamps.
- Sidecar JSON is usable by downstream clients.
- No increase in hard failures for difficult audio.