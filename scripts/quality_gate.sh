#!/usr/bin/env bash
# Mandatory quality gate (cron + local): health → fast pytest → optional live MP3 sync.
# Do not pipe live_sync_quality_check.py through tee — it breaks exit status capture.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_URL="${LYRICS_SYNC_BASE_URL:-http://127.0.0.1:8005}"
echo "health: $BASE_URL/health"
curl -sf "$BASE_URL/health" >/dev/null

echo "pytest: fast unit + smoke + cleanup + async_jobs"
pytest tests/test_sync_quality_unit.py tests/test_api_smoke.py tests/test_cleanup.py tests/test_async_jobs.py -q

if [[ "${SKIP_LIVE_SYNC:-}" == "1" ]]; then
  echo "live sync: skipped (SKIP_LIVE_SYNC=1)"
  exit 0
fi

TRACK_ID="${LIVE_SYNC_TRACK_ID:-c7721ca1-e8d2-4045-8a5b-e53cfb29e7d2}"
echo "live sync: LIVE_SYNC_TRACK_ID=$TRACK_ID"
LIVE_SYNC_TRACK_ID="$TRACK_ID" python3 scripts/live_sync_quality_check.py