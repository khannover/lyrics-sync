1) [COMPLETED]
    Add rate limiting to the /sync and /sync/mp3-only endpoints using slowapi.
    Limit each IP to 5 requests per hour. Return a proper 429 response with a
    message when the limit is exceeded. Install slowapi, add the Limiter middleware
    to the FastAPI app, and decorate both POST endpoints with @limiter.limit("5/hour").

2) [COMPLETED]
    Add a job queue using asyncio.Semaphore to limit concurrent Whisper alignment
    jobs to 2 at a time. All additional requests should wait in line rather than
    being rejected. Add a GET /queue endpoint that returns the current number of
    waiting jobs and the total semaphore slots.

3) [COMPLETED]
    Add an automatic cleanup background task using FastAPI lifespan that runs every
    10 minutes and deletes job folders in /tmp/lyric-sync that are older than 1 hour.
    Use asyncio.create_task and shutil.rmtree. The task should start on app startup
    and be cancelled on shutdown.

4) [COMPLETED]
    Extend the existing GET /health endpoint to also return disk usage stats for the
    work directory: free_gb and used_gb, rounded to 2 decimal places. Use
    shutil.disk_usage().

5) [COMPLETED]
    Add a proper README.md to the repository. Include: a short description of what
    the service does, the tech stack (FastAPI, faster-whisper, DTW, ffmpeg), how to
    run it with Docker Compose, the two API endpoints (/sync and /sync/mp3-only)
    with example curl commands, and a note about the web UI at /.


