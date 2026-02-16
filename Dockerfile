FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify
RUN python -c "\
from faster_whisper import WhisperModel; \
print('faster-whisper OK'); \
"

COPY app/ ./app/
RUN mkdir -p /app/models

# Pre-download model at build time
RUN python -c "from app.alignment import ensure_model; ensure_model()"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

