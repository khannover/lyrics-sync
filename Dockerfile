# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

# Install build-essential and python3-dev to provide stdlib.h and other headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a virtual environment to make copying to the next stage easier
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim

# Install ONLY the runtime dependencies (ffmpeg, libsndfile1, libchromaprint-tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-compiled dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Ensure the app uses the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code and models
COPY app/ ./app/
RUN mkdir -p /app/models
COPY models/ /app/models/

# Pre-download models at build time
ARG MODEL_SIZE=small
ENV MODEL_SIZE=${MODEL_SIZE}
RUN python -c "from app.alignment import ensure_model; ensure_model()"
RUN python -c "from app.genre_classifier import ensure_genre_model; ensure_genre_model()"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
