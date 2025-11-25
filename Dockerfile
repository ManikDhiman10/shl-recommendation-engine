# Use official slim Python image
FROM python:3.11-slim

LABEL maintainer="you"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /app

# Install OS-level deps required by faiss, sentence-transformers, torch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libopenblas-dev \
    libomp-dev \
    ffmpeg \
    unzip \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install torch from PyTorch CPU index first (avoids heavy build fallback),
# then install the rest of requirements.
# If you want GPU support, replace the torch install line with the appropriate CUDA wheel.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY . /app

# Ensure correct folders exist (vector_store and data can be mounted or baked-in)
RUN mkdir -p /app/vector_store /app/data /app/static \
    && chown -R root:root /app

# Expose port
EXPOSE ${PORT}

# Healthcheck: allow more startup time because loading indices + models can be slow
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
  CMD wget -q -O - http://127.0.0.1:${PORT}/health || exit 1

# Use uvicorn to run FastAPI; keep one worker to avoid model-loading concurrency issues.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
