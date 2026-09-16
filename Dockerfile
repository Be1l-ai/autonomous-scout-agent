FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# build-essential + cmake are required: llama-cpp-python is compiled from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as uid 1000; everything we write must be owned by it.
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .

# Always build llama-cpp-python from source rather than using a prebuilt wheel.
# The abetlen.github.io CPU wheel index has served musl-linked builds (Alpine
# ABI) that segfault-on-import here, because this image is glibc-based
# (python:3.11-slim = Debian). Compiling locally guarantees the ABI matches.
RUN pip install --upgrade pip
RUN CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_BLAS=OFF" \
    pip install --no-cache-dir llama-cpp-python==0.3.1
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the weights into the image so the Space doesn't time out on first boot.
RUN mkdir -p models data .cache && python download_model.py

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# One worker only: the SQLite connection and the llama.cpp context are per-process.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]