# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Similar Question Finder API
# ─────────────────────────────────────────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder): install Python deps in isolation
#   Stage 2 (runtime): minimal image with only what's needed to run
#
# WHY CUDA base image?
#   sentence-transformers and faiss-gpu both need CUDA runtime libraries.
#   The nvidia/cuda image provides these without us needing to install them.
#
# To build:
#   docker build -t similar-questions-api .
#
# To run (with GPU):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     similar-questions-api
#
# To run (CPU only):
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data similar-questions-api
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

# Set non-interactive mode for apt to prevent prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.1 support FIRST
# (must be done before requirements.txt to use the right CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
COPY requirements.txt /tmp/requirements.txt
# Remove faiss-gpu line and install separately to handle CUDA version matching
RUN grep -v "^faiss" /tmp/requirements.txt > /tmp/req_no_faiss.txt && \
    pip install --no-cache-dir -r /tmp/req_no_faiss.txt && \
    pip install --no-cache-dir faiss-gpu==1.7.4


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell sentence-transformers to cache models inside the container's data volume
ENV TRANSFORMERS_CACHE=/app/data/models/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/models/st_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy only the application code (not raw data or training artifacts)
COPY api/       /app/api/
COPY src/       /app/src/
COPY config.yaml /app/config.yaml

# data/ is mounted as a volume at runtime — do NOT bake model/index into image
# This keeps the image small and allows updating the index without rebuilding.
VOLUME ["/app/data"]

EXPOSE 8000

# Health check: Docker will mark container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with single worker in container (scale with replicas, not workers)
# WHY single worker? Each worker loads the full model into GPU memory.
# With a 420MB model, 4 workers would need ~1.7GB VRAM just for model weights.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
