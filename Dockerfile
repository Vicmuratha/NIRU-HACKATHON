# ═══════════════════════════════════════════════════════════
#  SafEye — Production Dockerfile
#  Multi-stage build · Non-root user · Optimised layers
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Dependencies ──
FROM python:3.11-slim AS builder

# System deps needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (prevents 5GB+ GPU download)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──
FROM python:3.11-slim AS runtime

LABEL maintainer="SafEye Team" \
      version="3.1.0" \
      description="SafEye — AI-powered deepfake and misinformation detection" \
      org.opencontainers.image.source="https://github.com/Vicmuratha/NIRU-HACKATHON" \
      org.opencontainers.image.licenses="MIT"

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    tesseract-ocr \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r safeye && useradd -r -g safeye -d /app -s /sbin/nologin safeye

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=safeye:safeye . .

# Create required directories with proper permissions
RUN mkdir -p /app/uploads /app/data /app/models \
    && chown -R safeye:safeye /app

# Copy gunicorn config
COPY --chown=safeye:safeye gunicorn.conf.py /app/gunicorn.conf.py

# Switch to non-root user
USER safeye

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["tini", "--"]

# Start with gunicorn using our config
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]