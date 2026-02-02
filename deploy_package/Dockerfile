# Use Python 3.9 slim (Good choice for compatibility)
FROM python:3.9-slim

# Install system dependencies
# Added 'ffmpeg' which is crucial for librosa/audio to work on Linux
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# --- CRITICAL AZURE OPTIMIZATION START ---
# We install the CPU-only versions of Torch/Tensorflow MANUALLY here.
# This prevents downloading the 5GB+ GPU versions that crash the build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tensorflow-cpu
# -----------------------------------------

COPY requirements.txt /code/requirements.txt

# Install the rest of the dependencies
# Note: Ensure 'torch' and 'tensorflow' are REMOVED from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY . .

# Create uploads folder with permissions
RUN mkdir -p /code/uploads && chmod 777 /code/uploads

# Expose Port 8000 (Standard for Azure)
EXPOSE 8000

# Health Check (Adjusted for Port 8000)
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start Command
# Changed Port to 8000
# Added --timeout 600 (10 minutes) to allow AI models to load without crashing
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "600", "app:app"]