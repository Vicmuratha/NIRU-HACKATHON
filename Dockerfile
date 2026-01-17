# Use Python 3.9 slim for smaller image size
FROM python:3.9-slim

# Install system dependencies required for OpenCV, librosa, and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /code
WORKDIR /code

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the code
COPY . .

# Create uploads folder with proper permissions
RUN mkdir -p /code/uploads && chmod 777 /code/uploads

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Health check to ensure the app is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Start the Flask app using Gunicorn with optimized settings for HF Spaces
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "app:app"]
