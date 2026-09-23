# ============================================================================
# Dockerfile for Clinical Vitals Diagnostic System
# ============================================================================
# This Dockerfile creates a containerized version of the Gradio Live Clinical
# Diagnostic Web App with GPU support.

# Use official PyTorch base image with CUDA support
# For CPU-only version, use: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MediaPipe (this can take a while)
RUN pip install --no-cache-dir mediapipe

# Copy application files
COPY . /app

# Copy model files (if available in build context)
# Note: For production, mount these as volumes or download at runtime
COPY face_landmarker.task /app/face_landmarker.task || echo "face_landmarker.task not found in build context"

# Create models directory
RUN mkdir -p /app/sliding_window_model/models

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["/entrypoint.sh"]
