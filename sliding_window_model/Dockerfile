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
# Non-interactive package installs (prevents tzdata region/city prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
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

# Install Python dependencies directly into the base environment.
# The base image already ships PyTorch 2.0.1 + CUDA runtime; the previous
# venv layer forced re-downloading the full ~2 GB torch stack.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MediaPipe (this can take a while)
RUN pip install --no-cache-dir mediapipe

# Copy application files
COPY . /app

# Note: face_landmarker.task is not copied at build time; entrypoint.sh
# downloads it at runtime if missing (or mount it as a volume).

# Create models directory
RUN mkdir -p /app/sliding_window_model/models

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["/entrypoint.sh"]
