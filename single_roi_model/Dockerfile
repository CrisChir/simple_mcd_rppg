# ============================================================================
# Dockerfile for Single ROI (Forehead) PhysNet3D model
# ============================================================================
# Usage:
#   docker build -t single-roi-rppg .
#   docker run --rm -v /path/to/preprocessed_data:/data \
#       -v /path/to/videos:/videos single-roi-rppg smoke-test
# See DOCKER_TUTORIAL.md (repo root) for the full tutorial.

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY single_roi_forehead_physnet.py /app/single_roi_forehead_physnet.py
COPY smoke_test.py /app/smoke_test.py

# Committed weights are copied in; override by mounting a volume to /app/models
COPY models /app/models

ENV RPPG_DATA_DIR=/data/preprocessed
ENV RPPG_MODEL_DIR=/app/models
ENV RPPG_MODEL_SAVE_PATH=/app/models/single_roi_forehead_trained.pth
ENV CUDA_DEVICE=0

# Default: show help. Override the command via `docker run ... <args>`
ENTRYPOINT ["python", "/app/single_roi_forehead_physnet.py"]
CMD ["--help"]
