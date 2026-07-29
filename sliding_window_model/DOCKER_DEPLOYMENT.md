# Docker Container Deployment Guide
# Clinical Vitals Diagnostic System

This guide provides comprehensive instructions for containerizing and deploying the Gradio Live Clinical Diagnostic Web App using Docker.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dockerfile](#dockerfile)
4. [Docker Compose](#docker-compose)
5. [Building the Image](#building-the-image)
6. [Running the Container](#running-the-container)
7. [Configuration Options](#configuration-options)
8. [Deployment Scenarios](#deployment-scenarios)
9. [Troubleshooting](#troubleshooting)
10. [Security Considerations](#security-considerations)
11. [Performance Optimization](#performance-optimization)

---

## Overview

This Docker deployment allows you to run the Clinical Vitals Diagnostic System in a containerized environment, ensuring consistent performance across different systems and making deployment easier.

### Key Features

- **Isolated Environment**: All dependencies are contained within the Docker image
- **Portable**: Run on any system with Docker installed
- **Reproducible**: Consistent environment across development, testing, and production
- **Scalable**: Easy to deploy multiple instances or scale horizontally

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                            │
├─────────────────────────────────────────────────────────────┤
│  Gradio Web Server (Port 7860)                                │
│  ├─ MediaPipe FaceLandmarker Task                            │
│  ├─ PyTorch Models (Stage 1 & Stage 2)                       │
│  ├─ OpenCV for Video Processing                               │
│  └─ SciPy for Signal Processing                               │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Host System                                 │
│  ├─ Docker Engine                                              │
│  ├─ GPU (Optional for CUDA acceleration)                      │
│  └─ Port Mapping (7860:7860)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Docker Installation

Ensure Docker is installed on your system:

**Linux (Ubuntu/Debian):**
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
sudo docker run hello-world
```

**macOS:**
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Windows:**
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. NVIDIA Container Toolkit (For GPU Support)

If you want to use GPU acceleration with CUDA:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU support
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 3. Project Structure

Ensure your project directory has the following structure:

```
clinical-vitals-app/
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── sliding_window_model/
│   ├── gradio_live_clinical_diagnostic.ipynb  # Gradio notebook
│   ├── models/
│   │   ├── spatiotemporal_physnet_best_shuffle.pth
│   │   └── clinical_vitals_stage2_best.pth
│   └── Dockerfile               # Optional: Model-specific Dockerfile
├── face_landmarker.task         # MediaPipe model
└── README.md
```

---

## Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
# ============================================================================
# Dockerfile for Clinical Vitals Diagnostic System
# ============================================================================

# Use official PyTorch base image with CUDA support
# For CPU-only version, use: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install MediaPipe dependencies
RUN apt-get install -y --no-install-recommends \
    libmediapipe-dev \
    || echo "MediaPipe dev package not available, will install from source"

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

# Copy model files
COPY face_landmarker.task /app/face_landmarker.task
COPY sliding_window_model/models/ /app/sliding_window_model/models/

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["/entrypoint.sh"]
```

### CPU-Only Dockerfile

For systems without NVIDIA GPU:

```dockerfile
# Use official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MediaPipe
RUN pip install --no-cache-dir mediapipe

# Copy application files
COPY . /app

# Copy model files
COPY face_landmarker.task /app/face_landmarker.task
COPY sliding_window_model/models/ /app/sliding_window_model/models/

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["/entrypoint.sh"]
```

---

## docker-compose.yml

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  clinical-vitals-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: clinical-vitals-app
    restart: unless-stopped
    ports:
      - "7860:7860"
    volumes:
      # Mount model files (optional, for development)
      - ./face_landmarker.task:/app/face_landmarker.task:ro
      - ./sliding_window_model/models:/app/sliding_window_model/models:ro
      # Mount data directory for persistent storage
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # For CPU-only mode, remove the deploy section and add:
    # devices: []
    
  # Optional: Nginx reverse proxy for production
  nginx:
    image: nginx:alpine
    container_name: clinical-vitals-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - clinical-vitals-app
```

---

## requirements.txt

Create a `requirements.txt` file:

```text
# Core dependencies
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.2
opencv-python==4.7.0.72
Pillow==9.5.0

# MediaPipe
mediapipe==0.10.2

# Gradio
gradio==3.44.2

# Utilities
pathlib2==2.3.7
scikit-learn==1.3.0
seaborn==0.12.2
pandas==2.0.3

# For notebook support (optional)
ipython==8.12.0
jupyter==1.0.0
nbconvert==7.4.0
```

---

## entrypoint.sh

Create an `entrypoint.sh` script:

```bash
#!/bin/bash

# ============================================================================
# Entrypoint script for Clinical Vitals Diagnostic System
# ============================================================================

set -e

echo "=========================================="
echo "Clinical Vitals Diagnostic System"
echo "=========================================="
echo ""

# Check if GPU is available
echo "Checking CUDA availability..."
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA is available"
    python -c "import torch; print('CUDA device count:', torch.cuda.device_count())"
else
    echo "⚠️  CUDA is not available, using CPU"
fi

# Check model files
echo ""
echo "Checking model files..."
MODEL_DIR="/app/sliding_window_model/models"
FACE_MODEL="/app/face_landmarker.task"

if [ -f "$MODEL_DIR/spatiotemporal_physnet_best_shuffle.pth" ]; then
    echo "✅ Stage 1 model found"
else
    echo "❌ Stage 1 model NOT found at $MODEL_DIR/spatiotemporal_physnet_best_shuffle.pth"
fi

if [ -f "$MODEL_DIR/clinical_vitals_stage2_best.pth" ]; then
    echo "✅ Stage 2 model found"
else
    echo "❌ Stage 2 model NOT found at $MODEL_DIR/clinical_vitals_stage2_best.pth"
fi

if [ -f "$FACE_MODEL" ]; then
    echo "✅ MediaPipe face landmarker model found"
else
    echo "❌ MediaPipe face landmarker model NOT found at $FACE_MODEL"
fi

echo ""
echo "Starting Gradio application..."
echo "Access the application at: http://localhost:7860"
echo ""

# Run the Gradio application
# Note: We use exec to replace the current process with the Python process
exec python /app/sliding_window_model/gradio_live_clinical_diagnostic.py
```

---

## Building the Image

### Build the Docker Image

```bash
# Navigate to your project directory
cd /path/to/clinical-vitals-app

# Build the Docker image
docker build -t clinical-vitals-app:latest .

# For CPU-only version
docker build -t clinical-vitals-app:cpu -f Dockerfile.cpu .
```

### Build with Docker Compose

```bash
# Build using docker-compose
docker-compose build

# Build with no cache (clean build)
docker-compose build --no-cache
```

### Verify the Image

```bash
# List Docker images
docker images

# Inspect the image
docker inspect clinical-vitals-app:latest

# Check image size
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}"
```

---

## Running the Container

### Run with Docker

```bash
# Basic run (with GPU support)
docker run --gpus all -p 7860:7860 --name clinical-vitals-app clinical-vitals-app:latest

# Run with CPU only
docker run -p 7860:7860 --name clinical-vitals-app clinical-vitals-app:cpu

# Run with volume mounts (for development)
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/face_landmarker.task:/app/face_landmarker.task:ro \
  -v $(pwd)/sliding_window_model/models:/app/sliding_window_model/models:ro \
  -v $(pwd)/data:/app/data \
  --name clinical-vitals-app clinical-vitals-app:latest

# Run in detached mode (background)
docker run --gpus all -d -p 7860:7860 --name clinical-vitals-app clinical-vitals-app:latest

# View logs
docker logs clinical-vitals-app

# Follow logs in real-time
docker logs -f clinical-vitals-app
```

### Run with Docker Compose

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Stop and remove containers, networks
docker-compose down -v
```

### Access the Application

After starting the container, access the application at:
- **Local**: http://localhost:7860
- **Network**: http://<your-server-ip>:7860

---

## Configuration Options

### Environment Variables

You can configure the application using environment variables:

```bash
# Set environment variables in Docker run
docker run --gpus all -p 7860:7860 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7860 \
  -e GRADIO_SHARE=False \
  -e MODEL_DIR=/app/sliding_window_model/models \
  -e FACE_MODEL_PATH=/app/face_landmarker.task \
  --name clinical-vitals-app clinical-vitals-app:latest
```

### Common Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server host address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `GRADIO_SHARE` | `False` | Enable public sharing |
| `MAX_FRAMES` | `450` | Maximum frames to process |
| `SAMPLING_RATE` | `30.0` | Video sampling rate (FPS) |
| `CUDA_DEVICE` | `0` | CUDA device index |

### Custom Configuration File

Create a `.env` file for Docker Compose:

```env
# Gradio Configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False

# Model Paths
MODEL_DIR=/app/sliding_window_model/models
FACE_MODEL_PATH=/app/face_landmarker.task

# Processing Parameters
MAX_FRAMES=450
SAMPLING_RATE=30.0

# CUDA Configuration
CUDA_DEVICE=0
```

---

## Deployment Scenarios

### 1. Local Development

For development and testing on your local machine:

```bash
# Build and run
docker-compose -f docker-compose.dev.yml up --build

# Or use the main docker-compose.yml
docker-compose up --build
```

**docker-compose.dev.yml:**
```yaml
version: '3.8'

services:
  clinical-vitals-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: clinical-vitals-app-dev
    ports:
      - "7860:7860"
    volumes:
      - ./:/app:rw
      - /tmp/.X11-unix:/tmp/.X11-unix:ro
    environment:
      - DISPLAY=${DISPLAY}
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Production Deployment

For production deployment with Nginx reverse proxy:

```bash
# Start all services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  clinical-vitals-app:
    restart: always
    environment:
      - GRADIO_SHARE=False
      - GRADIO_SERVER_NAME=0.0.0.0
    
  nginx:
    image: nginx:alpine
    container_name: clinical-vitals-nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - clinical-vitals-app
```

**nginx.conf:**
```nginx
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    upstream gradio {
        server clinical-vitals-app:7860;
    }
    
    server {
        listen 80;
        server_name localhost;
        return 301 https://$host$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        
        location / {
            proxy_pass http://gradio;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Gradio
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### 3. Cloud Deployment (AWS, GCP, Azure)

#### AWS ECS

1. Push your Docker image to ECR:
```bash
# Authenticate with ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Create repository
aws ecr create-repository --repository-name clinical-vitals-app

# Tag and push image
docker tag clinical-vitals-app:latest <account-id>.dkr.ecr.<region>.amazonaws.com/clinical-vitals-app:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/clinical-vitals-app:latest
```

2. Create ECS task definition and service

#### Google Cloud Run

```bash
# Push to Google Container Registry
gcloud auth configure-docker
docker tag clinical-vitals-app:latest gcr.io/<project-id>/clinical-vitals-app:latest
docker push gcr.io/<project-id>/clinical-vitals-app:latest

# Deploy to Cloud Run
gcloud run deploy clinical-vitals-app \
  --image gcr.io/<project-id>/clinical-vitals-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 7860 \
  --memory 4Gi \
  --cpu 2
```

#### Azure Container Instances

```bash
# Push to Azure Container Registry
az acr login --name <registry-name>
docker tag clinical-vitals-app:latest <registry-name>.azurecr.io/clinical-vitals-app:latest
docker push <registry-name>.azurecr.io/clinical-vitals-app:latest

# Deploy to Azure Container Instances
az container create \
  --resource-group <resource-group> \
  --name clinical-vitals-app \
  --image <registry-name>.azurecr.io/clinical-vitals-app:latest \
  --cpu 2 \
  --memory 4 \
  --ports 7860 \
  --dns-name-label clinical-vitals-app \
  --restart-policy Always
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Not Available

**Error:** `CUDA is not available`

**Solution:**
- Ensure you have NVIDIA drivers installed
- Install NVIDIA Container Toolkit
- Use `--gpus all` flag when running the container
- Verify with: `nvidia-smi`

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 2. MediaPipe Model Not Found

**Error:** `MediaPipe model not found`

**Solution:**
- Ensure `face_landmarker.task` is in the correct location
- Check volume mounts
- Download the model from MediaPipe's repository

```bash
# Download MediaPipe face landmarker model
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

#### 3. Model Checkpoints Not Found

**Error:** `Stage 1/2 model not found`

**Solution:**
- Ensure model files are in the correct directory
- Check volume mounts
- Verify file permissions

```bash
# Check if models exist in container
docker exec -it clinical-vitals-app ls -la /app/sliding_window_model/models/
```

#### 4. Port Already in Use

**Error:** `port is already allocated`

**Solution:**
- Find and kill the process using the port
- Use a different port

```bash
# Find process using port 7860
sudo lsof -i :7860

# Kill the process
kill -9 <PID>

# Or use a different port
docker run -p 7861:7860 ...
```

#### 5. Out of Memory

**Error:** `Out of memory`

**Solution:**
- Increase Docker memory allocation
- Use a smaller batch size
- Use CPU-only mode

```bash
# Increase Docker memory (in Docker Desktop settings)
# Or limit memory usage
docker run --memory=4g ...
```

#### 6. OpenCV Errors

**Error:** `libGL.so.1: cannot open shared object file`

**Solution:**
- Install required system libraries
- Use the correct base image

```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6
```

#### 7. Gradio Connection Issues

**Error:** `Connection refused` or timeout

**Solution:**
- Check if container is running
- Verify port mapping
- Check firewall settings

```bash
# Check running containers
docker ps

# Check container logs
docker logs clinical-vitals-app

# Test connection from host
curl http://localhost:7860
```

### Debugging Commands

```bash
# Enter the container for debugging
docker exec -it clinical-vitals-app /bin/bash

# Check Python environment
python -c "import torch; print(torch.cuda.is_available())"

# Check MediaPipe
python -c "import mediapipe; print(mediapipe.__version__)"

# Check model files
ls -la /app/sliding_window_model/models/

# Test model loading
python -c "
import torch
from sliding_window_model.gradio_live_clinical_diagnostic import SpatioTemporalPhysNet
model = SpatioTemporalPhysNet()
print('Model loaded successfully')
"
```

---

## Security Considerations

### 1. Authentication

Add authentication to your Gradio application:

```python
# In your Gradio app
import gradio as gr

auth = gr.Auth(
    username="admin",
    password="securepassword"
)

with gr.Blocks() as demo:
    # Your app configuration
    pass

demo.launch(auth=auth)
```

### 2. HTTPS

Always use HTTPS in production:
- Use Nginx reverse proxy with SSL
- Use Let's Encrypt for free SSL certificates
- Never expose Gradio directly on port 80

### 3. Network Security

- Use firewall rules to restrict access
- Consider using a VPN for internal access
- Limit exposed ports

```bash
# Example: Restrict Docker to specific IP
docker run -p 192.168.1.100:7860:7860 ...
```

### 4. Data Privacy

- Process videos in memory (don't save to disk)
- Implement data retention policies
- Comply with GDPR/HIPAA if handling medical data

### 5. Container Security

- Use non-root user in container
- Regularly update base images
- Scan for vulnerabilities

```dockerfile
# Use non-root user
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app
```

---

## Performance Optimization

### 1. GPU Acceleration

- Use CUDA-enabled Docker images
- Ensure proper GPU driver installation
- Monitor GPU usage

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU usage in container
docker stats clinical-vitals-app
```

### 2. Model Optimization

- Use TorchScript for model optimization
- Quantize models for faster inference
- Use smaller models if appropriate

```python
# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)
```

### 3. Docker Optimization

- Use multi-stage builds to reduce image size
- Clean up cache and temporary files
- Use Alpine-based images when possible

```dockerfile
# Multi-stage build example
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Build your application
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# Final stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Copy only necessary files
COPY --from=builder /app /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app
CMD ["python", "app.py"]
```

### 4. Resource Limits

Set appropriate resource limits:

```bash
# Limit CPU and memory
docker run --cpus=2 --memory=4g ...

# In docker-compose
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 5. Caching

- Use Docker layer caching
- Cache pip installations
- Cache model downloads

```dockerfile
# Cache pip installations
RUN pip install --cache-dir=/tmp/pip-cache -r requirements.txt
```

---

## Monitoring and Logging

### 1. Docker Logging

```bash
# View container logs
docker logs clinical-vitals-app

# Follow logs in real-time
docker logs -f clinical-vitals-app

# View logs with timestamps
docker logs -t clinical-vitals-app

# View last N lines
docker logs --tail=100 clinical-vitals-app
```

### 2. Resource Monitoring

```bash
# View container resource usage
docker stats clinical-vitals-app

# View detailed resource usage
docker stats --no-stream clinical-vitals-app
```

### 3. Health Checks

Add health checks to your Dockerfile:

```dockerfile
# Add health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:7860 || exit 1
```

### 4. Prometheus Metrics (Optional)

For advanced monitoring, integrate Prometheus:

```python
# In your application
from prometheus_client import start_http_server

start_http_server(8000)
```

---

## Updates and Maintenance

### 1. Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild Docker image
docker-compose build --no-cache

# Restart services
docker-compose down && docker-compose up -d
```

### 2. Updating Dependencies

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Rebuild Docker image
docker-compose build --no-cache
```

### 3. Cleaning Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove everything (careful!)
docker system prune -a
```

---

## Conclusion

This Docker deployment guide provides everything you need to containerize and deploy the Clinical Vitals Diagnostic System. The containerized approach ensures:

- **Consistency**: Same environment across development, testing, and production
- **Portability**: Run on any system with Docker installed
- **Isolation**: Dependencies don't conflict with host system
- **Scalability**: Easy to deploy multiple instances
- **Maintainability**: Simple updates and rollbacks

For production deployments, consider:
- Adding proper authentication
- Implementing HTTPS
- Setting up monitoring and logging
- Implementing backup strategies
- Following security best practices

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Gradio Documentation](https://gradio-app.github.io/gradio/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)

---

*Last updated: 2024*
*Version: 1.0*
