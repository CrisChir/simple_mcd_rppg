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
    echo "⚠️  Stage 1 model NOT found at $MODEL_DIR/spatiotemporal_physnet_best_shuffle.pth"
    echo "   The application will still start, but Stage 1 predictions will fail."
fi

if [ -f "$MODEL_DIR/clinical_vitals_stage2_best.pth" ]; then
    echo "✅ Stage 2 model found"
else
    echo "⚠️  Stage 2 model NOT found at $MODEL_DIR/clinical_vitals_stage2_best.pth"
    echo "   The application will still start, but Stage 2 predictions will fail."
fi

if [ -f "$FACE_MODEL" ]; then
    echo "✅ MediaPipe face landmarker model found"
else
    echo "⚠️  MediaPipe face landmarker model NOT found at $FACE_MODEL"
    # Remove a stray directory that a bad build context may have copied in
    # ('/app/face_landmarker.task: Is a directory' would otherwise kill wget)
    rm -rf "$FACE_MODEL"
    echo "   Attempting to download..."
    
    # Try to download the model
    wget -q --show-progress -O /app/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    
    if [ -f "$FACE_MODEL" ]; then
        echo "✅ MediaPipe face landmarker model downloaded successfully"
    else
        echo "❌ Failed to download MediaPipe face landmarker model"
        echo "   Please download manually from:"
        echo "   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        echo "   and mount it: -v ./face_landmarker.task:/app/face_landmarker.task:ro"
    fi
fi

echo ""
echo "Starting Gradio application..."
echo "Access the application at: http://localhost:7860"
echo ""

# Run the Gradio application
# Note: We use exec to replace the current process with the Python process.
# The Dockerfile copies the build context (sliding_window_model/) into /app,
# so the script lives at /app/gradio_live_clinical_diagnostic.py.
exec python /app/gradio_live_clinical_diagnostic.py
