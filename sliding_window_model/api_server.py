#!/usr/bin/env python3
"""Purpose-built REST API for the Clinical Vitals Diagnostic System.

Thin FastAPI wrapper around the exact same pipeline used by the Gradio app
(gradio_live_clinical_diagnostic.py): same MediaPipe landmarker, same Stage 1
/ Stage 2 checkpoints, same HR calculation. Returns clean JSON instead of
Gradio's event-stream + HTML report format.

Endpoints:
    GET  /health        -> service status
    GET  /vitals/keys   -> the vitals the model predicts, with units
    POST /predict       -> multipart upload of a face video, returns:
                           - 9 vitals as {key: {value, unit}}
                           - heart_rate_bpm (FFT of the Stage 1 rPPG waveform)
                           - ppg waveform samples (30 FPS)
                           - ppg_plot_png_base64 (graph of the waveform)
    POST /predict/path  -> same, but takes a JSON body with a path to a video
                           already present inside the container
                           {"video_path": "/app/data/foo.avi"}

Usage:
    python api_server.py

Environment Variables:
    API_HOST: Server host (default: 0.0.0.0)
    API_PORT: Server port (default: 7861)
    MODEL_DIR: Directory containing model checkpoints
    FACE_MODEL_PATH: Path to MediaPipe face landmarker model
    MAX_FRAMES: Maximum frames to process (default: 450)
    CUDA_DEVICE: CUDA device index (default: 0)

The models are loaded lazily on first prediction so the health endpoint
responds immediately after startup.
"""

import base64
import io
import os
import urllib.request

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import gradio_live_clinical_diagnostic as app


API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '7861'))

api = FastAPI(
    title="Clinical Vitals Diagnostic API",
    description="rPPG-based vital signs prediction from face video",
    version="1.0.0",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_face_landmarker():
    """Download the MediaPipe landmarker if missing (the compose service for
    this API bypasses entrypoint.sh, which normally does the download)."""
    path = app.FACE_MODEL_PATH
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        print(f"Downloading face landmarker to {path}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, path)


def load_models():
    """Load landmarker + Stage 1 + Stage 2 once, storing them on the app module."""
    ensure_face_landmarker()
    app.landmarker = app.initialize_mediapipe_landmarker()
    if app.landmarker is None:
        raise RuntimeError("MediaPipe face landmarker could not be initialized")

    app.model_stage1 = app.load_stage1_model()
    if app.model_stage1 is None:
        raise RuntimeError("Stage 1 model could not be loaded")

    model2, vital_stats, vital_keys, vital_units = app.load_stage2_model()
    if model2 is None:
        raise RuntimeError("Stage 2 model could not be loaded")

    app.model_stage2 = model2
    app.VITAL_STATS = vital_stats
    app.VITAL_KEYS = vital_keys
    app.VITAL_UNITS = vital_units


def ensure_models_loaded():
    if getattr(app, 'model_stage2', None) is None:
        load_models()


def waveform_to_base64_png(ppg_clean, frame_count, bpm):
    """Render the same rPPG waveform plot the Gradio app shows, as base64 PNG."""
    time_axis = np.arange(len(ppg_clean)) / app.SAMPLING_RATE
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, ppg_clean, color='tab:blue', linewidth=1.2)
    ax.set_title(f"rPPG BVP Waveform — FFT Heart Rate: {bpm:.1f} BPM")
    ax.set_xlabel(f"Time (s) — {app.SAMPLING_RATE:.0f} FPS, {frame_count} frames")
    ax.set_ylabel("Normalized BVP")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def run_prediction(video_path: str):
    """Shared prediction path for both /predict endpoints."""
    ensure_models_loaded()
    try:
        ppg_clean, vitals_raw, frame_count = app.predict_vitals_from_video(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if ppg_clean is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in video. Ensure a well-lit face is visible.",
        )

    bpm = app.calculate_bpm_from_fft(ppg_clean, fs=app.SAMPLING_RATE)

    vitals = {}
    for i, key in enumerate(app.VITAL_KEYS):
        vitals[key] = {
            "value": round(float(vitals_raw[i]), 2),
            "unit": app.VITAL_UNITS[i],
        }

    return {
        "heart_rate_bpm": round(float(bpm), 1),
        "heart_rate_source": "stage1_ppg_fft",
        "frames_processed": int(frame_count),
        "sampling_rate_fps": app.SAMPLING_RATE,
        "ppg_waveform": [round(float(v), 6) for v in ppg_clean],
        "ppg_plot_png_base64": waveform_to_base64_png(ppg_clean, frame_count, bpm),
        "vitals": vitals,
        "notes": "Research output, not a medical diagnosis.",
    }


@api.get("/health")
def health():
    ready = getattr(app, 'model_stage2', None) is not None
    return {"status": "ok", "models_loaded": ready}


@api.get("/vitals/keys")
def vitals_keys():
    ensure_models_loaded()
    return [
        {"key": k, "unit": u} for k, u in zip(app.VITAL_KEYS, app.VITAL_UNITS)
    ]


@api.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Upload a face video; get vitals, HR, waveform and plot as JSON."""
    os.makedirs('/tmp/api_uploads', exist_ok=True)
    dest = os.path.join('/tmp/api_uploads', os.path.basename(video.filename or 'video'))
    with open(dest, 'wb') as f:
        f.write(await video.read())
    return run_prediction(dest)


@api.post("/predict/path")
def predict_path(body: dict):
    """Predict from a video already present inside the container."""
    video_path = body.get('video_path')
    if not video_path:
        raise HTTPException(status_code=400, detail="'video_path' is required")
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"File not found: {video_path}")
    return run_prediction(video_path)


if __name__ == "__main__":
    import uvicorn

    print("=" * 44)
    print("Clinical Vitals Diagnostic REST API")
    print("=" * 44)
    print(f"Starting on {API_HOST}:{API_PORT}")
    print(f"Docs (Swagger UI): http://localhost:{API_PORT}/docs")
    uvicorn.run(api, host=API_HOST, port=API_PORT, log_level="info")
