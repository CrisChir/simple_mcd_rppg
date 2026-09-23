# Single ROI Model

This folder contains the **single region (single ROI) rPPG model** — a variant of the
Simple MCD-RPPG pipeline that estimates the blood volume pulse (BVP) signal and heart
rate from a single region of interest, instead of the multi-ROI setup used in
`sliding_window_model/` and `deeper_model/`.

## Contents

- `single_roi_forehead_physnet.py` — training and inference code for the single-ROI (forehead) PhysNet3D model
- `models/` — trained model weights (see `models/README.md`), including
  `single_roi_forehead_PRODUCTION.pth`
- `smoke_test.py` — dependency-free smoke tests (no dataset or GPU needed)
- `download_dataset.py` — helper to download the preprocessed dataset from Hugging Face
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` — container setup

## Overview

The model extracts a 24x24 forehead ROI (via MediaPipe face landmarks for raw video,
or precomputed `roi_forehead` crops from `.npz` files) and reconstructs the PPG
waveform with an edge-safe spatiotemporal network. Two key fixes over a naive
implementation:

1. **Replicate padding** in the temporal encoder, preventing edge amplitude collapse.
2. **Overlapping sliding-window inference** (160-frame chunks, 50% overlap, Hann
   blending), keeping amplitude consistent for videos of any length.

## Dataset (downloaded separately — required for training)

The preprocessed dataset is **not** included in this repository. Download it from
Hugging Face:

```bash
pip install datasets
python download_dataset.py              # -> ../data/preprocessed
python download_dataset.py /some/dir    # custom location
```

Training expects one `.npz` file per recording containing a PPG signal
(key `ppg_values` or `ppg`) and a precomputed forehead crop
(`roi_forehead`, shape `(T, 24, 24, 3)`). If no `.npz` files are found,
training exits with a clear `FileNotFoundError`.

## Usage

```bash
pip install -r requirements.txt

# smoke test (no dataset or GPU required)
python smoke_test.py

# training
export RPPG_DATA_DIR=/path/to/preprocessed   # default: ./data/preprocessed
python single_roi_forehead_physnet.py train

# video inference (uses models/single_roi_forehead_PRODUCTION.pth)
python single_roi_forehead_physnet.py infer \
    --video /path/to/video.avi \
    --model models/single_roi_forehead_PRODUCTION.pth \
    --max-frames 450
```

Programmatic inference:

```python
from single_roi_forehead_physnet import run_video_inference_demo
run_video_inference_demo("path/to/video.avi", model_path="models/single_roi_forehead_PRODUCTION.pth", max_frames=450)
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RPPG_DATA_DIR` | `./data/preprocessed` | Directory with `.npz` training files |
| `RPPG_MODEL_DIR` | `./models` | Checkpoints directory + MediaPipe landmarker download location |
| `RPPG_MODEL_SAVE_PATH` | `single_roi_forehead_trained.pth` | Where training saves the best checkpoint |
| `CUDA_DEVICE` | `0` | GPU index; falls back to CPU if unavailable |

## Docker

```bash
docker build -t single-roi-rppg .
docker run --rm --entrypoint python single-roi-rppg /app/smoke_test.py   # smoke test
docker run --rm --gpus all \
    -v $(pwd)/../data/preprocessed:/data/preprocessed:ro \
    -v $(pwd)/models:/app/models \
    single-roi-rppg train
```

See [`../DOCKER_TUTORIAL.md`](../DOCKER_TUTORIAL.md) for the full container
tutorial (both single-ROI and 10-ROI models) and testing instructions.

## Related folders

- `../sliding_window_model/` — sliding window, multi-ROI spatiotemporal PhysNet
- `../deeper_model/` — deeper multi-ROI spatiotemporal PhysNet
