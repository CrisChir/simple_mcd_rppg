# Docker Tutorial: One ROI & 10 ROI Models

This tutorial explains how to run the two model variants of Simple MCD-RPPG in
Docker containers, how to provide the dataset (which is **not** included in
this repository and must be downloaded separately), and how to test that
everything works.

| Model | Folder | What it does |
|---|---|---|
| **One ROI model** | `single_roi_model/` | Reconstructs the PPG waveform from a single 24x24 forehead ROI (MediaPipe landmarks or precomputed `roi_forehead` crops) |
| **10 ROI model** | `sliding_window_model/` | Gradio live clinical diagnostic web app combining a 10-ROI spatiotemporal PhysNet (Stage 1) with a vitals head (Stage 2) |

---

## 1. Prerequisites

- **Docker Engine** 20.10+ and the Docker Compose plugin
  (`docker compose version`). Install via [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  or the [official apt repository](https://docs.docker.com/engine/install/).
- **NVIDIA GPU (optional)**: for GPU acceleration install the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  and verify with `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`.
  Both containers fall back to CPU automatically.
- **Git LFS (optional)**: model weights are committed directly; if you later
  move them to LFS you will need `git-lfs`.

Quick check:

```bash
docker --version
docker compose version
nvidia-smi   # only if you have an NVIDIA GPU
```

---

## 2. Get the code and the dataset

### 2.1 Clone the repository

```bash
git clone https://github.com/CrisChir/simple_mcd_rppg.git
cd simple_mcd_rppg
```

### 2.2 Download the dataset (separate step — required for training)

The preprocessed dataset is distributed via Hugging Face Datasets
([CrisChir/simple_mcd_rppg](https://huggingface.co/datasets/CrisChir/simple_mcd_rppg),
a derivative of [wengziheng/mcd_rppg](https://huggingface.co/datasets/wengziheng/mcd_rppg)).
It is **not** part of this repository.

Option A — helper script (installs `datasets` if missing):

```bash
pip install datasets
python single_roi_model/download_dataset.py          # -> ./data/preprocessed
python single_roi_model/download_dataset.py /some/dir  # custom location
```

Option B — plain Python:

```python
from datasets import load_dataset
dataset = load_dataset("CrisChir/simple_mcd_rppg")
dataset.save_to_disk("./data/preprocessed")
```

Training expects one `.npz` file per recording in the data directory, each
containing a PPG signal (key `ppg_values` or `ppg`) and precomputed ROI crops
(e.g. `roi_forehead`, shape `(T, 24, 24, 3)` for the single-ROI model, plus the
other 10 ROI keys for the 10-ROI model). If your data is raw video only, use
the inference commands instead of training.

Recommended layout:

```
simple_mcd_rppg/
├── data/
│   ├── preprocessed/     # downloaded dataset (.npz files)
│   └── videos/          # raw .avi/.mp4 videos for inference demos
```

The compose files mount `../data/preprocessed` (read-only) — this matches the
layout above.

---

## 3. One ROI model container (`single_roi_model/`)

### 3.1 Build the image

```bash
cd single_roi_model
docker build -t single-roi-rppg .
```

### 3.2 Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RPPG_DATA_DIR` | `/data/preprocessed` | Directory with `.npz` training files |
| `RPPG_MODEL_DIR` | `./models` (in image: `/app/models`) | Where checkpoints and the MediaPipe landmarker live |
| `RPPG_MODEL_SAVE_PATH` | `single_roi_forehead_trained.pth` | Where new training weights are saved |
| `CUDA_DEVICE` | `0` | GPU index; falls back to CPU if unavailable |

### 3.3 Run a quick smoke test (no dataset, no GPU needed)

The image entrypoint is the training script, so run the smoke test with an
explicit override:

```bash
docker run --rm --entrypoint python single-roi-rppg /app/smoke_test.py
```

Expected output ends with `All 5 checks passed`. The checks verify the model
forward pass, that the committed `models/single_roi_forehead_PRODUCTION.pth`
checkpoint loads, sliding-window inference on a 450-frame input, the
short-signal filter guard, and the empty-dataset error path.

### 3.4 Train (dataset required)

With Docker Compose (mounts `../data/preprocessed` and persists trained weights
in `./models`):

```bash
docker compose up single-roi-rppg train
```

Or with plain `docker run`:

```bash
docker run --rm --gpus all \
    -v $(pwd)/../data/preprocessed:/data/preprocessed:ro \
    -v $(pwd)/models:/app/models \
    single-roi-rppg train
```

Training writes the best checkpoint to `RPPG_MODEL_SAVE_PATH` (default
`/app/models/single_roi_forehead_trained.pth`, i.e. `./models/` on the host
thanks to the volume mount). If no `.npz` files are found you get a clear
`FileNotFoundError` reminding you to download the dataset and set
`RPPG_DATA_DIR`.

### 3.5 Inference on a raw video

Use the committed production checkpoint (`models/single_roi_forehead_PRODUCTION.pth`):

```bash
docker run --rm --gpus all \
    -v $(pwd)/../data/videos:/videos:ro \
    -v $(pwd)/models:/app/models \
    single-roi-rppg infer \
    --video /videos/my_video.avi \
    --model /app/models/single_roi_forehead_PRODUCTION.pth \
    --max-frames 450
```

This runs MediaPipe face landmarking per frame, crops the forehead ROI,
predicts with the 160-frame overlapping sliding window, and shows the
reconstructed waveform plus the FFT heart-rate estimate. (Inside a
headless container the matplotlib figure is not displayed; run this step
on a machine with a display, or adapt the plot to save a PNG.)

### 3.6 Run without Docker (local Python)

```bash
cd single_roi_model
pip install -r requirements.txt
export RPPG_DATA_DIR=/path/to/preprocessed
python smoke_test.py
python single_roi_forehead_physnet.py train
python single_roi_forehead_physnet.py infer --video /path/to/video.avi \
    --model models/single_roi_forehead_PRODUCTION.pth
```

---

## 4. 10 ROI model container (`sliding_window_model/`)

The 10-ROI model ships as a Gradio web app: Stage 1 is a 10-ROI (forehead,
cheeks, nose, chin, eyes, mouth, full face) spatiotemporal PhysNet producing the
rPPG signal, Stage 2 maps it to clinical vitals.

### 4.1 Files

- `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` — container setup
- `gradio_live_clinical_diagnostic.py` — the web app (port 7860)
- `models/spatiotemporal_physnet_best_shuffle.pth` — Stage 1 checkpoint
- `models/clinical_vitals_stage2_best.pth` — Stage 2 checkpoint
- `requirements.txt` — pinned dependencies

### 4.2 Build and launch

```bash
cd sliding_window_model
docker build -t clinical-vitals-app .
docker compose up
```

The entrypoint prints a pre-flight report: CUDA availability, presence of both
checkpoints, and the MediaPipe `face_landmarker.task` model (auto-downloaded if
missing). Then the app starts.

### 4.3 Use the app

Open `http://localhost:7860`. Upload or record a face video / use the webcam;
the app extracts the 10 ROIs, runs both stages, and displays the rPPG
waveform, heart rate, and vitals estimates.

### 4.4 Configuration

`docker-compose.yml` exposes these environment variables:

| Variable | Default |
|---|---|
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `0.0.0.0` / `7860` |
| `GRADIO_SHARE` | `False` |
| `MODEL_DIR` | `/app/sliding_window_model/models` |
| `FACE_MODEL_PATH` | `/app/face_landmarker.task` |
| `MAX_FRAMES` | `450` |
| `CUDA_DEVICE` | `0` |

Keep `GRADIO_SHARE=False` on untrusted networks — the public-share link is
only as safe as the network it is opened on.

### 4.5 Testing the container

**Entry point check** — a correct startup shows all three pre-flight lines
with check marks:

```bash
docker compose up -d
docker compose logs | grep -E "Stage 1|Stage 2|landmarker"
# Expected:
# ✅ Stage 1 model found
# ✅ Stage 2 model found
# ✅ MediaPipe face landmarker model found
```

**HTTP check:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860
# Expected: 200
```

**Model loading check** — run the app's import path directly to catch
weight/architecture mismatches early:

```bash
docker compose exec clinical-vitals-app python - <<'EOF'
import torch
from pathlib import Path
sd = torch.load('/app/sliding_window_model/models/spatiotemporal_physnet_best_shuffle.pth',
                map_location='cpu', weights_only=False)
print('Stage 1 keys:', len(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd))
EOF
```

**End-to-end functional test** — the definitive test is running a short
inference through the web app with a 10–15 second face video at ~30 FPS and
confirming a plausible heart rate (roughly 45–150 BPM, stable within a few
BPM across repeat uploads of the same video).

---

## 5. Side-by-side comparison

| | One ROI model | 10 ROI model |
|---|---|---|
| Interface | CLI (`train` / `infer`) | Gradio web app on port 7860 |
| ROIs | Forehead only | 10 facial regions |
| Dataset needed? | Only for `train` | No (inference only in the app) |
| GPU | Optional | Optional |
| Smoke test | `smoke_test.py` (5 checks) | Startup log + `curl` + model-load check |

---

## 6. Troubleshooting

- **`FileNotFoundError: No .npz files found in DATA_DIR=...`** — the dataset
  was not downloaded or is not mounted. Redo section 2.2 and check the
  `docker run -v` / compose volume paths.
- **`CUDA is not available, using CPU`** — either no GPU, or the NVIDIA
  Container Toolkit is missing; re-run the `nvidia/cuda:11.0-base nvidia-smi`
  check. Training on CPU is possible but slow (the single-ROI model is small;
  expect minutes per epoch rather than seconds).
- **`expected 5D input (got 3D input)`** — you are running an old version of
  `single_roi_forehead_physnet.py` with the `BatchNorm3d` bug; pull the latest
  `main`.
- **Port 7860 already in use** — change the host mapping in
  `sliding_window_model/docker-compose.yml` (e.g. `"7861:7860"`) and reload
  with `docker compose up -d --force-recreate`.
- **`face_landmarker.task` download failed at container start** — download it
  manually from the URL printed by `entrypoint.sh` and mount it:
  `-v $(pwd)/face_landmarker.task:/app/face_landmarker.task:ro`.
- **Gradio figures don't render inside the single-ROI container** — the image
  sets `MPLBACKEND=Agg` for headless operation; the inference plot is meant
  for interactive use (section 3.6).
- **Permission denied writing checkpoints** — make sure the `./models` host
  directory is writable by the container user when mounting volumes.
