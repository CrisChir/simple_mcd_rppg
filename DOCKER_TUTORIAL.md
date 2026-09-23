# Docker Tutorial: One ROI & 10 ROI Models

This tutorial explains how to run the two model variants of Simple MCD-RPPG in
Docker containers, how to provide the dataset (which is **not** included in
this repository and must be downloaded separately), and how to test that
everything works. It covers **Linux, macOS (Intel and Apple Silicon), and
Windows 10/11** — per-OS setup is in section 1, and commands with per-OS
variants are shown throughout.

| Model | Folder | What it does |
|---|---|---|
| **One ROI model** | `single_roi_model/` | Reconstructs the PPG waveform from a single 24x24 forehead ROI (MediaPipe landmarks or precomputed `roi_forehead` crops) |
| **10 ROI model** | `sliding_window_model/` | Gradio live clinical diagnostic web app combining a 10-ROI spatiotemporal PhysNet (Stage 1) with a vitals head (Stage 2) |

---

## 1. Prerequisites

OS support at a glance:

| | Linux | macOS | Windows 10/11 |
|---|---|---|---|
| Docker install | apt repo (1.1) | Docker Desktop / Homebrew + colima (1.2) | Docker Desktop + WSL 2 (1.3) |
| GPU (CUDA) | Yes (NVIDIA + toolkit) | No — CPU only | Yes (NVIDIA via WSL 2) |
| Shell examples | bash (as-is) | zsh/bash (as-is) | PowerShell or WSL bash (1.4) |
| Apple Silicon | — | `--platform linux/amd64` (Rosetta) | — |

- **Docker Engine** 20.10+ and the Docker Compose plugin (`docker compose version`).
- **NVIDIA GPU (optional)**: both containers run on CPU and fall back to it
  automatically; GPU support differs per OS (see the table above and 1.1–1.3).
- **Git LFS (optional)**: model weights are committed directly; if you later
  move them to LFS you will need `git-lfs`.

### 1.1 Linux (Ubuntu/Debian)

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key and repository
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running docker without sudo (log out/in afterwards)
sudo usermod -aG docker $USER

# Verify
docker --version
sudo docker run hello-world
```

**GPU (NVIDIA only)**: install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
restart Docker, then verify:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

AMD/Intel GPUs are not supported by these containers — they run on CPU.

### 1.2 macOS (Intel and Apple Silicon)

```bash
# Option A: Docker Desktop (GUI) — download from
# https://www.docker.com/products/docker-desktop/
# or via Homebrew: brew install --cask docker
open -a Docker   # start Docker Desktop once installed

# Option B: CLI-only via Homebrew + colima (lightweight daemon)
# brew install docker docker-compose colima
# colima start

# Verify
docker --version
docker compose version
docker run hello-world
```

**GPU**: macOS has no CUDA support, so both containers always run on **CPU**.
This works fine for inference and the smoke tests; training is slower — for
serious training use a Linux/Windows machine with an NVIDIA GPU.

**Apple Silicon note**: the images are built from `pytorch/pytorch:...-runtime`
(x86-64). On M1/M2/M3 Macs Docker Desktop runs them under Rosetta 2
emulation. If the build or run fails with a platform error, force the
platform explicitly:

```bash
docker build --platform linux/amd64 -t single-roi-rppg .
docker run --platform linux/amd64 --rm single-roi-rppg --help
```

### 1.3 Windows (10/11)

Install **Docker Desktop for Windows**, which requires the WSL 2 backend:

```powershell
# Enable WSL 2 (if not already installed; reboot when prompted)
wsl --install

# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# During setup, keep "Use WSL 2 instead of Hyper-V" checked

# Verify (PowerShell)
docker --version
docker compose version
docker run hello-world
```

**GPU (NVIDIA only)**: with recent NVIDIA drivers + WSL 2, GPU access works
out of the box inside WSL containers. Verify:

```powershell
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If this fails, update your NVIDIA driver and confirm WSL 2 is the active
backend (`docker info | Select-String WSL`).

### 1.4 Shell syntax used in this tutorial

Commands in this tutorial use **bash** syntax (Linux/macOS, and Windows inside
WSL). On Windows PowerShell/Command Prompt, adapt paths and variables:

| bash / zsh | PowerShell | Notes |
|---|---|---|
| `$(pwd)` | `${PWD}` (PowerShell) or `%cd%` (cmd) | volume mount paths |
| `path/to/dir` | `path\to\dir` | separators |
| `export VAR=value` | `$env:VAR = "value"` | environment variables |
| `VAR=value cmd` | `$env:VAR = "value"; cmd` | per-command env vars |

Quick check:

```bash
docker --version
docker compose version
nvidia-smi   # Linux/Windows with NVIDIA GPU only; not available on macOS
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

Windows (PowerShell):

```powershell
pip install datasets
python single_roi_model\download_dataset.py
python single_roi_model\download_dataset.py D:\data\preprocessed
```

Option B — plain Python (same on all OSes):

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

macOS (Apple Silicon) — force the amd64 platform, see section 1.2:

```bash
docker build --platform linux/amd64 -t single-roi-rppg .
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

Windows (PowerShell):

```powershell
docker run --rm --entrypoint python single-roi-rppg /app/smoke_test.py
```
(The command is identical — only volume mounts differ between shells.)

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

Windows (PowerShell):

```powershell
docker run --rm --gpus all `
    -v ${PWD}\..\data\preprocessed:/data/preprocessed:ro `
    -v ${PWD}\models:/app/models `
    single-roi-rppg train
```

macOS / any CPU-only machine — omit `--gpus all`:

```bash
docker run --rm \
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

Windows (PowerShell) — put your videos under `data\videos` and adapt:

```powershell
docker run --rm --gpus all `
    -v ${PWD}\..\data\videos:/videos:ro `
    -v ${PWD}\models:/app/models `
    single-roi-rppg infer `
    --video /videos/my_video.avi `
    --model /app/models/single_roi_forehead_PRODUCTION.pth `
    --max-frames 450
```

This runs MediaPipe face landmarking per frame, crops the forehead ROI,
predicts with the 160-frame overlapping sliding window, and shows the
reconstructed waveform plus the FFT heart-rate estimate. (Inside a
headless container the matplotlib figure is not displayed; run this step
on a machine with a display, or adapt the plot to save a PNG.)

### 3.6 Run without Docker (local Python)

Linux / macOS:

```bash
cd single_roi_model
pip install -r requirements.txt
export RPPG_DATA_DIR=/path/to/preprocessed
python smoke_test.py
python single_roi_forehead_physnet.py train
python single_roi_forehead_physnet.py infer --video /path/to/video.avi \
    --model models/single_roi_forehead_PRODUCTION.pth
```

Windows (PowerShell):

```powershell
cd single_roi_model
pip install -r requirements.txt
$env:RPPG_DATA_DIR = "C:\path\to\preprocessed"
python smoke_test.py
python single_roi_forehead_physnet.py train
python single_roi_forehead_physnet.py infer --video C:\path\to\video.avi `
    --model models\single_roi_forehead_PRODUCTION.pth
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

macOS (Apple Silicon) — force the amd64 platform, see section 1.2:

```bash
docker build --platform linux/amd64 -t clinical-vitals-app .
docker compose up
```

Windows — run the same commands inside **PowerShell** or the **WSL terminal**;
if Docker Desktop uses the WSL 2 backend, `docker compose up` works unchanged.

The entrypoint prints a pre-flight report: CUDA availability, presence of both
checkpoints, and the MediaPipe `face_landmarker.task` model (auto-downloaded if
missing — it is not baked into the image at build time). Then the app starts.

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
sd = torch.load('/app/sliding_window_model/models/spatiotemporal_physnet_best_shuffle.pth',
                map_location='cpu', weights_only=False)
print('Stage 1 keys:', len(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd))
EOF
```

Windows (PowerShell) — heredocs are a bash feature; save the snippet to a file
first and pipe it:

```powershell
@'
import torch
sd = torch.load('/app/sliding_window_model/models/spatiotemporal_physnet_best_shuffle.pth',
                map_location='cpu', weights_only=False)
print('Stage 1 keys:', len(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd))
'@ | docker compose exec -T clinical-vitals-app python -
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

### Cross-platform issues

- **`/app/face_landmarker.task: Is a directory` (container exits with code 1)** —
  a stray `face_landmarker.task` *directory* (created by older compose files that
  mounted the then-nonexistent file) got copied into the image by `COPY . /app`.
  Delete it from your host checkout (`rm -rf sliding_window_model/face_landmarker.task`),
  pull the latest branch, and rebuild with `--no-cache`. The entrypoint now also
  removes such a stray directory before downloading, and `.dockerignore` keeps
  host artifacts out of the build context.
- **`python: can't open file '/app/sliding_window_model/gradio_live_clinical_diagnostic.py'`** —
  an older `entrypoint.sh` pointed at a path that does not exist in the image
  (the app is copied to `/app/gradio_live_clinical_diagnostic.py`). Pull the
  latest branch and rebuild; the entrypoint now uses the correct path.
- **`Stage 1 model NOT found at /app/sliding_window_model/models/spatiotemporal_physnet_best_shuffle.pth`** —
  that checkpoint was only in the repo-root `models/` folder. It is now also
  committed under `sliding_window_model/models/` so the compose mount provides
  it. `git pull` and re-run `docker compose up` (the mount is `:ro`, so no
  rebuild is needed). Note: `spatiotemporal_physnet_best_shuffle_150frames.pth`
  in the same folder is a *different* architecture (32-channel first conv) and
  is NOT a drop-in replacement for the Stage 1 checkpoint.
- **Build hangs at a geographic-area / timezone prompt (tzdata)** — older
  versions of the Dockerfiles let `apt-get` ask interactively for a region and
  city. Both now set `DEBIAN_FRONTEND=noninteractive` and `TZ=Etc/UTC`, so
  builds run unattended. Pull the latest branch and rebuild.
- **10-ROI build takes very long / re-downloads ~2 GB** — an earlier Dockerfile
  created a virtual environment and reinstalled the entire PyTorch stack on top
  of the base image. It now installs directly into the base environment, which
  already ships torch 2.0.1, and the build is much smaller and faster.
- **`OSError: libEGL.so.1 / libGLESv2.so.2: cannot open shared object file`** —
  MediaPipe's native library needs the OpenGL/EGL runtime libraries, which
  were missing from the images. Both Dockerfiles now install `libegl1` and
  `libgles2`; pull the latest branch and rebuild with `--no-cache`:

  ```bash
  docker build --no-cache --platform linux/amd64 -t single-roi-rppg .
  ```

  Note: this error means the MediaPipe library could not load at all — it is
  not about the `face_landmarker.task` model file, which downloads
  automatically on first use into `RPPG_MODEL_DIR` (default `/app/models`,
  i.e. your host `single_roi_model/models/` when mounted).
- **`RuntimeError: Numpy is not available` / `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`** —
  the `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` base image ships
  PyTorch built against NumPy 1.x, but an unpinned `numpy` install pulls in
  NumPy 2.x and breaks every `.cpu().numpy()` call. Both requirements files pin
  `numpy>=1.24.0,<2.0.0`, so **rebuild without cache** after pulling:

  ```bash
  docker build --no-cache --platform linux/amd64 -t single-roi-rppg .
  docker run --rm --entrypoint python single-roi-rppg /app/smoke_test.py
  # -> All 5 checks passed
  ```
  was not downloaded or is not mounted. Redo section 2.2 and check the
  `docker run -v` / compose volume paths.
- **`expected 5D input (got 3D input)`** — you are running an old version of
  `single_roi_forehead_physnet.py` with the `BatchNorm3d` bug; pull the latest
  `main`.
- **Port 7860 already in use** — change the host mapping in
  `sliding_window_model/docker-compose.yml` (e.g. `"7861:7860"`) and reload
  with `docker compose up -d --force-recreate`.
- **`face_landmarker.task` download failed at container start** — download it
  manually from the URL printed by `entrypoint.sh` and mount it:
  `-v $(pwd)/face_landmarker.task:/app/face_landmarker.task:ro`
  (PowerShell: `-v ${PWD}\face_landmarker.task:/app/face_landmarker.task:ro`).
- **Gradio figures don't render inside the single-ROI container** — the image
  sets `MPLBACKEND=Agg` for headless operation; the inference plot is meant
  for interactive use (section 3.6).
- **Permission denied writing checkpoints** — make sure the `./models` host
  directory is writable by the container user when mounting volumes.

### Linux

- **`permission denied while trying to connect to the Docker daemon socket`** —
  your user is not in the `docker` group; run `sudo usermod -aG docker $USER`
  and log out/in, or prefix commands with `sudo`.
- **`could not select devices driver ... capabilities: [[gpu]]`** — the NVIDIA
  Container Toolkit is not installed; see section 1.1.

### macOS

- **`Error response from daemon: could not select device driver "nvidia" with
  capabilities: [[gpu]]`** on `docker compose up` — macOS has no GPU passthrough,
  and older versions of `sliding_window_model/docker-compose.yml` hard-required
  an NVIDIA device. The reservation is now commented out; the app runs on CPU
  automatically. Pull the latest branch and re-run `docker compose up`
  (no rebuild needed — only the compose file changed).
- **`WARNING: The requested image's platform (linux/amd64) does not match the
  detected host platform (linux/arm64/v8)`** (Apple Silicon) — expected and
  harmless: the image runs under Rosetta 2 emulation. No action needed;
  optionally silence it with `docker run --platform linux/amd64 ...`.
- **`no matching manifest for linux/arm64/v8`** (Apple Silicon) — build/run
  with `--platform linux/amd64`; Rosetta 2 must be enabled in
  Docker Desktop → Settings → General.
- **The container is very slow** — x86-64 emulation on Apple Silicon is slow.
  Inference and smoke tests are fine; for real training use a Linux machine
  with an NVIDIA GPU or run locally with a native arm64 PyTorch build
  (section 3.6, `pip install torch` gives you a native wheel).
- **`docker: command not found` after installing Docker Desktop** — start it
  once via `open -a Docker` (the CLI socket only exists while the app runs),
  or use the `colima` route from section 1.2.
- **`nvidia-smi: command not found`** — expected; macOS has no CUDA. The
  containers run on CPU.

### Windows

- **`docker` cannot connect / daemon not running** — start Docker Desktop and
  wait for the whale icon; check the WSL 2 backend with `wsl --status`.
- **`wsl --install` requires a reboot** — reboot, reopen PowerShell as
  Administrator, and re-run `wsl --install` if needed.
- **`Error response from daemon: could not select devices driver ... [[gpu]]`** —
  update the NVIDIA driver (the Windows driver includes the WSL component),
  then verify with
  `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`.
- **Volume mount path errors (`${PWD}` empty in cmd)** — use PowerShell (not
  cmd) or use absolute Windows paths like
  `-v C:\data\preprocessed:/data/preprocessed:ro`; alternatively run all
  commands inside the WSL terminal, where the bash syntax works unchanged.
- **Line-ending issues with `entrypoint.sh` (`/bin/bash^M: bad interpreter`)** —
  if you edited the script on Windows, disable CRLF conversion:
  `git config --global core.autocrlf false` and re-checkout the repository.
- **`CUDA is not available, using CPU`** on a machine with an NVIDIA GPU —
  GPU passthrough requires WSL 2 (not Hyper-V) and a recent driver; check
  `docker info` shows the WSL backend.
