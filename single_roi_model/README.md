# Single ROI Model

This folder contains the **single region (single ROI) rPPG model** — a variant of the
Simple MCD-RPPG pipeline that estimates the blood volume pulse (BVP) signal and heart
rate from a single region of interest, instead of the multi-ROI setup used in
`sliding_window_model/` and `deeper_model/`.

## Contents

- `single_roi_forehead_physnet.py` — training and inference code for the single-ROI (forehead) PhysNet3D model
- `models/` — trained model weights (see `models/README.md`)

## Overview

The model extracts a 24x24 forehead ROI (via MediaPipe face landmarks for raw video,
or precomputed `roi_forehead` crops from `.npz` files) and reconstructs the PPG
waveform with an edge-safe spatiotemporal network. Two key fixes over a naive
implementation:

1. **Replicate padding** in the temporal encoder, preventing edge amplitude collapse.
2. **Overlapping sliding-window inference** (160-frame chunks, 50% overlap, Hann
   blending), guaranteeing consistent amplitude for videos of any length.

## Usage

Training (expects preprocessed `.npz` files in `DATA_DIR`, configurable in `Config`):

```bash
python single_roi_forehead_physnet.py
```

Standalone video inference after training:

```python
run_video_inference_demo("path/to/video.avi", max_frames=450)
```

## Related folders

- `../sliding_window_model/` — sliding window, multi-ROI spatiotemporal PhysNet
- `../deeper_model/` — deeper multi-ROI spatiotemporal PhysNet
