# Single ROI Model

This folder contains the **single region (single ROI) rPPG model** — a variant of the
Simple MCD-RPPG pipeline that estimates the blood volume pulse (BVP) signal and heart
rate from a single region of interest, instead of the multi-ROI setup used in
`sliding_window_model/` and `deeper_model/`.

## Contents

- `models/` — trained model weights (see `models/README.md`)
- Training and inference code for the single-ROI model

## Related folders

- `../sliding_window_model/` — sliding window, multi-ROI spatiotemporal PhysNet
- `../deeper_model/` — deeper multi-ROI spatiotemporal PhysNet
