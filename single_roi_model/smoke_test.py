#!/usr/bin/env python3
"""Smoke tests for the single-ROI (forehead) PhysNet3D model.

Runs without any dataset or GPU. Verifies that:
  1. The model forward pass produces the expected output shape.
  2. The committed PRODUCTION checkpoint loads with strict=True.
  3. Sliding-window inference works for videos longer than the training
     window and produces a finite signal.
  4. The short-signal guard in the bandpass filter works.
  5. Training raises a clear error when the dataset directory is empty
     (the dataset must be downloaded separately).

Usage:
    python smoke_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_roi_forehead_physnet as m


def main() -> int:
    checks = []

    model = m.SingleROIPhysNet()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 160, 24, 24))
    checks.append(("forward pass output shape == (1, 160)", tuple(out.shape) == (1, 160)))

    ckpt_path = Path(__file__).parent / 'models' / 'single_roi_forehead_PRODUCTION.pth'
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict']
        try:
            model.load_state_dict(sd, strict=True)
            checks.append(("PRODUCTION checkpoint loads (strict)", True))
        except RuntimeError as e:
            checks.append((f"PRODUCTION checkpoint loads (strict)", False))
    else:
        checks.append(("PRODUCTION checkpoint present", False))

    with torch.no_grad():
        pred = m.predict_with_sliding_window(model, torch.randn(1, 3, 450, 24, 24), seq_len=160)
    checks.append(("sliding-window inference returns 450 finite samples",
                   pred.shape == (450,) and np.isfinite(pred).all()))

    short = m.butter_bandpass_filter(np.random.randn(20))
    checks.append(("short-signal filter guard returns 20 samples unfiltered", len(short) == 20))

    with tempfile.TemporaryDirectory() as td:
        m.config.DATA_DIR = Path(td)
        try:
            m.train_single_roi_pipeline()
            checks.append(("empty dataset raises FileNotFoundError", False))
        except FileNotFoundError:
            checks.append(("empty dataset raises FileNotFoundError", True))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\n{len(failed)} check(s) FAILED")
        return 1
    print(f"\nAll {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
