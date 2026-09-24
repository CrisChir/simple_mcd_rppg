#!/usr/bin/env python3
"""Evaluate all checkpoints from the HuggingFace repo Bgeorge/Single_ROI_RPPG.

Downloads every `single_roi_forehead_*.pth` from
https://huggingface.co/Bgeorge/Single_ROI_RPPG into `models_hf_test/`
(the default model in `models/` is NOT touched), extracts the forehead ROI
from a test video ONCE, and runs every checkpoint through the same
sliding-window + bandpass + FFT pipeline used by `single_roi_forehead_physnet.py`.
Then prints a comparison table and writes `hf_model_comparison.json`
and per-model waveform PNGs.

The HF repo contains four architecture families. 13 checkpoints match the
current `SingleROIPhysNet`; 5 are legacy variants whose architectures are
reconstructed below from the checkpoint tensor shapes (weights are the
source of truth):

  - `features.*` + `temporal.*` (STRICT_BANDPASS)
  - `features.*` + `final_conv` (TRUE_3D)
  - `spatial_conv`(3x3D) + `temporal_encoder`(3 blocks) + `output_conv` (TCN_VELOCITY, fully_optimized)
  - `spatial_conv`(2x3D) + `temporal_encoder`(2 blocks) + `output_conv` (physnet_best)

Usage:
    python eval_hf_models.py --video /path/to/video.avi [--max-frames 450]
    python eval_hf_models.py --list            (just download + load check)
    python eval_hf_models.py --skip-download   (use already-downloaded files)

Environment: same as the main script (RPPG_MODEL_DIR etc. are irrelevant
here; only the video path and CUDA_DEVICE matter).
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_roi_forehead_physnet as m


HF_REPO = "https://huggingface.co/Bgeorge/Single_ROI_RPPG/resolve/main"
CHECKPOINTS = [
    "single_roi_forehead_BANDPASS.pth",
    "single_roi_forehead_BULLETPROOF.pth",
    "single_roi_forehead_CHROMINANCE.pth",
    "single_roi_forehead_ENHANCED.pth",
    "single_roi_forehead_FINAL.pth",
    "single_roi_forehead_FIXED.pth",
    "single_roi_forehead_FREQ_GUIDED.pth",
    "single_roi_forehead_PRODUCTION.pth",
    "single_roi_forehead_RESIDUAL.pth",
    "single_roi_forehead_RHYTHM_TRACKER.pth",
    "single_roi_forehead_SMOOTH.pth",
    "single_roi_forehead_STABLE.pth",
    "single_roi_forehead_STRICT_BANDPASS.pth",
    "single_roi_forehead_TCN_VELOCITY.pth",
    "single_roi_forehead_TRUE_3D.pth",
    "single_roi_forehead_ULTIMATE.pth",
    "single_roi_forehead_fully_optimized.pth",
    "single_roi_forehead_physnet_best.pth",
]
DOWNLOAD_DIR = Path(os.environ.get("HF_TEST_DIR", "models_hf_test"))


# ---------------------------------------------------------------------------
# Legacy architectures, reconstructed from checkpoint tensor shapes
# (weights are the source of truth; families found in the HF repo)
# ---------------------------------------------------------------------------


class LegacyFeaturesTemporal(nn.Module):
    """STABLE, STRICT_BANDPASS: features.* (3 conv3d blocks, kernel (3,3,3) at
    depth) + temporal.* (Conv1d k7 + Conv1d k1)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        return torch.squeeze(self.temporal(x), dim=1)


class LegacyFeaturesFinalConv(nn.Module):
    """TRUE_3D: features.* (4 conv3d blocks) + final_conv 1x1x1x1."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.final_conv = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        return torch.squeeze(x, dim=-1).squeeze(-1).squeeze(1)


class LegacySpatialTemporalOutput(nn.Module):
    """Family: spatial_conv.* (2 blocks: 1x5x5, 1x3x3) + temporal_encoder
    (parameterized conv1d blocks) + optional output_conv.

    The HF checkpoints differ in the number of parameterless layers between
    blocks (`gap`): ELU+Dropout (gap=2) or ELU only (gap=1), and in whether
    the final 1-channel conv lives inside temporal_encoder or in a separate
    output_conv module. All variants are covered by parameters:

      BANDPASS:      [(64,k7),(32,k5)] gap2, final conv k7 in encoder
      BULLETPROOF:   [(64,k9),(32,k7),(16,k5)] gap2 + output_conv
      ENHANCED/FIXED/SMOOTH: [(64,k15),(32,k15),(16,k15)] gap2 + output_conv
      TCN_VELOCITY:  [(64,k15,d1),(64,k15,d2),(32,k15,d4)] gap2 + output_conv
      fully_optimized: [(64,k15),(32,k15),(16,k15)] gap1, final conv in encoder
      physnet_best:  [(32,k15,d1),(16,k15,d2)] gap1, final conv in encoder
    """

    def __init__(self, in_channels: int = 3, blocks=None, gap: int = 2,
                 final_in_encoder: bool = False, final_kernel: int = 1,
                 final_dropout: bool = False, use_output_conv: bool = False):
        super().__init__()
        if not blocks:
            raise ValueError("blocks spec required")
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        layers = []
        last_c = 64
        n = len(blocks)
        for i, (c_out, k, d) in enumerate(blocks):
            layers.append(nn.Conv1d(last_c, c_out, kernel_size=k,
                                    padding=(k // 2) * d, dilation=d))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ELU(inplace=True))
            if gap == 2 and (i < n - 1 or final_dropout):
                layers.append(nn.Dropout(0.3))
            last_c = c_out
        if final_in_encoder:
            layers.append(nn.Conv1d(last_c, 1, kernel_size=final_kernel,
                                    padding=final_kernel // 2))
        self.temporal_encoder = nn.Sequential(*layers)
        self.use_output_conv = use_output_conv
        if use_output_conv:
            self.output_conv = nn.Conv1d(last_c, 1, kernel_size=1)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.temporal_encoder(x)
        if self.use_output_conv:
            x = self.output_conv(x)
        return torch.squeeze(x, dim=1)


class LegacyResidual(nn.Module):
    """FREQ_GUIDED, RESIDUAL: spatial_conv.* + temporal_in + 2 residual
    blocks + temporal_out + output_conv."""

    class _ResBlock(nn.Module):
        def __init__(self, channels: int, kernel: int = 5):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel,
                                   padding=kernel // 2)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel,
                                   padding=kernel // 2)
            self.bn2 = nn.BatchNorm1d(channels)

        def forward(self, x):
            y = torch.nn.functional.elu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return torch.nn.functional.elu(x + y)

    def __init__(self, in_channels: int = 3, output_kernel: int = 1):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.temporal_in = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
        )
        self.res_block1 = self._ResBlock(64, 5)
        self.res_block2 = self._ResBlock(64, 5)
        self.temporal_out = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
        )
        self.output_conv = nn.Conv1d(32, 1, kernel_size=output_kernel,
                                     padding=output_kernel // 2)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.temporal_in(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.temporal_out(x)
        return torch.squeeze(self.output_conv(x), dim=1)


class LegacyChrominance(nn.Module):
    """CHROMINANCE: rgb_mixer (3->8 1x1x1x1, kept for weight compatibility;
    the trained forward runs spatial_conv on the RGB input) + spatial_conv
    (8->32->64 with a leading 3->8 1x1x1x1 conv) + temporal_encoder
    (64->64->64->32, gap2) + output_conv."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.rgb_mixer = nn.Conv3d(in_channels, 8, kernel_size=(1, 1, 1))
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ELU(inplace=True),
            nn.Conv3d(8, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
        )
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.temporal_encoder(x)
        return torch.squeeze(self.output_conv(x), dim=1)


def _st(blocks, gap=2, final_in_encoder=False, final_kernel=1,
        final_dropout=False, use_output_conv=False):
    return lambda: LegacySpatialTemporalOutput(
        blocks=blocks, gap=gap, final_in_encoder=final_in_encoder,
        final_kernel=final_kernel, final_dropout=final_dropout,
        use_output_conv=use_output_conv)


LEGACY_BUILDERS = {
    "single_roi_forehead_BANDPASS.pth": _st([(64, 7, 1), (32, 5, 1)],
                                            final_in_encoder=True, final_kernel=7,
                                            final_dropout=True),
    "single_roi_forehead_BULLETPROOF.pth": _st([(64, 9, 1), (32, 7, 1), (16, 5, 1)],
                                               use_output_conv=True),
    "single_roi_forehead_CHROMINANCE.pth": LegacyChrominance,
    "single_roi_forehead_ENHANCED.pth": _st([(64, 15, 1), (32, 15, 1), (16, 15, 1)],
                                           use_output_conv=True),
    "single_roi_forehead_FIXED.pth": _st([(64, 15, 1), (32, 15, 1), (16, 15, 1)],
                                         use_output_conv=True),
    "single_roi_forehead_FREQ_GUIDED.pth": LegacyResidual,
    "single_roi_forehead_RESIDUAL.pth": lambda: LegacyResidual(output_kernel=7),
    "single_roi_forehead_SMOOTH.pth": _st([(64, 15, 1), (32, 15, 1), (16, 15, 1)],
                                          use_output_conv=True),
    "single_roi_forehead_STABLE.pth": LegacyFeaturesTemporal,
    "single_roi_forehead_STRICT_BANDPASS.pth": LegacyFeaturesTemporal,
    "single_roi_forehead_TCN_VELOCITY.pth": _st([(64, 15, 1), (64, 15, 2), (32, 15, 4)],
                                                use_output_conv=True),
    "single_roi_forehead_TRUE_3D.pth": LegacyFeaturesFinalConv,
    "single_roi_forehead_fully_optimized.pth": _st([(64, 15, 1), (32, 15, 1), (16, 15, 1)],
                                                   gap=1, final_in_encoder=True),
    "single_roi_forehead_physnet_best.pth": _st([(32, 15, 1), (16, 15, 2)],
                                               gap=1, final_in_encoder=True),
}


def download_checkpoints() -> int:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name in CHECKPOINTS:
        dest = DOWNLOAD_DIR / name
        if dest.exists():
            print(f"  [cached] {name}")
            ok += 1
            continue
        url = f"{HF_REPO}/{urllib.request.quote(name)}"
        try:
            print(f"  [download] {name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print("ok")
            ok += 1
        except Exception as e:
            print(f"FAILED ({e})")
    return ok


def load_checkpoint_model(path: Path):
    """Return (model, architecture_name) or raise."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    name = path.name
    if name in LEGACY_BUILDERS:
        model = LEGACY_BUILDERS[name]()
        arch = type(model).__name__
    else:
        model = m.SingleROIPhysNet()
        arch = "SingleROIPhysNet (current)"

    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        raise RuntimeError(f"state_dict mismatch: {e}")
    model.eval()
    return model, arch


def extract_roi_tensor(video_path: str, max_frames: int):
    """Run the main script's video pipeline up to the model input tensor."""
    frames = m.load_video_frames_ffmpeg(video_path, target_frames=max_frames)
    if len(frames) == 0:
        raise RuntimeError("no frames decoded from video")
    landmarker = m.init_mediapipe_landmarker()
    roi_patches = []
    detected = 0
    for frame_rgb in frames:
        mp_image = m.mp.Image(image_format=m.mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image)
        roi_24x24 = np.zeros((m.config.ROI_SIZE[1], m.config.ROI_SIZE[0], 3), dtype=np.uint8)
        if hasattr(detection_result, "face_landmarks") and len(detection_result.face_landmarks) > 0:
            roi_24x24 = m.crop_roi_from_landmarks(frame_rgb, detection_result, "forehead", m.config.ROI_SIZE)
            detected += 1
        roi_patches.append(roi_24x24.astype(np.float32) / 255.0)
    tensor = torch.from_numpy(np.array(roi_patches, dtype=np.float32)).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor, len(frames), detected


def main():
    parser = argparse.ArgumentParser(description="Evaluate all HuggingFace Single_ROI_RPPG checkpoints")
    parser.add_argument("--video", type=str, default=None, help="Path to a face video for inference")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--list", action="store_true", help="Only download and verify checkpoints load")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading; use cached files")
    args = parser.parse_args()

    device = m.config.DEVICE
    print("=" * 60)
    print("HF checkpoint evaluation - Bgeorge/Single_ROI_RPPG")
    print("=" * 60)
    print(f"Device: {device} | Download dir: {DOWNLOAD_DIR.resolve()}")

    if not args.skip_download:
        n = download_checkpoints()
        if n != len(CHECKPOINTS):
            print(f"WARNING: only {n}/{len(CHECKPOINTS)} checkpoints available")

    if args.list or args.video is None:
        results = []
        for name in CHECKPOINTS:
            path = DOWNLOAD_DIR / name
            if not path.exists():
                results.append({"model": name, "status": "download failed"})
                continue
            try:
                _, arch = load_checkpoint_model(path)
                results.append({"model": name, "status": "loads OK", "architecture": arch})
            except Exception as e:
                results.append({"model": name, "status": f"load failed: {e}"})
        print()
        for r in results:
            print(f"  {r['model']:<45} {r['status']:<30} {r.get('architecture', '')}")
        return

    # Full evaluation with video
    max_frames = args.max_frames if args.max_frames is not None else m.config.TARGET_FRAMES
    print(f"\nExtracting forehead ROI from: {args.video} (max {max_frames} frames)")
    tensor_input, total_frames, detected = extract_roi_tensor(args.video, max_frames)
    print(f"  {total_frames} frames decoded, face detected in {detected}")
    if detected == 0:
        print("  ERROR: no face detected - cannot evaluate")
        return 1

    tensor_input = tensor_input.to(device)
    results = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in CHECKPOINTS:
        path = DOWNLOAD_DIR / name
        entry = {"model": name}
        if not path.exists():
            entry["status"] = "download failed"
            results.append(entry)
            continue
        try:
            model, arch = load_checkpoint_model(path)
            model.to(device)
            pred = m.predict_with_sliding_window(model, tensor_input, seq_len=m.config.SEQUENCE_LENGTH)
            filtered = m.butter_bandpass_filter(pred, fs=m.config.FS)
            std = np.std(filtered)
            if std < 1e-5:
                std = 1.0
            filtered = (filtered - np.mean(filtered)) / std
            bpm = m.calculate_bpm_from_fft(filtered, fs=m.config.FS)
            entry.update({
                "status": "ok",
                "architecture": arch,
                "bpm_fft": round(float(bpm), 2),
                "signal_std": round(float(std), 6),
            })
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(filtered, color="crimson", linewidth=1.0)
            ax.set_title(f"{name} - FFT HR: {bpm:.1f} BPM")
            ax.set_xlabel("Frame Index")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            png_path = DOWNLOAD_DIR / f"eval_{Path(name).stem}.png"
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            entry["waveform_png"] = str(png_path)
        except Exception as e:
            entry["status"] = f"inference failed: {e}"
        results.append(entry)
        print(f"  {name:<45} {entry.get('bpm_fft', '-'):>8}  {entry['status']}")

    report = {
        "video": os.path.abspath(args.video),
        "frames_processed": total_frames,
        "frames_with_face": detected,
        "sampling_rate_fps": m.config.FS,
        "results": results,
    }
    out = DOWNLOAD_DIR / "hf_model_comparison.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Report written to {out}")
    print(f"{'=' * 60}\n")
    print(f"{'Model':<45} {'BPM (FFT)':>10}")
    print("-" * 57)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['model']:<45} {r['bpm_fft']:>10.1f}")
        else:
            print(f"{r['model']:<45} {'FAIL':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
