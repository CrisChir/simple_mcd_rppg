# Simple MCD-RPPG

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

A simplified implementation for remote photoplethysmography (rPPG) using multi-channel detection (MCD) with a custom data preprocessing pipeline.

## Overview

This repository provides a simplified approach to **remote photoplethysmography (rPPG)** using **multi-channel detection (MCD)** techniques. The implementation features a custom data preprocessing flow, model architecture, and processing pipeline inspired by existing rPPG research.

### Key Features

- Custom data preprocessing pipeline for rPPG signals
- Multi-channel detection approach
- Simplified model architecture
- Preprocessed dataset support (Hugging Face Datasets format)

### Repository Layout

- `single_roi_model/` — single ROI (forehead) PhysNet3D training and inference (`single_roi_forehead_physnet.py`, Dockerfile, smoke tests)
- `sliding_window_model/` — 10-ROI spatiotemporal PhysNet + Gradio live clinical diagnostic web app (Docker deployment)
- `deeper_model/` — deeper 10-ROI spatiotemporal PhysNet variants
- `EDA&Preprocessing/` — dataset EDA and preprocessing notebooks
- `DOCKER_TUTORIAL.md` — tutorial for running both model variants in Docker containers, including dataset download and testing instructions

## Dataset

This work uses a **derivative dataset** based on:

**Source Dataset**: [wengziheng/mcd_rppg](https://huggingface.co/datasets/wengziheng/mcd_rppg)
Original dataset is unprocessed

The dataset used here contains preprocessed video data with synchronized physiological signals for remote heart rate estimation.

### Dataset Structure (Hugging Face Format)

```python
from datasets import load_dataset

# Load the preprocessed dataset
dataset = load_dataset("CrisChir/simple_mcd_rppg")

# Dataset splits
# dataset['train']      # Training split
# dataset['validation'] # Validation split  
# dataset['test']       # Test split

# Each example contains:
# - 'video': Video frames or face crops
# - 'bvp': Blood volume pulse signal
# - 'hr': Heart rate (ground truth)
# - 'metadata': Additional information (fps, duration, etc.)
```

### Dataset Card Boilerplate

```markdown
Original dataset 
---
language:
- en
license:
- cc-by-nc-nd-4.0
tags:
- rppg
- remote-photoplethysmography
- multi-channel-detection
- heart-rate-estimation
- computer-vision
- physiological-signals
datasets:
- wengziheng/mcd_rppg
---

# Dataset Card for Simple MCD-RPPG

## Dataset Description

- **Repository**: [CrisChir/simple_mcd_rppg](https://github.com/CrisChir/simple_mcd_rppg)
- **Source**: Derivative of [wengziheng/mcd_rppg](https://huggingface.co/datasets/wengziheng/mcd_rppg)
- **License** of original dataset: CC BY 4.0

### Dataset Summary

A preprocessed dataset for remote photoplethysmography (rPPG) research using multi-channel detection techniques. This dataset contains video frames with synchronized blood volume pulse (BVP) signals and heart rate annotations.

### Supported Tasks

- Heart rate estimation
- Remote photoplethysmography
- Vital sign monitoring
- Computer vision for healthcare

### Languages

- Python
- PyTorch/TensorFlow (for model training)

## Dataset Structure

### Data Instances

```python
{
    'video': [frame1, frame2, ..., frameN],      # List of video frames (numpy arrays)
    'bvp': [bvp_value1, bvp_value2, ..., bvp_valueN],  # Blood volume pulse signal
    'hr': heart_rate_value,                      # Ground truth heart rate (BPM)
    'fps': frames_per_second,                    # Video frame rate
    'duration': video_duration_seconds,          # Video duration
    'subject_id': 'subject_identifier',          # Subject identifier
    'environment': 'environment_type'           # Recording environment
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `video` | List[np.ndarray] | Video frames (H x W x C) |
| `bvp` | List[float] | Blood volume pulse signal |
| `hr` | float | Heart rate in beats per minute (BPM) |
| `fps` | float | Frames per second |
| `duration` | float | Duration in seconds |
| `subject_id` | str | Unique subject identifier |
| `environment` | str | Recording environment conditions |

### Data Splits

| Split | Description | Size |
|-------|-------------|------|
| Train | Training data | TBD |
| Validation | Validation data | TBD |
| Test | Test data | TBD |

## Dataset Creation

### Curation Rationale

This dataset was created to provide a simplified, preprocessed version of the MCD-RPPG dataset with a custom preprocessing pipeline that:
- Standardizes video frame extraction
- Synchronizes physiological signals
- Applies quality filtering
- Normalizes data for consistent training

### Source Data

**Primary Source**: [wengziheng/mcd_rppg](https://huggingface.co/datasets/wengziheng/mcd_rppg)

The original dataset contains raw video recordings with physiological signals. Our preprocessing pipeline applies the following transformations:

1. **Face Detection & Cropping**: Extract and align face regions
2. **Signal Synchronization**: Align video frames with BVP signals
3. **Quality Filtering**: Remove low-quality or corrupted samples
4. **Normalization**: Standardize frame sizes and signal ranges
5. **Segmentation**: Split long videos into manageable segments

### Preprocessing Code

```python
from datasets import Dataset, DatasetDict
import numpy as np
import cv2

def preprocess_video(video_path, bvp_path, fps=30, segment_length=10):
    """
    Preprocess video and BVP signal into standardized format.
    
    Args:
        video_path: Path to video file
        bvp_path: Path to BVP signal file
        fps: Target frames per second
        segment_length: Length of each segment in seconds
    
    Returns:
        List of preprocessed segments
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Load BVP signal
    bvp_signal = np.loadtxt(bvp_path)
    
    # Synchronize and segment
    segments = []
    for i in range(0, len(frames), fps * segment_length):
        segment_frames = frames[i:i + fps * segment_length]
        segment_bvp = bvp_signal[i:i + fps * segment_length]
        
        if len(segment_frames) == fps * segment_length:
            segments.append({
                'video': segment_frames,
                'bvp': segment_bvp.tolist(),
                'fps': fps,
                'duration': segment_length
            })
    
    return segments

# Example usage
train_segments = preprocess_video('train_video.mp4', 'train_bvp.txt')
test_segments = preprocess_video('test_video.mp4', 'test_bvp.txt')

# Create Hugging Face dataset
dataset = DatasetDict({
    'train': Dataset.from_list(train_segments),
    'test': Dataset.from_list(test_segments)
})

# Save to disk
dataset.save_to_disk('simple_mcd_rppg')
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load from local path
dataset = load_dataset('path/to/simple_mcd_rppg')

# Or from Hugging Face Hub (when published)
# dataset = load_dataset('CrisChir/simple_mcd_rppg')

# Access data
train_data = dataset['train']
print(f"Training samples: {len(train_data)}")
print(f"Features: {train_data.features}")
```

### Example Training Script

```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load dataset
dataset = load_dataset('CrisChir/simple_mcd_rppg')

# Preprocess function
def preprocess(example):
    # Convert video to tensor
    video = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 
                         for frame in example['video']])
    
    # Convert BVP to tensor
    bvp = torch.tensor(example['bvp'], dtype=torch.float32)
    
    return {'video': video, 'bvp': bvp, 'hr': example['hr']}

# Apply preprocessing
dataset = dataset.map(preprocess, batched=False)

# Create data loaders
train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
val_loader = DataLoader(dataset['validation'], batch_size=8)

# Training loop
for batch in train_loader:
    videos = batch['video']  # (B, T, C, H, W)
    bvp_signals = batch['bvp']  # (B, T)
    heart_rates = batch['hr']  # (B,)
    
    # Your model training code here
    # predictions = model(videos)
    # loss = criterion(predictions, heart_rates)
    # loss.backward()
    # optimizer.step()
```

## Citation

### BibTeX

```bibtex
@misc{simple_mcd_rppg,
  author = {CrisChir},
  title = {Simple MCD-RPPG: A Simplified Multi-Channel Remote Photoplethysmography Dataset},
  year = {2024},
  howpublished = {\url{https://github.com/CrisChir/simple_mcd_rppg}},
  note = {Accessed: 2024-07-23}
}

@misc{wengziheng_mcd_rppg,
  author = {Weng, Ziheng},
  title = {MCD-RPPG Dataset},
  year = {2023},
  howpublished = {\url{https://huggingface.co/datasets/wengziheng/mcd_rppg}},
  note = {Source dataset for Simple MCD-RPPG}
}
```

## License

This dataset is licensed under **CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material

The full license text is available in the [LICENSE](LICENSE) file.

## Acknowledgments

This work was inspired by:
- [MCD-RPPG GitHub Repository](https://github.com/wengziheng/mcd_rppg) - Original implementation
- [wengziheng/mcd_rppg Hugging Face Dataset](https://huggingface.co/datasets/wengziheng/mcd_rppg) - Source dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/CrisChir/simple_mcd_rppg).
