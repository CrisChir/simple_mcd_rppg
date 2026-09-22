# Single ROI (Forehead) PhysNet3D - PRODUCTION OVERLAP INFERENCE VERSION
#
# THE FIXES:
# 1. Replicate Padding: Network pads with edge frames instead of zeros, preventing edge collapse.
# 2. Sliding Window Inference: Predicts 160-frame chunks with 50% overlap and blends them.
#    This guarantees consistent amplitude across videos of ANY length.

import os
import sys
import glob
import logging
import math
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import scipy.signal
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


# ============================================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================================

ROI_LANDMARKS = {'forehead': [10, 67, 103, 109, 337, 338, 297, 151]}
FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def ensure_face_landmarker_model(model_path: str = None) -> str:
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, FACE_LANDMARKER_MODEL)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    return model_path


def init_mediapipe_landmarker(model_path: str = None) -> Any:
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe is not installed.")
    model_path = ensure_face_landmarker_model(model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_roi_from_landmarks(frame_rgb: np.ndarray, landmarks: Any, 
                             roi_name: str = 'forehead', 
                             target_size: Tuple[int, int] = None) -> np.ndarray:
    if target_size is None:
        target_size = config.ROI_SIZE
    if roi_name not in ROI_LANDMARKS:
        raise ValueError(f"Unknown ROI: {roi_name}")
    
    h, w, _ = frame_rgb.shape
    roi_pts = []
    for idx in ROI_LANDMARKS[roi_name]:
        if MEDIAPIPE_AVAILABLE and hasattr(landmarks, 'face_landmarks') and len(landmarks.face_landmarks) > 0:
            lm = landmarks.face_landmarks[0][idx]
            roi_pts.append([int(lm.x * w), int(lm.y * h)])
        else:
            raise ValueError("Invalid landmarks")
    
    if not roi_pts:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    roi_pts = np.array(roi_pts)
    x, y, w_box, h_box = cv2.boundingRect(roi_pts)
    padding = 5
    x, y = max(0, x - padding), max(0, y - padding)
    w_box, h_box = min(w - x, w_box + 2 * padding), min(h - y, h_box + 2 * padding)
    
    roi_crop = frame_rgb[y:y+h_box, x:x+w_box]
    if roi_crop.size == 0:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(roi_crop, target_size)


def load_video_frames_ffmpeg(video_path: str, target_frames: int = None) -> np.ndarray:
    if target_frames is None:
        target_frames = config.TARGET_FRAMES
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < target_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) > 0 and len(frames) < target_frames:
        frames.extend([frames[-1]] * (target_frames - len(frames)))
    return np.array(frames)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    DATA_DIR: Path = Path('/home/cristic/preprocessed_data/')
    MODEL_DIR: Path = Path('./models')
    MODEL_SAVE_PATH: Path = Path("single_roi_forehead_PRODUCTION.pth")
    
    ROI_NAME: str = 'roi_forehead'
    IN_CHANNELS: int = 3
    
    TARGET_FRAMES: int = 160
    ROI_SIZE: Tuple[int, int] = (24, 24)
    SEQUENCE_LENGTH: int = 160
    
    FS: float = 30.0
    LOWCUT: float = 0.75
    HIGHCUT: float = 2.5
    BUTTER_ORDER: int = 4
    
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100 
    LEARNING_RATE: float = 1e-4 
    NUM_WORKERS: int = 8
    GRAD_CLIP: float = 5.0
    
    DEVICE: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    PATIENCE: int = 20  
    LOG_BATCH_INTERVAL: int = 100

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def butter_bandpass_filter(data: np.ndarray, lowcut: float = None, highcut: float = None, 
                          fs: float = None, order: int = None) -> np.ndarray:
    if lowcut is None: lowcut = config.LOWCUT
    if highcut is None: highcut = config.HIGHCUT
    if fs is None: fs = config.FS
    if order is None: order = config.BUTTER_ORDER
    
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def calculate_bpm_from_fft(signal_1d: np.ndarray, fs: float = None, 
                         lowcut: float = None, highcut: float = None) -> float:
    if fs is None: fs = config.FS
    if lowcut is None: lowcut = config.LOWCUT
    if highcut is None: highcut = config.HIGHCUT
    
    signal_cleaned = signal_1d - np.mean(signal_1d)
    freqs, psd = welch(signal_cleaned, fs=fs, nperseg=len(signal_cleaned))
    
    valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    if len(valid_idx) == 0:
        return 75.0
    
    peak_freq = freqs[valid_idx[np.argmax(psd[valid_idx])]]
    return peak_freq * 60.0


# ============================================================================
# DATASET 
# ============================================================================

def load_npz_file(file_path: str) -> Optional[Dict]:
    try: return np.load(file_path, allow_pickle=True)
    except: return None

class SingleROIRPPGDataset(Dataset):
    def __init__(self, file_paths: List[str], sequence_length: int = None, augment: bool = False):
        self.file_paths = file_paths
        self.sequence_length = sequence_length or config.SEQUENCE_LENGTH
        self.augment = augment
        self.samples = []
        
        for path in file_paths:
            data = load_npz_file(path)
            if data is None: continue
            ppg_key = 'ppg_values' if 'ppg_values' in data else 'ppg'
            if ppg_key not in data: continue
            length = len(data[ppg_key])
            for w in range(length // self.sequence_length):
                self.samples.append((path, w * self.sequence_length, (w+1) * self.sequence_length))
    
    def __len__(self) -> int: return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, start, end = self.samples[idx]
        data = load_npz_file(file_path)
        if data is None: return self._get_dummy_sample()
        
        ppg_key = 'ppg_values' if 'ppg_values' in data else 'ppg'
        try: ppg_segment = data[ppg_key][start:end].astype(np.float32)
        except: return self._get_dummy_sample()
        
        ppg_segment = butter_bandpass_filter(ppg_segment, fs=config.FS)
        ppg_std = np.std(ppg_segment)
        if ppg_std < 1e-5: ppg_std = 1.0 
        ppg_normalized = (ppg_segment - np.mean(ppg_segment)) / ppg_std
        
        roi_key = config.ROI_NAME
        actual_key = next((k for k in data.files if roi_key in k), None)
        
        if actual_key is not None:
            roi_data = data[actual_key][start:end]
        else:
            for alt_key in ['roi_forehead', 'forehead', 'roi_head']:
                if alt_key in data:
                    roi_data = data[alt_key][start:end]
                    break
            else:
                roi_data = np.zeros((self.sequence_length, *config.ROI_SIZE, 3), dtype=np.uint8)
        
        if roi_data.shape != (self.sequence_length, *config.ROI_SIZE, 3):
            roi_data = np.zeros((self.sequence_length, *config.ROI_SIZE, 3), dtype=np.uint8)
        
        roi_data = roi_data.astype(np.float32) / 255.0
        
        if self.augment:
            if random.random() > 0.5: roi_data = np.flip(roi_data, axis=2)
            if random.random() > 0.5:
                roi_data = np.clip(roi_data * random.uniform(0.85, 1.15), 0.0, 1.0)
        
        roi_data = np.transpose(roi_data, (3, 0, 1, 2))
        return (torch.tensor(roi_data.copy(), dtype=torch.float32).contiguous(),
                torch.tensor(ppg_normalized.copy(), dtype=torch.float32).contiguous())
    
    def _get_dummy_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros((config.IN_CHANNELS, self.sequence_length, *config.ROI_SIZE), dtype=torch.float32).contiguous(),
                torch.zeros(self.sequence_length, dtype=torch.float32).contiguous())


# ============================================================================
# RHYTHM LOSS FUNCTION (Pearson + FFT Cross-Entropy)
# ============================================================================

class RhythmLoss(nn.Module):
    def __init__(self, fps=30.0, low=0.75, high=2.5, eps=1e-6):
        super().__init__()
        self.fps = fps
        self.low = low
        self.high = high
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        target_c = target - target.mean(dim=-1, keepdim=True)
        
        num = torch.sum(pred_c * target_c, dim=-1)
        den = torch.sqrt(torch.sum(pred_c**2, dim=-1) * torch.sum(target_c**2, dim=-1)) + self.eps
        pcc = num / den
        loss_pearson = 1.0 - torch.mean(pcc)
        
        pred_fft = torch.abs(torch.fft.rfft(pred_c, dim=-1)) ** 2
        target_fft = torch.abs(torch.fft.rfft(target_c, dim=-1)) ** 2
        
        freqs = torch.fft.rfftfreq(pred.size(-1), d=1.0/self.fps).to(pred.device)
        mask = (freqs >= self.low) & (freqs <= self.high)
        
        pred_psd = pred_fft[:, mask]
        target_psd = target_fft[:, mask]
        
        pred_prob = pred_psd / (torch.sum(pred_psd, dim=-1, keepdim=True) + self.eps)
        target_prob = target_psd / (torch.sum(target_psd, dim=-1, keepdim=True) + self.eps)
        
        loss_freq = -torch.mean(torch.sum(target_prob * torch.log(pred_prob + self.eps), dim=-1))
        
        total_loss = loss_pearson + loss_freq
        return total_loss, torch.mean(pcc).item()


# ============================================================================
# EDGE-SAFE TCN ARCHITECTURE
# ============================================================================

class SingleROIPhysNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # CRITICAL FIX 1: padding_mode='replicate'
        # Stops the edges of the video from collapsing to zero amplitude.
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, padding=7, dilation=1, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 64, kernel_size=15, padding=14, dilation=2, padding_mode='replicate'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 32, kernel_size=15, padding=28, dilation=4, padding_mode='replicate'),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Isolate the pulse locally per pixel
        x = x - x.mean(dim=2, keepdim=True)
        
        x = self.spatial_conv(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        
        ppg_pred = self.temporal_encoder(x)
        return torch.squeeze(ppg_pred, dim=1)


# ============================================================================
# SLIDING WINDOW INFERENCE UTILITY (CRITICAL FIX 2)
# ============================================================================

def predict_with_sliding_window(model: nn.Module, tensor_input: torch.Tensor, seq_len: int = 160) -> np.ndarray:
    """
    Feeds exactly 160 frames to the model at a time, with 50% overlap.
    Blends them together using a Hann window. This perfectly prevents amplitude
    bursts and guarantees stability regardless of video length.
    """
    B, C, T, H, W = tensor_input.shape
    stride = seq_len // 2
    
    if T <= seq_len:
        with torch.no_grad():
            return model(tensor_input).squeeze(0).cpu().numpy()
            
    ppg_pred = np.zeros(T)
    weight = np.zeros(T)
    hann = np.hanning(seq_len)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, T - seq_len + 1, stride):
            chunk = tensor_input[:, :, i:i+seq_len, :, :]
            pred_chunk = model(chunk).squeeze(0).cpu().numpy()
            
            # Standardize chunk to ensure identical amplitude before blending
            pred_chunk = (pred_chunk - np.mean(pred_chunk)) / (np.std(pred_chunk) + 1e-8)
            
            ppg_pred[i:i+seq_len] += pred_chunk * hann
            weight[i:i+seq_len] += hann
            
        # Handle the remainder if it doesn't divide evenly
        if T % stride != 0 and T > seq_len:
            chunk = tensor_input[:, :, -seq_len:, :, :]
            pred_chunk = model(chunk).squeeze(0).cpu().numpy()
            pred_chunk = (pred_chunk - np.mean(pred_chunk)) / (np.std(pred_chunk) + 1e-8)
            
            ppg_pred[-seq_len:] += pred_chunk * hann
            weight[-seq_len:] += hann
            
    # Normalize by the overlapping weights
    ppg_pred = ppg_pred / (weight + 1e-8)
    return ppg_pred


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray, fs: float = None) -> Dict[str, float]:
    if fs is None: fs = config.FS
    pcc_list, mae_list, rmse_list, snr_list = [], [], [], []
    
    for i in range(len(y_pred)):
        pred, true = y_pred[i], y_true[i]
        
        pcc = np.corrcoef(pred, true)[0, 1]
        pcc_list.append(0.0 if np.isnan(pcc) else pcc)
        
        bpm_pred = calculate_bpm_from_fft(pred, fs=fs)
        bpm_true = calculate_bpm_from_fft(true, fs=fs)
        
        mae_list.append(abs(bpm_pred - bpm_true))
        rmse_list.append((bpm_pred - bpm_true) ** 2)
        
        pred_detrended = pred - np.mean(pred)
        freqs, psd = welch(pred_detrended, fs=fs, nperseg=len(pred_detrended))
        
        gt_freq = bpm_true / 60.0
        signal_mask = (freqs >= (gt_freq - 0.15)) & (freqs <= (gt_freq + 0.15))
        noise_mask = ~signal_mask & (freqs >= 0.75) & (freqs <= 2.5)
        
        signal_power = np.sum(psd[signal_mask])
        noise_power = np.sum(psd[noise_mask]) + 1e-8
        snr_list.append(10 * np.log10(signal_power / noise_power))
    
    return {
        'PCC': float(np.mean(pcc_list)),
        'MAE': float(np.mean(mae_list)),
        'RMSE': float(math.sqrt(np.mean(rmse_list))),
        'SNR': float(np.mean(snr_list))
    }

def plot_training_curves(history: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]
    
    ax1.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['val_pccs'], label='Validation PCC', color='green', linewidth=2)
    ax2.set_title('Validation PCC', fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    ax3.plot(history['lr'], label='Learning Rate', color='purple', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_yscale('log')
    
    axes[1, 1].axis('off')
    plt.tight_layout(); plt.show()

def plot_prediction_comparison(pred_signal: np.ndarray, true_signal: np.ndarray, 
                              fs: float = None, title: str = None) -> None:
    if fs is None: fs = config.FS
    if title is None: title = "Predicted vs True PPG"
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(true_signal, label='True PPG', color='blue', linewidth=2, alpha=0.8)
    axes[0].plot(pred_signal, label='Predicted PPG', color='red', linewidth=2, alpha=0.8)
    axes[0].set_title(f'{title} - Time Domain', fontsize=14, fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    N = len(true_signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    true_fft, pred_fft = np.abs(np.fft.rfft(true_signal)), np.abs(np.fft.rfft(pred_signal))
    
    axes[1].plot(freqs, true_fft, label='True FFT', color='blue', linewidth=2, alpha=0.8)
    axes[1].plot(freqs, pred_fft, label='Predicted FFT', color='red', linewidth=2, alpha=0.8)
    axes[1].set_title('Frequency Domain - Full Spectrum'); axes[1].set_xlim(0, 4)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    mask = (freqs >= config.LOWCUT) & (freqs <= config.HIGHCUT)
    axes[2].plot(freqs[mask], true_fft[mask], label='True FFT', color='blue', linewidth=2, alpha=0.8)
    axes[2].plot(freqs[mask], pred_fft[mask], label='Predicted FFT', color='red', linewidth=2, alpha=0.8)
    axes[2].set_title('Frequency Domain - Heart Rate Range')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()

def infer_npz_file(model: nn.Module, npz_path: str, device: str = None, plot_results: bool = True) -> Dict[str, Any]:
    if device is None: device = config.DEVICE
    model = model.to(device)
    
    data = load_npz_file(npz_path)
    if data is None: return None
    
    roi_key = config.ROI_NAME
    actual_key = next((k for k in data.files if roi_key in k), None)
    if actual_key is None:
        for alt_key in ['roi_forehead', 'forehead', 'roi_head']:
            if alt_key in data:
                try: roi_data = data[alt_key]; break
                except: continue
        else: return None
    else: roi_data = data[actual_key]
    
    ppg_true = data['ppg_values'].astype(np.float32)
    if roi_data.ndim == 4: roi_data = np.transpose(roi_data, (3, 0, 1, 2))
    else: return None
    
    roi_data = roi_data.astype(np.float32) / 255.0
    roi_tensor = torch.tensor(roi_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # -------------------------------------------------------------
    # USING THE NEW SLIDING WINDOW PREDICTOR
    # -------------------------------------------------------------
    ppg_pred = predict_with_sliding_window(model, roi_tensor, seq_len=config.SEQUENCE_LENGTH)
    
    # Clean Reference Pipeline
    ppg_true = butter_bandpass_filter(ppg_true, fs=config.FS)
    ppg_std = np.std(ppg_true)
    if ppg_std < 1e-5: ppg_std = 1.0
    ppg_true_norm = (ppg_true - np.mean(ppg_true)) / ppg_std
    
    # Filter and Standardize Prediction
    ppg_pred_filtered = butter_bandpass_filter(ppg_pred, fs=config.FS)
    pred_std = np.std(ppg_pred_filtered)
    if pred_std < 1e-5: pred_std = 1.0
    ppg_pred_norm = (ppg_pred_filtered - np.mean(ppg_pred_filtered)) / pred_std
    
    ppg_pred_scaled = ppg_pred_norm * ppg_std + np.mean(ppg_true)
    ppg_true_denorm = ppg_true_norm * ppg_std + np.mean(ppg_true)
    
    pcc = np.corrcoef(ppg_pred_scaled, ppg_true_denorm)[0, 1]
    if np.isnan(pcc): pcc = 0.0
    
    bpm_pred = calculate_bpm_from_fft(ppg_pred_scaled, fs=config.FS)
    bpm_true = calculate_bpm_from_fft(ppg_true_denorm, fs=config.FS)
    mae = abs(bpm_pred - bpm_true)
    
    result = {'file': npz_path, 'ppg_pred': ppg_pred_scaled, 'ppg_true': ppg_true_denorm, 'bpm_pred': bpm_pred, 'bpm_true': bpm_true, 'pcc': pcc, 'mae': mae}
    logger.info(f"Inference: {Path(npz_path).name} | PCC={pcc:+.4f} | BPM Pred={bpm_pred:.2f} | BPM True={bpm_true:.2f} | MAE={mae:.2f}")
    if plot_results: plot_prediction_comparison(ppg_pred_scaled, ppg_true_denorm, fs=config.FS, title=f"{Path(npz_path).name} - PCC: {pcc:+.4f}")
    return result


def train_single_roi_pipeline():
    logger.info("=" * 70)
    logger.info("Initializing Single ROI PhysNet - PRODUCTION OVERLAP INFERENCE VERSION")
    logger.info("=" * 70)

    npz_files = sorted(list(config.DATA_DIR.glob('*.npz')))
    random.seed(42)
    shuffled_files = npz_files.copy()
    random.shuffle(shuffled_files)

    n_total = len(shuffled_files)
    n_train = max(1, int(n_total * config.TRAIN_RATIO))
    n_val = max(1, int(n_total * config.VAL_RATIO))

    train_dataset = SingleROIRPPGDataset(shuffled_files[:n_train], augment=True)
    val_dataset = SingleROIRPPGDataset(shuffled_files[n_train:n_train + n_val], augment=False)
    eval_dataset = SingleROIRPPGDataset(shuffled_files[n_train + n_val:], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    model = SingleROIPhysNet().to(config.DEVICE)
    criterion = RhythmLoss(fps=config.FS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)

    warmup_epochs = 5
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    best_val_pcc = -float('inf')
    patience_counter = 0
    best_model_weights = None
    train_losses, val_losses, val_pccs, learning_rates = [], [], [], []

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss, _ = criterion(outputs, targets)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any(): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss, val_pcc_accum = 0.0, 0.0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(config.DEVICE), val_targets.to(config.DEVICE)
                val_outputs = model(val_inputs)
                
                val_loss, val_pcc = criterion(val_outputs, val_targets)
                running_val_loss += val_loss.item() * val_inputs.size(0)
                val_pcc_accum += val_pcc * val_inputs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_pcc = val_pcc_accum / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_pccs.append(val_pcc)
        learning_rates.append(scheduler.get_last_lr()[0])

        logger.info(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:+.4f}")

        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            patience_counter = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'model_state_dict': model.state_dict(), 'config': config.__dict__}, config.MODEL_SAVE_PATH)
            logger.info(f"  --> Improved! Saved checkpoint (PCC: {val_pcc:+.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"\n[Early Stopping] Stopping at Epoch {epoch+1}")
                break

    if best_model_weights is not None: model.load_state_dict(best_model_weights)
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for eval_inputs, eval_targets in eval_loader:
            preds = model(eval_inputs.to(config.DEVICE)).cpu().numpy()
            for i in range(preds.shape[0]):
                p_std = np.std(preds[i])
                if p_std < 1e-5: p_std = 1.0
                preds[i] = (preds[i] - np.mean(preds[i])) / p_std
                
            all_preds.extend(preds)
            all_trues.extend(eval_targets.numpy())

    metrics = calculate_metrics(np.array(all_preds), np.array(all_trues))
    logger.info(f"\nFINAL METRICS -> PCC: {metrics['PCC']:.4f} | MAE: {metrics['MAE']:.2f} BPM | SNR: {metrics['SNR']:.2f} dB\n")

    plot_training_curves({'train_loss': train_losses, 'val_loss': val_losses, 'val_pccs': val_pccs, 'lr': learning_rates})

    for sample_file in random.sample(shuffled_files[n_train + n_val:], min(3, len(eval_dataset))):
        infer_npz_file(model, sample_file, device=config.DEVICE, plot_results=True)


# ============================================================================
# STANDALONE DEMO INFERENCE CELL (VIDEO TO WAVEFORM)
# ============================================================================

def run_video_inference_demo(video_path: str, model_path: str = None, max_frames: int = None):
    if max_frames is None: max_frames = config.TARGET_FRAMES
    if model_path is None: model_path = config.MODEL_SAVE_PATH
        
    print(f"\nInitiating video inference pipeline on: {video_path}")
    frames = load_video_frames_ffmpeg(video_path, target_frames=max_frames)
    if len(frames) == 0: return
    landmarker = init_mediapipe_landmarker()
    
    roi_patches = []
    display_box, display_frame_idx = None, 0
    
    for idx, frame_rgb in enumerate(frames):
        h, w, _ = frame_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image)
        roi_24x24 = np.zeros((config.ROI_SIZE[1], config.ROI_SIZE[0], 3), dtype=np.uint8)
        
        if hasattr(detection_result, 'face_landmarks') and len(detection_result.face_landmarks) > 0:
            landmarks = detection_result.face_landmarks[0]
            roi_24x24 = crop_roi_from_landmarks(frame_rgb, detection_result, 'forehead', config.ROI_SIZE)
            
            if display_box is None:
                xs = [int(landmarks[i].x * w) for i in ROI_LANDMARKS['forehead']]
                ys = [int(landmarks[i].y * h) for i in ROI_LANDMARKS['forehead']]
                if xs and ys:
                    display_box = (max(0, min(xs)-5), max(0, min(ys)-5), min(w, max(xs)+5), min(h, max(ys)+5))
                    display_frame_idx = idx
                    
        roi_patches.append(roi_24x24.astype(np.float32) / 255.0)
        
    if display_box is None: display_box = (int(w*0.4), int(h*0.15), int(w*0.6), int(h*0.35))
        
    tensor_input = torch.from_numpy(np.array(roi_patches, dtype=np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(config.DEVICE)
    
    model = SingleROIPhysNet().to(config.DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=False)['model_state_dict'])
        print(f"Loaded weights from '{model_path}'.")
    
    # Use overlapping sliding window to perfectly match training distribution
    pred_signal = predict_with_sliding_window(model, tensor_input, seq_len=config.SEQUENCE_LENGTH)
        
    filtered_rppg = butter_bandpass_filter(pred_signal, fs=config.FS)
    
    p_std = np.std(filtered_rppg)
    if p_std < 1e-5: p_std = 1.0
    filtered_rppg = (filtered_rppg - np.mean(filtered_rppg)) / p_std
    
    est_hr_fft = calculate_bpm_from_fft(filtered_rppg, fs=config.FS)
    
    sample_img = Image.fromarray(frames[display_frame_idx].copy())
    ImageDraw.Draw(sample_img).rectangle(list(display_box), outline="lime", width=3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(sample_img); ax1.set_title("Detected Forehead ROI"); ax1.axis('off')
    ax2.plot(filtered_rppg, color='crimson', label=f'rPPG Signal (FFT HR: {est_hr_fft:.1f} BPM)')
    ax2.set_title(f"Reconstructed Waveform for {os.path.basename(video_path)}")
    ax2.set_xlabel("Frame Index"); ax2.set_ylabel("Filtered Pulse Amplitude")
    ax2.grid(True, linestyle='--', alpha=0.5); ax2.legend(loc='upper right', fontsize=10)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    train_single_roi_pipeline()
    
    # UNCOMMENT BELOW TO TEST WITH A VIDEO AFTER TRAINING
    # test_video = "/home/cristic/data/Bgeorge/mcd_rppg/snapshots/929fb19c5ff2b5c8ed64a7c3a123744346674e88/video/9998_USBVideo_before.avi"
    # run_video_inference_demo(test_video, max_frames=450)
