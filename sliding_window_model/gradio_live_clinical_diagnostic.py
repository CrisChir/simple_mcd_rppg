#!/usr/bin/env python3
"""
Gradio Live Clinical Diagnostic Web App
============================================================================
Powered by MediaPipe FaceLandmarker Task, Spatio-Temporal 3D-CNN & Stage 2 Vitals AI

This script provides a complete web application for real-time clinical vitals 
prediction using camera-based photoplethysmography (rPPG).

Usage:
    python gradio_live_clinical_diagnostic.py

Environment Variables:
    GRADIO_SERVER_NAME: Server host (default: 0.0.0.0)
    GRADIO_SERVER_PORT: Server port (default: 7860)
    GRADIO_SHARE: Enable public sharing (default: False)
    MODEL_DIR: Directory containing model checkpoints
    FACE_MODEL_PATH: Path to MediaPipe face landmarker model
    MAX_FRAMES: Maximum frames to process (default: 450)
    CUDA_DEVICE: CUDA device index (default: 0)
"""

import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, find_peaks, butter, filtfilt
import gradio as gr

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment variables
GRADIO_SERVER_NAME = os.getenv('GRADIO_SERVER_NAME', '0.0.0.0')
GRADIO_SERVER_PORT = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
GRADIO_SHARE = os.getenv('GRADIO_SHARE', 'False').lower() == 'true'
MODEL_DIR = Path(os.getenv('MODEL_DIR', './sliding_window_model/models'))
FACE_MODEL_PATH = os.getenv('FACE_MODEL_PATH', '/app/face_landmarker.task')
MAX_FRAMES = int(os.getenv('MAX_FRAMES', '450'))
CUDA_DEVICE = int(os.getenv('CUDA_DEVICE', '0'))

# Device configuration
DEVICE = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ROI Configuration - 10 facial regions matching 468-point mesh
ROI_ORDER = [
    'roi_forehead',
    'roi_left_cheek_280', 
    'roi_right_cheek_50',
    'roi_nose',
    'roi_chin',
    'roi_chin_199',
    'roi_left_eye',
    'roi_right_eye',
    'roi_mouth',
    'roi_full_face'
]

# ROI Landmark indices for MediaPipe 468-point face mesh
ROI_LANDMARKS = {
    'roi_forehead': [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109],
    'roi_left_cheek_280': [280, 352, 376, 430, 411, 340],
    'roi_right_cheek_50': [50, 123, 147, 210, 187, 111],
    'roi_nose': [1, 2, 98, 327, 168, 6, 197],
    'roi_chin': [152, 377, 400, 378, 148, 176, 149],
    'roi_chin_199': [199, 200, 201, 208, 428, 421],
    'roi_left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'roi_right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263],
    'roi_mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'roi_full_face': [10, 152, 234, 454]
}

# Model paths
STAGE1_CHECKPOINT = MODEL_DIR / "spatiotemporal_physnet_best_shuffle.pth"
STAGE2_CHECKPOINT = MODEL_DIR / "clinical_vitals_stage2_best.pth"

# Video processing parameters
CROP_SIZE = (24, 24)
SAMPLING_RATE = 30.0  # FPS

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SpatioTemporalPhysNet(nn.Module):
    """
    Spatio-Temporal convolutional network for extracting micro-color variations
    across 10 face zones (30 raw input channels).
    
    Input: (Batch, 30, T, 24, 24) - 10 ROIs × 3 channels, T frames
    Output: (Batch, T) - PPG waveform
    """
    def __init__(self, num_rois=10, in_channels=3):
        super(SpatioTemporalPhysNet, self).__init__()
        self.num_rois = num_rois
        self.total_channels = num_rois * in_channels
        
        # Spatial convolution to extract features from each ROI
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(self.total_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # Temporal encoder to process the time series
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # x: (Batch, 30, T, 24, 24)
        x_spatial = self.spatial_conv(x)  # (Batch, 64, T, 1, 1)
        x_spatial = torch.squeeze(x_spatial, dim=-1)  # (Batch, 64, T, 1)
        x_spatial = torch.squeeze(x_spatial, dim=-1)  # (Batch, 64, T)
        
        # Permute for Conv1d: (Batch, 64, T) -> (Batch, 64, T)
        ppg_pred = self.temporal_encoder(x_spatial)  # (Batch, 1, T)
        return torch.squeeze(ppg_pred, dim=1)  # (Batch, T)


class ClinicalVitalsRegressor(nn.Module):
    """
    Stage 2 Model: Takes the predicted 450-frame PPG waveform and predicts 
    clinical vital sign scalars.
    
    Input: (Batch, 1, 450) - PPG waveform
    Output: (Batch, num_vitals) - Predicted vitals
    """
    def __init__(self, in_features=450, num_vitals=10):
        super(ClinicalVitalsRegressor, self).__init__()
        
        # 1D Conv feature extractor for PPG waveform morphology
        self.wave_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),  # (B, 16, 225)
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),  # (B, 32, 113)
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),                              # (B, 32, 1)
            nn.Flatten()                                          # (B, 32)
        )
        
        # Fully Connected Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_vitals)
        )
    
    def forward(self, predicted_ppg_wave):
        # predicted_ppg_wave shape: (Batch, 1, 450) or (Batch, 450)
        if predicted_ppg_wave.dim() == 2:
            predicted_ppg_wave = predicted_ppg_wave.unsqueeze(1)
        
        features = self.wave_feature_extractor(predicted_ppg_wave)
        vitals_pred = self.regressor(features)
        return vitals_pred


# ============================================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================================

def butter_bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30.0, order=4):
    """
    Apply zero-phase Butterworth bandpass filter to capture human heart rate (45-150 BPM).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def calculate_bpm_from_fft(signal, fs=30.0):
    """
    Calculate heart rate in BPM from PPG signal using FFT.
    """
    signal_detrended = signal - np.mean(signal)
    n = len(signal_detrended)
    
    # Compute FFT
    fft_vals = np.fft.fft(signal_detrended)
    fft_freq = np.fft.fftfreq(n, d=1.0/fs)
    
    # Only positive frequencies
    pos_mask = np.where(fft_freq > 0)
    fft_freq = fft_freq[pos_mask]
    fft_vals = np.abs(fft_vals[pos_mask])
    
    # Frequency range for human heart rate: 45-150 BPM = 0.75-2.5 Hz
    freq_mask = (fft_freq >= 0.75) & (fft_freq <= 2.5)
    
    if np.sum(freq_mask) == 0:
        return 60.0  # Default heart rate
    
    fft_freq_filtered = fft_freq[freq_mask]
    fft_vals_filtered = fft_vals[freq_mask]
    
    # Find peak frequency
    peak_idx = np.argmax(fft_vals_filtered)
    peak_freq = fft_freq_filtered[peak_idx]
    
    # Convert to BPM
    bpm = peak_freq * 60.0
    return float(bpm)


def detect_heartbeats(ppg_signal, fs=30.0, min_distance=15, min_height=0.2):
    """
    Detect heartbeats (peaks) in PPG signal.
    """
    peaks, _ = find_peaks(ppg_signal, distance=min_distance, height=min_height)
    return peaks


# ============================================================================
# MEDIAPIPE FACE LANDMARK DETECTION
# ============================================================================

def initialize_mediapipe_landmarker(model_path=FACE_MODEL_PATH):
    """
    Initialize MediaPipe FaceLandmarker task.
    """
    if not os.path.exists(model_path):
        print(f"Warning: MediaPipe model not found at {model_path}")
        # Try alternative paths
        alt_paths = [
            "/home/cristic/face_landmarker.task",
            "/app/face_landmarker.task",
            "./face_landmarker.task"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if not os.path.exists(model_path):
        print("Error: MediaPipe face landmarker model not found!")
        print("Please download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return None
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f"✅ MediaPipe FaceLandmarker initialized with model: {model_path}")
        return landmarker
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        return None


def crop_roi_from_landmarks(frame, landmarks, landmark_indices, crop_size=CROP_SIZE):
    """
    Crop a specific facial region using MediaPipe landmark coordinates.
    """
    h, w, c = frame.shape
    
    # Get landmark coordinates
    coords = np.array([
        (int(landmarks[idx].x * w), int(landmarks[idx].y * h)) 
        for idx in landmark_indices
    ])
    
    # Calculate bounding box
    x_min, y_min = np.clip(np.min(coords, axis=0), 0, [w, h])
    x_max, y_max = np.clip(np.max(coords, axis=0), 0, [w, h])
    
    # Handle edge cases
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
    
    # Crop and resize
    roi_crop = frame[y_min:y_max, x_min:x_max]
    roi_resized = cv2.resize(roi_crop, crop_size, interpolation=cv2.INTER_AREA)
    
    return roi_resized


def process_video_with_mediapipe(video_path, landmarker, max_frames=MAX_FRAMES):
    """
    Process video to extract facial ROIs using MediaPipe.
    """
    # Handle different input types
    if isinstance(video_path, str):
        # File path
        cap = cv2.VideoCapture(video_path)
    elif hasattr(video_path, 'name'):
        # File-like object
        cap = cv2.VideoCapture(video_path.name)
    else:
        print("❌ Unsupported video input type")
        return None, 0
    
    frames_rois = {key: [] for key in ROI_ORDER}
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect landmarks
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # Extract all ROIs
            for roi_name in ROI_ORDER:
                indices = ROI_LANDMARKS[roi_name]
                roi_24x24 = crop_roi_from_landmarks(frame_rgb, landmarks, indices)
                frames_rois[roi_name].append(roi_24x24 / 255.0)
        else:
            # No face detected - add zeros
            for roi_name in ROI_ORDER:
                frames_rois[roi_name].append(np.zeros((24, 24, 3), dtype=np.float32))
        
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        print("❌ No frames processed")
        return None, 0
    
    # Stack ROIs: (10, T, 24, 24, 3) -> (10, 3, T, 24, 24) -> (30, T, 24, 24)
    roi_list = [np.array(frames_rois[roi_name], dtype=np.float32) for roi_name in ROI_ORDER]
    stacked = np.stack(roi_list, axis=0)  # (10, T, 24, 24, 3)
    stacked = np.transpose(stacked, (0, 4, 1, 2, 3))  # (10, 3, T, 24, 24)
    stacked = np.reshape(stacked, (30, frame_count, 24, 24))  # (30, T, 24, 24)
    
    # Convert to tensor and add batch dimension
    tensor_input = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    print(f"✅ Processed {frame_count} frames with face landmarks")
    return tensor_input, frame_count


# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================

def load_stage1_model(checkpoint_path=STAGE1_CHECKPOINT, device=DEVICE):
    """
    Load Stage 1 SpatioTemporalPhysNet model.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Stage 1 checkpoint not found: {checkpoint_path}")
        print("Please ensure the model is in the correct location.")
        return None
    
    model = SpatioTemporalPhysNet(num_rois=10).to(device)
    
    try:
        # Try loading with weights_only=False first
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying weights_only=True...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            return None
    
    model.eval()
    print(f"✅ Stage 1 model loaded from {checkpoint_path}")
    return model


def load_stage2_model(checkpoint_path=STAGE2_CHECKPOINT, device=DEVICE):
    """
    Load Stage 2 ClinicalVitalsRegressor model.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Stage 2 checkpoint not found: {checkpoint_path}")
        print("Please ensure the model is in the correct location.")
        return None, None, None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading Stage 2 checkpoint: {e}")
        return None, None, None, None
    
    # Extract metadata
    vital_keys = checkpoint.get('vital_keys', [
        'vital_pulse', 'vital_respiratory', 'vital_saturation', 
        'vital_hemoglobin', 'vital_glycated_hemoglobin', 'vital_cholesterol',
        'vital_upper_ap', 'vital_lower_ap', 'vital_stress', 'vital_rigidity'
    ])
    
    vital_stats = checkpoint.get('vital_stats', {'mean': np.zeros(len(vital_keys)), 'std': np.ones(len(vital_keys))})
    vital_units = checkpoint.get('vital_units', ['BPM'] * len(vital_keys))
    
    # Initialize model
    num_vitals = len(vital_keys)
    model = ClinicalVitalsRegressor(in_features=450, num_vitals=num_vitals).to(device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading Stage 2 model weights: {e}")
        return None, None, None, None
    
    model.eval()
    print(f"✅ Stage 2 model loaded from {checkpoint_path}")
    print(f"   Vitals to predict: {vital_keys}")
    print(f"   Units: {vital_units}")
    
    return model, vital_stats, vital_keys, vital_units


# ============================================================================
# PREDICTION PIPELINE
# ============================================================================

def predict_vitals_from_video(video_path):
    """
    Complete prediction pipeline: video -> PPG waveform -> clinical vitals.
    """
    if video_path is None:
        print("❌ No video input provided")
        return None, None, 0
    
    # Step 1: Extract ROIs using MediaPipe
    input_tensor, total_frames = process_video_with_mediapipe(video_path, landmarker, max_frames=MAX_FRAMES)
    
    if input_tensor is None:
        print("❌ No face detected in video")
        return None, None, 0
    
    # Step 2: Stage 1 - Extract PPG waveform
    with torch.no_grad():
        raw_output_ppg = model_stage1(input_tensor).cpu().numpy()[0]
    
    # Step 3: Filter PPG waveform
    pad_m = 30
    pred_padded = np.pad(raw_output_ppg, pad_m, mode='reflect')
    pred_filtered = butter_bandpass_filter(pred_padded - np.mean(pred_padded), fs=SAMPLING_RATE)
    pred_cropped = pred_filtered[pad_m:-pad_m]
    ppg_clean = (pred_cropped - np.mean(pred_cropped)) / (np.std(pred_cropped) + 1e-8)
    
    # Ensure we have exactly 450 frames for Stage 2
    if len(ppg_clean) < 450:
        ppg_for_s2 = np.pad(ppg_clean, (0, 450 - len(ppg_clean)), mode='reflect')
    else:
        ppg_for_s2 = ppg_clean[:450]
    
    # Step 4: Stage 2 - Predict clinical vitals
    ppg_s2_tensor = torch.tensor(ppg_for_s2, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(DEVICE)
    
    with torch.no_grad():
        vitals_norm = model_stage2(ppg_s2_tensor).cpu().numpy()[0]
    
    # Denormalize vitals
    vitals_raw = (vitals_norm * np.array(VITAL_STATS['std'])) + np.array(VITAL_STATS['mean'])
    
    return ppg_clean, vitals_raw, total_frames


def create_ppg_plot(ppg_clean, frame_count):
    """
    Create matplotlib figure for PPG waveform visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot PPG waveform
    ax.plot(ppg_clean, color='#2ca02c', linewidth=2.2, label='Predicted rPPG Waveform')
    
    # Detect and plot peaks (heartbeats)
    peaks = detect_heartbeats(ppg_clean)
    if len(peaks) > 0:
        ax.plot(peaks, ppg_clean[peaks], "ro", markersize=6, label=f'Detected Cardiac Beats ({len(peaks)})')
    
    # Calculate and display BPM
    bpm = calculate_bpm_from_fft(ppg_clean, fs=SAMPLING_RATE)
    
    ax.set_title(f"Real-Time Camera Photoplethysmogram (rPPG Waveform) | BPM: {bpm:.1f}", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Frames Timeline ({SAMPLING_RATE} FPS) | Total Frames: {frame_count}", fontsize=10)
    ax.set_ylabel("Normalized Amplitude", fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def create_medical_report_html(vitals_raw, vitals_keys, vitals_units):
    """
    Create HTML medical health report.
    """
    html_card = """
    <div style='background-color: #f8f9fa; border-radius: 12px; padding: 20px; border: 1px solid #dee2e6;'>
        <h2 style='color: #2c3e50; margin-top: 0; border-bottom: 2px solid #007bff; padding-bottom: 8px;'>
            🏥 Camera Clinical Diagnostic Health Report
        </h2>
        <table style='width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;'>
            <thead>
                <tr style='background-color: #007bff; color: white;'>
                    <th style='padding: 10px; text-align: left;'>Vital Sign / Parameter</th>
                    <th style='padding: 10px; text-align: left;'>Predicted Value</th>
                    <th style='padding: 10px; text-align: left;'>Medical Unit</th>
                    <th style='padding: 10px; text-align: left;'>Clinical Status</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for idx, (key, val, unit) in enumerate(zip(vitals_keys, vitals_raw, vitals_units)):
        # Format vital name
        name = key.replace('vital_', '').replace('_', ' ').title()
        
        # Determine status (simple thresholds for demonstration)
        status_badge = "<span style='color: green; font-weight: bold;'>● Normal</span>"
        bg_color = "#ffffff" if idx % 2 == 0 else "#f1f3f5"
        
        # Special handling for known vitals
        if 'pulse' in key.lower():
            if val < 60 or val > 100:
                status_badge = "<span style='color: orange; font-weight: bold;'>⚠ Brady/Tachy</span>"
        elif 'saturation' in key.lower():
            if val < 95:
                status_badge = "<span style='color: red; font-weight: bold;'>● Low</span>"
        elif 'upper' in key.lower() and 'ap' in key.lower():
            if val > 140:
                status_badge = "<span style='color: red; font-weight: bold;'>● High</span>"
            elif val > 120:
                status_badge = "<span style='color: orange; font-weight: bold;'>⚠ Elevated</span>"
        elif 'lower' in key.lower() and 'ap' in key.lower():
            if val < 60:
                status_badge = "<span style='color: red; font-weight: bold;'>● Low</span>"
            elif val < 80:
                status_badge = "<span style='color: orange; font-weight: bold;'>⚠ Low</span>"
        
        html_card += f"""
            <tr style='background-color: {bg_color}; border-bottom: 1px solid #e9ecef;'>
                <td style='padding: 10px; font-weight: bold;'>{name}</td>
                <td style='padding: 10px; color: #0056b3; font-size: 16px; font-weight: bold;'>{val:.2f}</td>
                <td style='padding: 10px; color: #6c757d;'>{unit}</td>
                <td style='padding: 10px;'>{status_badge}</td>
            </tr>
        """
    
    html_card += """
            </tbody>
        </table>
        <p style='color: #6c757d; font-size: 11px; margin-top: 15px;'>
            * AI-assisted contactless video diagnostic report. Powered by MediaPipe Tasks API & 3D-CNN.<br>
            ** Note: This is a research prototype. Not for clinical use without validation.
        </p>
    </div>
    """
    
    return html_card


def predict_vitals_gradio_pipeline(video_path):
    """
    Gradio-compatible prediction function.
    """
    if video_path is None:
        return None, "<div style='color:red;'>Please upload or record a video.</div>"
    
    try:
        # Run prediction pipeline
        ppg_clean, vitals_raw, frame_count = predict_vitals_from_video(video_path)
        
        if ppg_clean is None:
            return None, "<div style='color:red;'>Error: No face detected in video! Please ensure your face is visible and well-lit.</div>"
        
        # Create visualization
        fig = create_ppg_plot(ppg_clean, frame_count)
        
        # Create medical report
        html_report = create_medical_report_html(vitals_raw, VITAL_KEYS, VITAL_UNITS)
        
        return fig, html_report
        
    except Exception as e:
        error_msg = f"<div style='color:red;'>Error during prediction: {str(e)}</div>"
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to initialize and launch the application.
    """
    global landmarker, model_stage1, model_stage2, VITAL_STATS, VITAL_KEYS, VITAL_UNITS
    
    print("\n" + "="*60)
    print("CLINICAL VITALS DIAGNOSTIC SYSTEM")
    print("="*60)
    
    # Initialize MediaPipe landmarker
    print("\nInitializing MediaPipe FaceLandmarker...")
    landmarker = initialize_mediapipe_landmarker()
    
    if landmarker is None:
        print("❌ Failed to initialize MediaPipe FaceLandmarker")
        print("Please ensure the face_landmarker.task file is available")
        return
    
    # Load Stage 1 model
    print("\nLoading Stage 1 model...")
    model_stage1 = load_stage1_model()
    
    if model_stage1 is None:
        print("❌ Failed to load Stage 1 model")
        print(f"Expected at: {STAGE1_CHECKPOINT}")
        return
    
    # Load Stage 2 model
    print("\nLoading Stage 2 model...")
    model_stage2, VITAL_STATS, VITAL_KEYS, VITAL_UNITS = load_stage2_model()
    
    if model_stage2 is None:
        print("❌ Failed to load Stage 2 model")
        print(f"Expected at: {STAGE2_CHECKPOINT}")
        return
    
    print("\n" + "="*60)
    print("✅ ALL SYSTEMS READY!")
    print("="*60)
    
    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Clinical Vitals Diagnostic") as demo:
        gr.Markdown(
            """
            # 🏥 Camera-Based Clinical Vitals & rPPG Diagnostic System
            ### Powered by MediaPipe FaceLandmarker Task, Spatio-Temporal 3D-CNN & Stage 2 Vitals AI
            
            **Instructions:**
            - Upload a video file (MP4) or use your webcam to record a 15-30 second clip
            - Ensure your face is well-lit and visible in the frame
            - Keep your face relatively still for best results
            - Click "Execute Clinical Analysis" to process the video
            
            **Note:** This is a research prototype. Results are for demonstration purposes only.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video Input")
                video_input = gr.Video(
                    label="Record Webcam or Upload Video MP4",
                    sources=["webcam", "upload"],
                    format="mp4",
                    mirror_webcam=True
                )
                analyze_btn = gr.Button(
                    "🔍 Execute Clinical Analysis", 
                    variant="primary", 
                    size="lg"
                )
                
                # Add model status info
                gr.Markdown(f"""
                **Model Status:**
                - ✅ MediaPipe FaceLandmarker: Loaded
                - ✅ Stage 1 (PPG Extraction): Loaded
                - ✅ Stage 2 (Vitals Prediction): Loaded
                
                **Device:** {DEVICE}
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                wave_plot = gr.Plot(label="Extracted rPPG BVP Waveform")
                report_output = gr.HTML(label="Clinical Health Report")
        
        # Connect button to prediction function
        analyze_btn.click(
            fn=predict_vitals_gradio_pipeline,
            inputs=[video_input],
            outputs=[wave_plot, report_output]
        )
        
        # Add footer
        gr.Markdown(
            """
            ---
            **About:** This system uses advanced computer vision and deep learning to extract vital signs from video.
            The technology is based on remote photoplethysmography (rPPG), which detects subtle color changes in the skin.
            
            **Citation:** If you use this work, please cite the original repository.
            
            **Contact:** For questions or issues, please refer to the project documentation.
            """
        )
    
    # Launch the application
    print(f"\nStarting Gradio server on {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    print("Access the application at: http://localhost:7860")
    print("(Press Ctrl+C to stop)")
    
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        debug=True
    )


if __name__ == "__main__":
    main()
