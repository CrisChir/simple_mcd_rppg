# Sliding Window Model - Clinical Vitals Diagnostic System

This directory contains the Gradio Live Clinical Diagnostic Web App and Docker deployment configuration for the rPPG-based vital signs prediction system.

## 📁 Directory Structure

```
sliding_window_model/
├── gradio_live_clinical_diagnostic.ipynb  # Jupyter Notebook version
├── gradio_live_clinical_diagnostic.py    # Python script version
├── Dockerfile                              # Docker configuration
├── docker-compose.yml                      # Docker Compose configuration
├── entrypoint.sh                           # Entrypoint script
├── requirements.txt                        # Python dependencies
├── DOCKER_DEPLOYMENT.md                    # Detailed Docker deployment guide
├── models/                                 # Model checkpoints (not included in git)
│   ├── spatiotemporal_physnet_best_shuffle.pth
│   └── clinical_vitals_stage2_best.pth
└── README.md                               # This file
```

## 🚀 Quick Start

### Option 1: Run with Python (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python gradio_live_clinical_diagnostic.py

# Access at: http://localhost:7860
```

### Option 2: Run with Docker

```bash
# Build the image
docker-compose build

# Start the application
docker-compose up -d

# Access at: http://localhost:7860

# Stop the application
docker-compose down
```

### Option 3: Run with Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook

# Open gradio_live_clinical_diagnostic.ipynb and run all cells
```

## 📋 Requirements

### Python Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:

- `torch>=2.0.0` - PyTorch for deep learning
- `mediapipe>=0.10.0` - MediaPipe for face landmark detection
- `gradio>=3.44.0` - Gradio for web interface
- `opencv-python>=4.7.0` - OpenCV for video processing
- `numpy`, `scipy`, `matplotlib` - Numerical and visualization libraries

### Model Files

The following model files are required:

1. **`face_landmarker.task`** - MediaPipe face landmarker model
   - Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

2. **`spatiotemporal_physnet_best_shuffle.pth`** - Stage 1 model for PPG extraction
   - Should be in the `models/` directory

3. **`clinical_vitals_stage2_best.pth`** - Stage 2 model for vitals prediction
   - Should be in the `models/` directory

### Hardware Requirements

- **CPU**: Minimum 4 cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support recommended for best performance
- **Storage**: 5GB for Docker image, additional space for model files

## 🎯 Features

### Real-Time rPPG Extraction
- Uses MediaPipe Tasks API for efficient face landmark detection
- Extracts 10 facial regions of interest (ROIs)
- Processes video at 30 FPS for real-time performance

### Clinical Vitals Prediction
- **Stage 1**: SpatioTemporalPhysNet (3D-CNN) extracts PPG waveform from facial ROIs
- **Stage 2**: ClinicalVitalsRegressor predicts 10 clinical vital signs:
  - Pulse (BPM)
  - Respiratory rate (br/m)
  - Blood oxygen saturation (%)
  - Hemoglobin (g/dL)
  - Glycated hemoglobin (%)
  - Cholesterol (mg/dL)
  - Upper blood pressure (mmHg)
  - Lower blood pressure (mmHg)
  - Stress level (pts)
  - Rigidity (pts)

### Interactive Web Interface
- Upload video files or use webcam directly
- Real-time visualization of PPG waveform
- Comprehensive clinical health report
- Responsive design for desktop and mobile

## 📊 How It Works

### Processing Pipeline

```
Video Input
    ↓
MediaPipe Face Landmark Detection
    ↓
Extract 10 Facial ROIs (24x24 pixels each)
    ↓
Stage 1: SpatioTemporalPhysNet (3D-CNN)
    ↓
PPG Waveform Extraction
    ↓
Bandpass Filtering (0.75-2.5 Hz)
    ↓
Stage 2: ClinicalVitalsRegressor
    ↓
Clinical Vitals Prediction
    ↓
Interactive Visualization & Report
```

### Model Architecture

#### Stage 1: SpatioTemporalPhysNet
```
Input: (Batch, 30, T, 24, 24)  # 10 ROIs × 3 channels, T frames
    ↓
3D Spatial Convolution (64 filters, 1x3x3 kernel)
    ↓
Batch Normalization + ELU
    ↓
Adaptive Average Pooling
    ↓
1D Temporal Convolution (32, 16 filters)
    ↓
Output: (Batch, T)  # PPG waveform
```

#### Stage 2: ClinicalVitalsRegressor
```
Input: (Batch, 1, 450)  # PPG waveform
    ↓
1D Conv Feature Extractor (16, 32 filters)
    ↓
Adaptive Average Pooling + Flatten
    ↓
Fully Connected Layers (64 units)
    ↓
Output: (Batch, 10)  # 10 vital signs
```

## 🔧 Configuration

### Environment Variables

You can configure the application using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server host address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `GRADIO_SHARE` | `False` | Enable public sharing |
| `MODEL_DIR` | `./sliding_window_model/models` | Directory containing model checkpoints |
| `FACE_MODEL_PATH` | `/app/face_landmarker.task` | Path to MediaPipe face landmarker model |
| `MAX_FRAMES` | `450` | Maximum frames to process |
| `CUDA_DEVICE` | `0` | CUDA device index |

### Example Configuration

```bash
# Set environment variables before running
export GRADIO_SERVER_PORT=8080
export MAX_FRAMES=300
export CUDA_DEVICE=1

# Then run the application
python gradio_live_clinical_diagnostic.py
```

## 🐳 Docker Deployment

For detailed Docker deployment instructions, see `DOCKER_DEPLOYMENT.md`.

### Quick Docker Commands

```bash
# Build the image
docker build -t clinical-vitals-app .

# Run with GPU support
docker run --gpus all -p 7860:7860 clinical-vitals-app

# Run with CPU only
docker run -p 7860:7860 clinical-vitals-app

# Run with Docker Compose
docker-compose up -d
```

### Docker Volume Mounts

To use your local model files:

```bash
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/models:/app/sliding_window_model/models:ro \
  -v $(pwd)/face_landmarker.task:/app/face_landmarker.task:ro \
  clinical-vitals-app
```

## 📈 Usage Tips

### For Best Results

1. **Lighting**: Ensure good, even lighting on your face
2. **Position**: Face the camera directly, fill most of the frame
3. **Stability**: Keep your face relatively still during recording
4. **Duration**: Record for at least 15-30 seconds for accurate results
5. **Background**: Use a plain background to help with face detection

### Troubleshooting

**No face detected:**
- Check lighting conditions
- Ensure your face is visible in the frame
- Try moving closer to the camera

**Slow performance:**
- Use a smaller `MAX_FRAMES` value
- Ensure GPU acceleration is enabled
- Close other resource-intensive applications

**Model loading errors:**
- Verify model files are in the correct location
- Check file permissions
- Ensure model files are not corrupted

## 📚 Technical Details

### Signal Processing

The system uses a **Butterworth bandpass filter** (0.75-2.5 Hz) to isolate the heart rate signal from the PPG waveform. This corresponds to a heart rate range of 45-150 BPM.

### Heart Rate Calculation

Heart rate is calculated using **Fast Fourier Transform (FFT)** on the filtered PPG signal. The peak frequency in the 0.75-2.5 Hz range is identified and converted to BPM.

### Clinical Vitals Normalization

Stage 2 model outputs are normalized using statistics from the training data:
- Mean and standard deviation for each vital sign
- Denormalization formula: `raw_value = (normalized_value * std) + mean`

## 🔬 Validation and Testing

### Testing the System

1. **Face Detection Test**: Verify that MediaPipe can detect faces in your video
2. **ROI Extraction Test**: Check that all 10 facial regions are correctly extracted
3. **PPG Waveform Test**: Visualize the extracted PPG waveform for quality
4. **Vitals Prediction Test**: Verify that predicted values are within reasonable ranges

### Expected Output Ranges

| Vital Sign | Normal Range | Unit |
|------------|--------------|------|
| Pulse | 60-100 | BPM |
| Respiratory Rate | 12-20 | br/m |
| Blood Oxygen | 95-100 | % |
| Hemoglobin (Male) | 13.8-17.2 | g/dL |
| Hemoglobin (Female) | 12.1-15.1 | g/dL |
| Glycated Hemoglobin | <5.7 | % |
| Cholesterol | <200 | mg/dL |
| Upper BP | 90-120 | mmHg |
| Lower BP | 60-80 | mmHg |

## 📝 Citation

If you use this work in your research, please cite the original repository:

```bibtex
@misc{simple_mcd_rppg,
  author = {CrisChir},
  title = {Simple MCD rPPG: Multi-Channel Deep Learning for Remote Photoplethysmography},
  year = {2024},
  howpublished = {\url{https://github.com/CrisChir/simple_mcd_rppg}},
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file in the repository root for details.

## 🆘 Support

For questions, issues, or feature requests:

1. Check the documentation in this README
2. Review the Docker deployment guide
3. Open an issue in the GitHub repository
4. For urgent matters, contact the repository maintainer

## 🔗 Related Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gradio Documentation](https://gradio-app.github.io/gradio/)
- [Remote PPG Research Papers](https://scholar.google.com/scholar?q=remote+photoplethysmography)

---

*Last updated: 2024*
*Version: 1.0*
