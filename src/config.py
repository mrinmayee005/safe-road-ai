"""
Safe Road AI - Phase 1 Configuration Module
Centralizes paths, hyperparameters, model configurations, and experimental constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_VIDEOS_DIR = RAW_DATA_DIR / "videos"
RAW_SENSORS_DIR = RAW_DATA_DIR / "sensors"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# User CCD Real Dashcam Paths
ARCHIVE_DIR = PROJECT_ROOT / "archive"
CCD_DATA_DIR = DATA_DIR / "ccd"
CCD_VIDEOS_DIR = CCD_DATA_DIR / "videos"
CCD_SENSORS_DIR = CCD_DATA_DIR / "sensors"

# Real-World Telematics Benchmark Paths
TELEMATICS_DATA_DIR = DATA_DIR / "telematics"
TELEMATICS_VIDEOS_DIR = TELEMATICS_DATA_DIR / "videos"
TELEMATICS_SENSORS_DIR = TELEMATICS_DATA_DIR / "sensors"

for d in [
    RAW_VIDEOS_DIR, RAW_SENSORS_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    CCD_VIDEOS_DIR, CCD_SENSORS_DIR, TELEMATICS_VIDEOS_DIR, TELEMATICS_SENSORS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# Video Processing Parameters
VIDEO_CONFIG = {
    "target_size": (224, 224),       # MobileNetV3 / ResNet input resolution
    "sample_fps": 5,                 # Sampled frames per second (down from 30 to save compute)
    "batch_size": 16,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "models": ["mobilenet_v3_small", "resnet18"],
    "primary_model": "mobilenet_v3_small"
}

# Sensor Processing Parameters
SENSOR_CONFIG = {
    "sampling_rate_hz": 50,          # IMU sampling frequency (50 Hz = 20ms delta)
    "window_size_sec": 1.0,          # 1.0 second analysis window (50 samples)
    "window_stride_sec": 0.2,        # 200ms step between windows (80% overlap for rapid detection)
    "models": ["random_forest", "gradient_boosting", "extra_trees"],
    "primary_model": "random_forest",
    "random_state": 42
}

# Fusion & Decision Parameters
FUSION_CONFIG = {
    "default_alpha": 0.55,           # Video weight in Pfinal = alpha * Pv + (1 - alpha) * Ps
    "default_threshold": 0.50,       # Decision threshold T
    "alpha_search_range": (0.0, 1.0, 0.05),
}

# Temporal Confirmation Parameters
TEMPORAL_CONFIG = {
    "smoothing_window_size": 3,      # Number of consecutive prediction windows to smooth
    "persistence_required": 2,       # Minimum positive detections in smoothing window to trigger alert
    "cooldown_seconds": 5.0          # Minimum time before triggering a second independent alert
}

# GPS Mock Default Coordinates (e.g. for emergency alert simulator)
DEFAULT_GPS = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "location_name": "MG Road, Bangalore"
}
