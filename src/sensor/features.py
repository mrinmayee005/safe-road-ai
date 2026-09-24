"""
Safe Road AI - Sensor Feature Engineering Module
Computes orientation-independent acceleration and angular velocity magnitudes,
signal derivatives (jerk), and statistical windowed features.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import SENSOR_CONFIG


def compute_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes orientation-independent acceleration magnitude A and
    gyroscope angular velocity magnitude G:
        A = sqrt(ax^2 + ay^2 + az^2)
        G = sqrt(gx^2 + gy^2 + gz^2)
    Also computes numerical derivatives (jerk):
        jerk_A = d(A)/dt
        jerk_G = d(G)/dt
    """
    df = df.copy()
    
    # Orientation-independent magnitudes
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    
    # Rate of change (jerk proxy) at 50 Hz (dt = 0.02s)
    df['acc_jerk'] = np.abs(np.gradient(df['acc_mag'].values, 0.02))
    df['gyro_jerk'] = np.abs(np.gradient(df['gyro_mag'].values, 0.02))
    
    # Fill any edge NaNs/Infs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def extract_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts structured statistical and physical features from a 1-second (or custom) sensor window:
    - Mean, Std, Max, Min, Range, Median, Energy, RMS, 90th percentile for Acc and Gyro
    - Maximum Jerk
    - Directional peak accelerations (lateral, longitudinal, vertical, yaw)
    """
    a = window_df['acc_mag'].values
    g = window_df['gyro_mag'].values
    jerk_a = window_df['acc_jerk'].values
    jerk_g = window_df['gyro_jerk'].values
    
    features = {
        # Acceleration Magnitude Features
        'acc_mean': float(np.mean(a)),
        'acc_std': float(np.std(a)),
        'acc_max': float(np.max(a)),
        'acc_min': float(np.min(a)),
        'acc_range': float(np.ptp(a)),
        'acc_median': float(np.median(a)),
        'acc_energy': float(np.mean(a ** 2)),
        'acc_rms': float(np.sqrt(np.mean(a ** 2))),
        'acc_p90': float(np.percentile(a, 90)),
        'acc_jerk_max': float(np.max(jerk_a)),
        'acc_jerk_mean': float(np.mean(jerk_a)),

        # Gyroscope Magnitude Features
        'gyro_mean': float(np.mean(g)),
        'gyro_std': float(np.std(g)),
        'gyro_max': float(np.max(g)),
        'gyro_min': float(np.min(g)),
        'gyro_range': float(np.ptp(g)),
        'gyro_median': float(np.median(g)),
        'gyro_energy': float(np.mean(g ** 2)),
        'gyro_rms': float(np.sqrt(np.mean(g ** 2))),
        'gyro_p90': float(np.percentile(g, 90)),
        'gyro_jerk_max': float(np.max(jerk_g)),

        # Directional Extreme Features
        'ax_max_abs': float(np.max(np.abs(window_df['ax']))),
        'ay_max_abs': float(np.max(np.abs(window_df['ay']))),
        'az_max_abs': float(np.max(np.abs(window_df['az']))),
        'gz_max_abs': float(np.max(np.abs(window_df['gz']))),  # Yaw rotation
    }
    return features


def process_sensor_timeseries(
    sensor_df: pd.DataFrame,
    window_size_sec: float = None,
    window_stride_sec: float = None,
    sampling_hz: int = None
) -> Tuple[pd.DataFrame, List[float], List[int]]:
    """
    Takes a raw sensor dataframe, computes magnitudes and jerk, and segments
    into overlapping windows to create tabular feature rows.
    
    Returns:
        features_df: DataFrame of extracted features for each window
        window_timestamps: Center timestamp (seconds) of each window
        labels: Ground-truth label for each window (1 if accident, 0 if normal)
    """
    window_size_sec = window_size_sec or SENSOR_CONFIG["window_size_sec"]
    window_stride_sec = window_stride_sec or SENSOR_CONFIG["window_stride_sec"]
    sampling_hz = sampling_hz or SENSOR_CONFIG["sampling_rate_hz"]
    
    window_samples = int(window_size_sec * sampling_hz)
    stride_samples = max(1, int(window_stride_sec * sampling_hz))
    
    df_prepared = compute_magnitudes(sensor_df)
    n_samples = len(df_prepared)
    
    feature_rows = []
    window_timestamps = []
    labels = []
    
    for start_idx in range(0, n_samples - window_samples + 1, stride_samples):
        end_idx = start_idx + window_samples
        w_df = df_prepared.iloc[start_idx:end_idx]
        
        feats = extract_window_features(w_df)
        feature_rows.append(feats)
        
        # Center timestamp
        if 'timestamp_ms' in w_df.columns:
            t_center = (w_df['timestamp_ms'].iloc[0] + w_df['timestamp_ms'].iloc[-1]) / 2000.0
        else:
            t_center = (start_idx + end_idx) / (2.0 * sampling_hz)
        window_timestamps.append(t_center)
        
        # Window label: 1 if >= 30% of window contains an accident event
        if 'label' in w_df.columns:
            w_label = 1 if (w_df['label'] == 1).mean() >= 0.3 else 0
        else:
            w_label = 0
        labels.append(w_label)
        
    features_df = pd.DataFrame(feature_rows)
    return features_df, window_timestamps, labels
