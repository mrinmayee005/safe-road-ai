"""
Safe Road AI - Synthetic & Benchmark Paired Dataset Generator
Generates realistic paired driving videos (MP4) and synchronized 6-axis IMU time-series (CSV)
covering both normal driving (smooth, turns, speed bumps, hard braking) and accident events
(frontal collision, rear-end impact, T-bone collision, vehicle spin).
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT, DATA_DIR, RAW_VIDEOS_DIR, RAW_SENSORS_DIR, SENSOR_CONFIG


def generate_synthetic_video_and_sensor(
    scenario_type: str,
    output_video_path: Path,
    output_sensor_path: Path,
    duration_sec: float = 6.0,
    fps: int = 30,
    imu_hz: int = 50,
    seed: int = 42
):
    """
    Renders a synchronized video clip (.mp4) and motion sensor time-series (.csv).
    
    scenario_type:
      - 'normal_smooth': highway driving with gentle road motion
      - 'normal_turn': city driving with lane changes and 90-deg turn
      - 'normal_speedbump': sudden vertical bump (false alarm test case)
      - 'normal_hard_brake': sudden deceleration without impact (false alarm test case)
      - 'accident_frontal': vehicle looming ahead followed by high-g frontal crash
      - 'accident_rearend': sudden high-energy impact with preceding car
      - 'accident_tbone_spin': lateral collision shock with high angular velocity spin
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_video_frames = int(duration_sec * fps)
    total_sensor_samples = int(duration_sec * imu_hz)
    
    # -------------------------------------------------------------
    # 1. GENERATE SYNCHRONIZED SENSOR TIME-SERIES (50 Hz)
    # -------------------------------------------------------------
    # Standard Earth gravity on Z-axis mounted upright: az ~ 9.8 m/s^2
    # ax: lateral (left/right), ay: longitudinal (forward/back), az: vertical
    # gx, gy, gz: angular velocity (rad/s)
    
    time_imu = np.linspace(0, duration_sec, total_sensor_samples)
    dt_imu = duration_sec / total_sensor_samples
    
    # Baseline vibration noise
    ax = np.random.normal(0.0, 0.15, total_sensor_samples)
    ay = np.random.normal(0.0, 0.15, total_sensor_samples)
    az = np.random.normal(9.81, 0.20, total_sensor_samples)
    
    gx = np.random.normal(0.0, 0.03, total_sensor_samples)
    gy = np.random.normal(0.0, 0.03, total_sensor_samples)
    gz = np.random.normal(0.0, 0.03, total_sensor_samples)
    
    is_accident = False
    accident_start_sec = 3.5
    accident_end_sec = 4.8
    
    if scenario_type == 'normal_smooth':
        # Gentle road bumps
        ay += 0.2 * np.sin(time_imu * 0.5)
        
    elif scenario_type == 'normal_turn':
        # Lane change / curve around t = 2.0 to 4.0
        mask_turn = (time_imu >= 2.0) & (time_imu <= 4.0)
        ax[mask_turn] += 1.8 * np.sin((time_imu[mask_turn] - 2.0) * np.pi / 2.0)
        gz[mask_turn] += 0.45 * np.sin((time_imu[mask_turn] - 2.0) * np.pi / 2.0)
        
    elif scenario_type == 'normal_speedbump':
        # Vertical shock at t = 3.0s (up to ~18 m/s^2, but no crash and no high yaw)
        idx_bump = int(3.0 * imu_hz)
        bump_len = int(0.3 * imu_hz)
        if idx_bump + bump_len < total_sensor_samples:
            pulse = np.sin(np.linspace(0, np.pi, bump_len)) * 8.5
            az[idx_bump:idx_bump+bump_len] += pulse
            ay[idx_bump:idx_bump+bump_len] -= pulse * 0.3
            
    elif scenario_type == 'normal_hard_brake':
        # Sharp deceleration at t = 2.5s to 3.8s (ay goes to -5.0 m/s^2)
        mask_brake = (time_imu >= 2.5) & (time_imu <= 3.8)
        ay[mask_brake] -= 5.2 * np.sin((time_imu[mask_brake] - 2.5) * np.pi / 1.3)
        az[mask_brake] += 1.2 * np.sin((time_imu[mask_brake] - 2.5) * np.pi / 1.3)
        
    elif scenario_type in ['accident_frontal', 'accident_rearend']:
        is_accident = True
        # Severe crash impact at t = 3.5s to 4.2s
        # Deceleration spike up to 35-50 m/s^2 (3.5g - 5g), high jerk
        idx_crash = int(accident_start_sec * imu_hz)
        crash_len = int(0.7 * imu_hz)
        t_crash = np.linspace(0, np.pi * 3, crash_len)
        decay = np.exp(-np.linspace(0, 3.5, crash_len))
        
        impact_ay = -42.0 * np.sin(t_crash) * decay
        impact_az = 25.0 * np.cos(t_crash * 1.5) * decay
        impact_ax = (random.choice([-1, 1]) * 18.0) * np.sin(t_crash * 2) * decay
        
        ay[idx_crash:idx_crash+crash_len] += impact_ay
        az[idx_crash:idx_crash+crash_len] += impact_az
        ax[idx_crash:idx_crash+crash_len] += impact_ax
        
        # Gyro pitch/roll perturbation
        gx[idx_crash:idx_crash+crash_len] += 2.8 * np.sin(t_crash) * decay
        gy[idx_crash:idx_crash+crash_len] += 3.2 * np.cos(t_crash) * decay
        gz[idx_crash:idx_crash+crash_len] += 1.9 * np.sin(t_crash * 2) * decay
        
    elif scenario_type == 'accident_tbone_spin':
        is_accident = True
        idx_crash = int(accident_start_sec * imu_hz)
        crash_len = int(1.2 * imu_hz)
        t_crash = np.linspace(0, np.pi * 4, crash_len)
        decay = np.exp(-np.linspace(0, 2.5, crash_len))
        
        # Massive lateral shock + violent spin (yaw gz > 4.5 rad/s)
        ax[idx_crash:idx_crash+crash_len] += 38.0 * np.sin(t_crash) * decay
        ay[idx_crash:idx_crash+crash_len] += -22.0 * np.cos(t_crash) * decay
        az[idx_crash:idx_crash+crash_len] += 16.0 * np.sin(t_crash * 1.5) * decay
        gz[idx_crash:idx_crash+crash_len] += 5.2 * np.sin(t_crash * 0.8) * decay
        gx[idx_crash:idx_crash+crash_len] += 2.5 * np.cos(t_crash * 1.2) * decay

    # Create sensor dataframe
    labels = []
    for t in time_imu:
        if is_accident and (t >= accident_start_sec and t <= accident_end_sec):
            labels.append(1)  # Accident window
        else:
            labels.append(0)  # Normal window
            
    sensor_df = pd.DataFrame({
        'timestamp_ms': (time_imu * 1000).astype(int),
        'ax': np.round(ax, 4),
        'ay': np.round(ay, 4),
        'az': np.round(az, 4),
        'gx': np.round(gx, 4),
        'gy': np.round(gy, 4),
        'gz': np.round(gz, 4),
        'label': labels
    })
    sensor_df.to_csv(output_sensor_path, index=False)

    # -------------------------------------------------------------
    # 2. RENDER SYNCHRONIZED VIDEO (OpenCV MP4)
    # -------------------------------------------------------------
    width, height = 480, 360
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    horizon_y = int(height * 0.45)
    
    # Track preceding vehicle state
    car_distance = 100.0  # arbitrary distance units
    car_lateral = width // 2
    
    for f in range(total_video_frames):
        current_time = f / fps
        is_accident_frame = is_accident and (current_time >= accident_start_sec)
        
        # Create road frame background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky (dark gradient or day sky)
        frame[0:horizon_y, :] = (210, 180, 130)  # Light blueish sky
        # Ground / Road
        frame[horizon_y:, :] = (55, 60, 65)     # Dark asphalt
        
        # Road perspective trapezoid
        pts = np.array([
            [width * 0.38, horizon_y],
            [width * 0.62, horizon_y],
            [width * 0.95, height],
            [width * 0.05, height]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (45, 48, 52))
        
        # Lane markings (animated moving down)
        lane_offset = int((current_time * 80) % 40)
        for y_lane in range(horizon_y + 10, height, 40):
            y_pos = y_lane + lane_offset
            if y_pos < height:
                w_lane = max(2, int((y_pos - horizon_y) / 25))
                cv2.line(frame, (width // 2, y_pos), (width // 2, min(height, y_pos + 15)), (230, 230, 230), w_lane)

        # Draw preceding vehicle or crash event
        if scenario_type in ['normal_speedbump']:
            # Road speed bump line
            bump_y = int(horizon_y + ((current_time / 3.0) * (height - horizon_y)) % (height - horizon_y))
            if 2.5 <= current_time <= 3.5:
                cv2.line(frame, (int(width * 0.2), bump_y), (int(width * 0.8), bump_y), (0, 215, 255), 4)

        if scenario_type in ['accident_frontal', 'accident_rearend', 'normal_hard_brake', 'accident_tbone_spin']:
            # Vehicle in front
            if current_time < accident_start_sec:
                # Vehicle approaching or steady
                car_distance = max(15.0, 100.0 - (current_time / accident_start_sec) * 80.0)
            else:
                # Impact state
                car_distance = 12.0
                
            scale = max(0.2, min(3.5, 60.0 / car_distance))
            car_w = int(60 * scale)
            car_h = int(45 * scale)
            car_x = int(car_lateral - car_w // 2)
            car_y = int(horizon_y + (height - horizon_y) * 0.7 - car_h)
            
            if scenario_type == 'accident_tbone_spin' and current_time >= accident_start_sec:
                # Car swerves violently sideways
                car_x += int(np.sin((current_time - accident_start_sec) * 12) * 90)
                car_y += int(np.cos((current_time - accident_start_sec) * 12) * 20)

            # Draw vehicle body (red vehicle)
            cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (20, 25, 180), -1)
            cv2.rectangle(frame, (car_x + 5, car_y + 5), (car_x + car_w - 5, car_y + car_h // 2), (40, 40, 40), -1)
            # Tail lights
            tl_r = max(2, int(4 * scale))
            cv2.circle(frame, (car_x + tl_r + 2, car_y + car_h - tl_r - 2), tl_r, (0, 0, 255), -1)
            cv2.circle(frame, (car_x + car_w - tl_r - 2, car_y + car_h - tl_r - 2), tl_r, (0, 0, 255), -1)

            # Crash impact visualization on video
            if is_accident_frame and (current_time - accident_start_sec < 1.2):
                # Severe visual camera jitter / collision shock
                jitter_x = random.randint(-15, 15)
                jitter_y = random.randint(-15, 15)
                M = np.float32([[1, 0, jitter_x], [0, 1, jitter_y]])
                frame = cv2.warpAffine(frame, M, (width, height))
                
                # Impact flash / debris / smoke circle
                flash_alpha = max(0.0, 1.0 - (current_time - accident_start_sec) / 1.0)
                overlay = frame.copy()
                cv2.circle(overlay, (width // 2, car_y + car_h // 2), int(70 * scale), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.4 * flash_alpha, frame, 1.0 - 0.4 * flash_alpha, 0, frame)

        # Vehicle dashboard bottom overlay (smartphone fixed inside four-wheeler cabin)
        dash_pts = np.array([
            [0, height],
            [width, height],
            [width, int(height * 0.90)],
            [0, int(height * 0.90)]
        ], np.int32)
        cv2.fillPoly(frame, [dash_pts], (25, 25, 25))
        
        # Telemetry info bar on raw video
        cv2.putText(
            frame,
            f"SAFE ROAD AI | T: {current_time:.2f}s | Scenario: {scenario_type}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        out.write(frame)

    out.release()


def generate_paired_dataset(num_samples_per_category: int = 5, output_dir: Path = None):
    """
    Generates balanced training, validation, and testing suites.
    """
    scenarios = [
        ('normal_smooth', 'normal'),
        ('normal_turn', 'normal'),
        ('normal_speedbump', 'normal'),
        ('normal_hard_brake', 'normal'),
        ('accident_frontal', 'accident'),
        ('accident_rearend', 'accident'),
        ('accident_tbone_spin', 'accident')
    ]
    
    metadata = []
    sample_id = 1
    
    print(f"[Dataset Generator] Generating paired dataset...")
    
    for scenario_name, category in scenarios:
        for i in range(num_samples_per_category):
            seed = sample_id * 101 + i * 17
            vid_filename = f"{category}_{scenario_name}_{i+1:02d}.mp4"
            csv_filename = f"{category}_{scenario_name}_{i+1:02d}.csv"
            
            vid_path = RAW_VIDEOS_DIR / vid_filename
            csv_path = RAW_SENSORS_DIR / csv_filename
            
            generate_synthetic_video_and_sensor(
                scenario_type=scenario_name,
                output_video_path=vid_path,
                output_sensor_path=csv_path,
                duration_sec=6.0,
                fps=30,
                imu_hz=SENSOR_CONFIG["sampling_rate_hz"],
                seed=seed
            )
            
            # Balanced split: 60% train, 20% val, 20% test per scenario
            if num_samples_per_category >= 5:
                if i < int(num_samples_per_category * 0.6):
                    split = 'train'
                elif i < int(num_samples_per_category * 0.8):
                    split = 'val'
                else:
                    split = 'test'
            else:
                # If small count like 3, distribute: 0 -> train, 1 -> val, 2 -> test
                if i % 3 == 0:
                    split = 'train'
                elif i % 3 == 1:
                    split = 'val'
                else:
                    split = 'test'
                
            metadata.append({
                'sample_id': sample_id,
                'scenario': scenario_name,
                'category': category,
                'label': 1 if category == 'accident' else 0,
                'video_path': str(vid_path.relative_to(PROJECT_ROOT if output_dir is None else output_dir)),
                'sensor_path': str(csv_path.relative_to(PROJECT_ROOT if output_dir is None else output_dir)),
                'split': split
            })
            sample_id += 1
            
    df_meta = pd.DataFrame(metadata)
    meta_path = DATA_DIR / "dataset_metadata.csv"
    df_meta.to_csv(meta_path, index=False)
    print(f"[Dataset Generator] Complete! Created {len(df_meta)} paired samples.")
    print(f"  - Train: {(df_meta['split']=='train').sum()} samples")
    print(f"  - Val:   {(df_meta['split']=='val').sum()} samples")
    print(f"  - Test:  {(df_meta['split']=='test').sum()} samples")
    print(f"  - Saved manifest: {meta_path}")
    return df_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate paired multimodal dataset for Safe Road AI Phase 1")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per scenario category (default: 5)")
    args = parser.parse_args()
    
    generate_paired_dataset(num_samples_per_category=args.samples)
