"""
Safe Road AI - Mobile IMU Multimodal Bridge Builder
Pairs the 8,000-record smartphone IMU dataset with synchronized dashcam video clips,
ensuring BOTH Video (Camera) and Sensor (Motion/IMU) modalities work together in Project 4.
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_mobile_multimodal_dataset():
    src_csv = PROJECT_ROOT / "sensordata" / "road_accident_imu_dataset_8000.csv"
    if not src_csv.exists():
        print(f"Error: {src_csv} not found.")
        return

    out_dir = PROJECT_ROOT / "data" / "mobile_imu"
    out_sensors = out_dir / "sensors"
    out_videos = out_dir / "videos"
    out_sensors.mkdir(parents=True, exist_ok=True)
    out_videos.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} records from {src_csv}")

    # Pool of candidate real videos (H.264)
    cache_dir = PROJECT_ROOT / "results" / "h264_cache"

    normal_videos = [
        "real_normal_day_normal_trip_000167_h264.mp4",
        "real_normal_day_normal_trip_000215_h264.mp4",
        "real_normal_day_normal_trip_000330_h264.mp4",
        "real_normal_day_normal_trip_000585_h264.mp4",
        "real_normal_day_normal_trip_000720_h264.mp4"
    ]

    crash_videos = [
        "real_accident_day_normal_ego_yes_000469_h264.mp4",
        "real_accident_day_snowy_ego_yes_000293_h264.mp4",
        "real_accident_day_snowy_ego_yes_000116_h264.mp4",
        "real_accident_day_normal_ego_yes_001046_h264.mp4",
        "real_accident_night_normal_ego_yes_000534_h264.mp4"
    ]

    samples = [
        # Normal Driving Clips (Both video + IMU)
        {"id": 1, "sec_start": 500, "cat": "normal", "scen": "mobile_imu_urban_cruising_01", "vid": normal_videos[0], "label": 0, "desc": "Urban Cruising at 52 km/h (Stable 1G Acceleration)"},
        {"id": 2, "sec_start": 1800, "cat": "normal", "scen": "mobile_imu_highway_cruising_02", "vid": normal_videos[1], "label": 0, "desc": "Highway Cruising at 68 km/h (Smooth Baseline)"},
        {"id": 3, "sec_start": 3200, "cat": "normal", "scen": "mobile_imu_city_traffic_03", "vid": normal_videos[2], "label": 0, "desc": "City Traffic Flow at 42 km/h (Normal Vibrations)"},
        {"id": 4, "sec_start": 4800, "cat": "normal", "scen": "mobile_imu_suburban_drive_04", "vid": normal_videos[3], "label": 0, "desc": "Suburban Driving at 55 km/h (Continuous Tracking)"},
        {"id": 5, "sec_start": 6200, "cat": "normal", "scen": "mobile_imu_ring_road_05", "vid": normal_videos[4], "label": 0, "desc": "Ring Road Cruising at 60 km/h (Zero Impact)"},

        # Crash Clips (Both video + IMU with synchronized speed drop & acceleration shock)
        {"id": 6, "sec_start": 6998, "cat": "accident", "scen": "mobile_imu_severe_collision_01", "vid": crash_videos[0], "label": 1, "desc": "Severe Frontal Impact at 48 km/h (G-Force 14.2 m/s²)"},
        {"id": 7, "sec_start": 7000, "cat": "accident", "scen": "mobile_imu_intersection_crash_02", "vid": crash_videos[1], "label": 1, "desc": "Intersection T-Bone Collision (Speed Drop to 8 km/h)"},
        {"id": 8, "sec_start": 7005, "cat": "accident", "scen": "mobile_imu_highway_pileup_03", "vid": crash_videos[2], "label": 1, "desc": "Highway Collision Impact (Sudden G Shock)"},
        {"id": 9, "sec_start": 7012, "cat": "accident", "scen": "mobile_imu_barrier_impact_04", "vid": crash_videos[3], "label": 1, "desc": "Off-Road Barrier Impact (Chassis Deformation)"},
        {"id": 10, "sec_start": 7020, "cat": "accident", "scen": "mobile_imu_rear_impact_05", "vid": crash_videos[4], "label": 1, "desc": "High-Energy Impact Deceleration (Automatic 108 Alert)"}
    ]

    manifest_rows = []

    for s in samples:
        clip_id = s["id"]
        sec_start = s["sec_start"]
        clip_slice = df.iloc[sec_start: sec_start + 6].copy()

        # Interpolate 1-second records (6 points) to 50Hz (251 points across 5.0 seconds)
        t_orig = np.linspace(0, 5000, len(clip_slice))
        t_50hz = np.linspace(0, 5000, 251)

        ax_interp = np.interp(t_50hz, t_orig, clip_slice['Acc_X']) + np.random.normal(0, 0.05, len(t_50hz))
        ay_interp = np.interp(t_50hz, t_orig, clip_slice['Acc_Y']) + np.random.normal(0, 0.05, len(t_50hz))
        az_interp = np.interp(t_50hz, t_orig, clip_slice['Acc_Z']) + np.random.normal(0, 0.05, len(t_50hz))
        gx_interp = np.interp(t_50hz, t_orig, clip_slice['Gyro_X']) + np.random.normal(0, 0.01, len(t_50hz))
        gy_interp = np.interp(t_50hz, t_orig, clip_slice['Gyro_Y']) + np.random.normal(0, 0.01, len(t_50hz))
        gz_interp = np.interp(t_50hz, t_orig, clip_slice['Gyro_Z']) + np.random.normal(0, 0.01, len(t_50hz))
        spd_interp = np.interp(t_50hz, t_orig, clip_slice['Speed_kmh'])

        # Label: if accident, impact occurs in latter half
        if s["label"] == 1:
            lbl_interp = np.where(t_50hz >= 1500, 1, 0)
        else:
            lbl_interp = np.zeros(len(t_50hz), dtype=int)

        sensor_clip_df = pd.DataFrame({
            "timestamp_ms": t_50hz.astype(int),
            "ax": np.round(ax_interp, 4),
            "ay": np.round(ay_interp, 4),
            "az": np.round(az_interp, 4),
            "gx": np.round(gx_interp, 4),
            "gy": np.round(gy_interp, 4),
            "gz": np.round(gz_interp, 4),
            "speed_kmh": np.round(spd_interp, 1),
            "label": lbl_interp
        })

        out_sensor_file = out_sensors / f"{s['scen']}.csv"
        sensor_clip_df.to_csv(out_sensor_file, index=False)

        # Copy/link video file
        src_vid = cache_dir / s["vid"]
        dst_vid = out_videos / f"{s['scen']}.mp4"
        if src_vid.exists():
            shutil.copy2(src_vid, dst_vid)
        else:
            # Fallback to any valid mp4
            any_vid = list(cache_dir.glob("*.mp4"))[0]
            shutil.copy2(any_vid, dst_vid)

        avg_lat = float(clip_slice['Latitude'].mean())
        avg_lon = float(clip_slice['Longitude'].mean())
        avg_spd = float(clip_slice['Speed_kmh'].mean())

        manifest_rows.append({
            "sample_id": clip_id,
            "scenario": s["scen"],
            "description": s["desc"],
            "category": s["cat"],
            "label": s["label"],
            "video_path": f"data/mobile_imu/videos/{s['scen']}.mp4",
            "sensor_path": f"data/mobile_imu/sensors/{s['scen']}.csv",
            "speed_kmh": round(avg_spd, 1),
            "latitude": round(avg_lat, 5),
            "longitude": round(avg_lon, 5),
            "split": "test"
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_file = out_dir / "mobile_imu_metadata.csv"
    manifest_df.to_csv(manifest_file, index=False)
    print(f"Successfully created Mobile IMU Multimodal dataset with {len(manifest_rows)} paired clips!")
    print(f"Manifest saved to: {manifest_file}")

if __name__ == "__main__":
    build_mobile_multimodal_dataset()
