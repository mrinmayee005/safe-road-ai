"""
Safe Road AI - Synchronized Multimodal Inference Engine
Executes end-to-end synchronized inference across video and sensor modalities,
computes real-time fused decisions, applies temporal confirmation, and renders
an annotated video with HUD telemetry overlay.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import cv2
import torch

from src.config import DEFAULT_GPS, RESULTS_DIR
from src.video.models import VideoPipeline
from src.sensor.models import SensorClassifier
from src.sensor.features import compute_magnitudes, extract_window_features
from src.fusion.weighted_fusion import WeightedProbabilityFusion
from src.fusion.temporal_decision import TemporalDecisionEngine


class MultimodalInferenceEngine:
    """
    Coordinates synchronized video frame and sensor stream inference.
    """
    def __init__(
        self,
        video_model_arch: str = "mobilenet_v3_small",
        sensor_model_type: str = "random_forest",
        alpha: float = 0.55,
        threshold: float = 0.50,
        temporal_window: int = 3,
        temporal_persistence: int = 2
    ):
        self.video_pipeline = VideoPipeline(architecture=video_model_arch)
        self.sensor_classifier = SensorClassifier(model_type=sensor_model_type)
        self.fusion = WeightedProbabilityFusion(alpha=alpha, threshold=threshold)
        self.temporal_engine = TemporalDecisionEngine(
            smoothing_window_size=temporal_window,
            persistence_required=temporal_persistence,
            threshold=threshold
        )

        # Attempt to load trained checkpoints if available
        try:
            self.video_pipeline.load_checkpoint()
        except Exception as e:
            print(f"[Warning] Could not load video checkpoint: {e}. Using initial weights.")

        try:
            self.sensor_classifier.load_model()
        except Exception as e:
            print(f"[Warning] Could not load sensor checkpoint: {e}. Train sensor model first.")

    def run_synchronized_inference(
        self,
        video_path: Path or str,
        sensor_csv_path: Path or str,
        render_annotated_video: bool = False,
        output_video_path: Optional[Path or str] = None
    ) -> Dict[str, Any]:
        """
        Runs frame-by-frame and window-synchronized multimodal inference.
        Returns detailed timeline dictionary and metadata.
        """
        video_path_str = str(video_path).replace("\\", "/")
        sensor_csv_str = str(sensor_csv_path).replace("\\", "/")

        video_path = Path(video_path_str)
        sensor_csv_path = Path(sensor_csv_str)

        # Robust Cross-Platform Resolution for Linux (Streamlit Cloud) & Windows
        if not video_path.exists():
            if (PROJECT_ROOT / video_path_str).exists():
                video_path = PROJECT_ROOT / video_path_str
            elif (PROJECT_ROOT / "data" / video_path_str).exists():
                video_path = PROJECT_ROOT / "data" / video_path_str
            elif (PROJECT_ROOT / video_path.name).exists():
                video_path = PROJECT_ROOT / video_path.name
            else:
                raise FileNotFoundError(f"Video file not found: {video_path}")

        if not sensor_csv_path.exists():
            if (PROJECT_ROOT / sensor_csv_str).exists():
                sensor_csv_path = PROJECT_ROOT / sensor_csv_str
            elif (PROJECT_ROOT / "data" / sensor_csv_str).exists():
                sensor_csv_path = PROJECT_ROOT / "data" / sensor_csv_str
            elif (PROJECT_ROOT / sensor_csv_path.name).exists():
                sensor_csv_path = PROJECT_ROOT / sensor_csv_path.name
            else:
                raise FileNotFoundError(f"Sensor file not found: {sensor_csv_path}")

        # 1. Load sensor data and compute continuous magnitudes
        sensor_df = pd.read_csv(sensor_csv_path)
        sensor_df = compute_magnitudes(sensor_df)
        has_labels = 'label' in sensor_df.columns

        # 2. Open video stream
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 480
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if render_annotated_video:
            out_path = Path(output_video_path) if output_video_path else RESULTS_DIR / f"annotated_{video_path.stem}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            except Exception as e:
                print(f"[Warning] Failed to initialize VideoWriter on this platform: {e}")
                writer = None

        timeline = []
        self.temporal_engine.reset()

        frame_idx = 0
        accident_detected = False
        first_alert_time = None
        
        # Sensor window parameters: 1.0 second window at 50Hz = 50 samples
        imu_hz = 50
        window_samples = imu_hz

        t_start_proc = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            timestamp_ms = int(current_time * 1000)

            # Sample video every few frames to simulate lightweight real-time compute
            is_sample_frame = (frame_idx % max(1, int(round(fps / 5))) == 0)

            if is_sample_frame:
                # 3. Video inference on sampled frame
                pv_arr = self.video_pipeline.predict_frames([frame])
                pv = float(pv_arr[0]) if len(pv_arr) > 0 else 0.0

                # 4. Sensor window inference synchronized with current_time
                # Window ends at current_time, spans past 1.0 second
                curr_sensor_idx = int(current_time * imu_hz)
                start_sensor_idx = max(0, curr_sensor_idx - window_samples)
                
                if curr_sensor_idx < len(sensor_df) and (curr_sensor_idx - start_sensor_idx) >= 15:
                    w_df = sensor_df.iloc[start_sensor_idx:curr_sensor_idx+1]
                    feats = extract_window_features(w_df)
                    feats_df = pd.DataFrame([feats])
                    _, ps_arr = self.sensor_classifier.predict(feats_df)
                    ps = float(ps_arr[0])
                    cur_acc_mag = float(w_df['acc_mag'].iloc[-1])
                    cur_gyro_mag = float(w_df['gyro_mag'].iloc[-1])
                else:
                    ps = 0.0
                    cur_acc_mag = 9.81
                    cur_gyro_mag = 0.0

                # 5. Multimodal Fusion
                _, p_final_val = self.fusion.predict(pv, ps)
                p_final = float(p_final_val)

                # 6. Temporal Confirmation Rule
                is_alert, smoothed_p = self.temporal_engine.update(p_final, current_time)

                if is_alert and not accident_detected:
                    accident_detected = True
                    first_alert_time = current_time

                # Current ground-truth label (if available)
                if has_labels and curr_sensor_idx < len(sensor_df):
                    true_label = int(sensor_df['label'].iloc[curr_sensor_idx])
                else:
                    true_label = 0

                timeline.append({
                    "timestamp_sec": round(current_time, 2),
                    "pv": round(pv, 3),
                    "ps": round(ps, 3),
                    "p_final": round(p_final, 3),
                    "smoothed_p": round(smoothed_p, 3),
                    "is_alert": bool(is_alert),
                    "true_label": true_label,
                    "acc_mag": round(cur_acc_mag, 2),
                    "gyro_mag": round(cur_gyro_mag, 2)
                })

            else:
                # Use latest prediction values for non-sampled frames
                latest = timeline[-1] if timeline else {
                    "pv": 0.0, "ps": 0.0, "p_final": 0.0, "smoothed_p": 0.0,
                    "is_alert": False, "acc_mag": 9.81, "gyro_mag": 0.0
                }
                pv = latest["pv"]
                ps = latest["ps"]
                p_final = latest["p_final"]
                smoothed_p = latest["smoothed_p"]
                is_alert = latest["is_alert"]
                cur_acc_mag = latest["acc_mag"]
                cur_gyro_mag = latest["gyro_mag"]

            # 7. Render HUD Overlay if video output requested
            if writer is not None:
                hud_frame = frame.copy()
                
                # Top Banner: Normal or Accident Confirmed
                if accident_detected and (current_time >= first_alert_time):
                    # Red Alert HUD
                    cv2.rectangle(hud_frame, (0, 0), (width, 50), (0, 0, 180), -1)
                    cv2.putText(
                        hud_frame,
                        f"CRASH DETECTED! (P: {smoothed_p:.2f}) - DISPATCHING GPS ALERT",
                        (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    # Green Active Monitoring HUD
                    cv2.rectangle(hud_frame, (0, 0), (width, 40), (20, 110, 30), -1)
                    cv2.putText(
                        hud_frame,
                        f"SAFE ROAD AI | MONITORING ACTIVE | T: {current_time:.1f}s",
                        (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # Telemetry Panel Bottom-Left
                panel_w, panel_h = 240, 105
                y_p = height - panel_h - 10
                overlay = hud_frame.copy()
                cv2.rectangle(overlay, (10, y_p), (10 + panel_w, y_p + panel_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, hud_frame, 0.25, 0, hud_frame)

                cv2.putText(hud_frame, f"Video P(v): {pv:.2f}", (20, y_p + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)
                cv2.putText(hud_frame, f"Sensor P(s): {ps:.2f}", (20, y_p + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
                cv2.putText(hud_frame, f"Fused P(final): {p_final:.2f}", (20, y_p + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 150), 1)
                cv2.putText(hud_frame, f"Motion: {cur_acc_mag:.1f} m/s^2 | {cur_gyro_mag:.1f} rad/s", (20, y_p + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
                
                # Mock GPS Stamp
                cv2.putText(
                    hud_frame,
                    f"GPS: {DEFAULT_GPS['latitude']:.4f}, {DEFAULT_GPS['longitude']:.4f}",
                    (width - 210, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (200, 200, 200),
                    1
                )

                if writer is not None and writer.isOpened():
                    writer.write(hud_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        total_elapsed = time.perf_counter() - t_start_proc
        fps_proc = frame_idx / total_elapsed if total_elapsed > 0 else 0

        result = {
            "video_path": str(video_path),
            "sensor_path": str(sensor_csv_path),
            "accident_detected": bool(accident_detected),
            "first_alert_time_sec": round(first_alert_time, 2) if first_alert_time is not None else None,
            "total_duration_sec": round(frame_idx / fps, 2),
            "processed_fps": round(fps_proc, 1),
            "timeline": timeline,
            "annotated_video_path": str(out_path) if render_annotated_video else None,
            "gps_coordinates": DEFAULT_GPS
        }
        return result
