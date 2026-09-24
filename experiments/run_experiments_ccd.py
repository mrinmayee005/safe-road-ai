"""
Safe Road AI - Phase 1 CCD Real-World Dashcam Experiments Runner
Evaluates:
  - Video models (MobileNetV3-Small vs ResNet18) on real dashcam frames (CCD)
  - Sensor models (Random Forest, Gradient Boosting, Extra Trees) on kinematic IMU
  - Multimodal Fusion (E3) and Temporal Confirmation (E4)
  - Produces:
      * results/metrics_summary_ccd.json
      * results/confusion_matrices_ccd.png
      * results/experiment_comparison_chart_ccd.png
      * results/roc_curves_ccd.png
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure project root in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, CCD_DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, VIDEO_CONFIG, SENSOR_CONFIG, FUSION_CONFIG
from src.video.preprocessor import VideoPreprocessor
from src.video.models import VideoPipeline
from src.sensor.features import process_sensor_timeseries
from src.sensor.models import SensorClassifier
from src.fusion.weighted_fusion import WeightedProbabilityFusion, StackingMetaFusion
from src.fusion.temporal_decision import TemporalDecisionEngine
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_confusion_matrices, plot_experiment_comparisons, plot_roc_curves


def extract_frame_samples_from_split(
    df_meta_split: pd.DataFrame,
    preprocessor: VideoPreprocessor,
    samples_per_clip: int = 8
) -> Tuple[List[torch.Tensor], List[int]]:
    frame_tensors = []
    frame_labels = []

    for _, row in df_meta_split.iterrows():
        vid_path = PROJECT_ROOT / row['video_path']
        if not vid_path.exists():
            continue
        frames, _ = preprocessor.sample_frames_from_video(vid_path)
        if not frames:
            continue
        
        step = max(1, len(frames) // samples_per_clip)
        selected_frames = frames[::step][:samples_per_clip]
        
        tensors = [preprocessor.preprocess_frame(f) for f in selected_frames]
        labels = [row['label']] * len(tensors)
        
        frame_tensors.extend(tensors)
        frame_labels.extend(labels)

    return frame_tensors, frame_labels


def extract_sensor_features_from_split(df_meta_split: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    all_feature_dfs = []
    all_labels = []

    for _, row in df_meta_split.iterrows():
        csv_path = PROJECT_ROOT / row['sensor_path']
        if not csv_path.exists():
            continue
        df_sensor = pd.read_csv(csv_path)
        feat_df, _, labels = process_sensor_timeseries(df_sensor)
        if len(feat_df) > 0:
            all_feature_dfs.append(feat_df)
            all_labels.extend(labels)

    if not all_feature_dfs:
        return pd.DataFrame(), np.array([])

    combined_features = pd.concat(all_feature_dfs, ignore_index=True)
    return combined_features, np.array(all_labels)


def run_ccd_benchmark():
    print("=" * 75)
    print("      SAFE ROAD AI — USER REAL DASHCAM BENCHMARK (CCD 75K FRAMES)")
    print("=" * 75)

    meta_path = CCD_DATA_DIR / "ccd_metadata.csv"
    if not meta_path.exists():
        print("[Notice] ccd_metadata.csv not found! Building CCD dataset first...")
        from data.ccd_dataset_builder import build_ccd_dataset
        build_ccd_dataset(num_samples=100)

    df_meta = pd.read_csv(meta_path)
    train_meta = df_meta[df_meta['split'] == 'train']
    val_meta = df_meta[df_meta['split'] == 'val']
    test_meta = df_meta[df_meta['split'] == 'test']

    print(f"Loaded CCD Real Manifest: {len(df_meta)} samples total.")
    print(f"  - Train: {len(train_meta)} clips | Val: {len(val_meta)} clips | Test: {len(test_meta)} clips")

    # =========================================================================
    # PART 1: SENSOR BRANCH TRAINING
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> SENSOR BRANCH: Feature Extraction & Model Training (CCD)")
    print("-" * 60)
    
    X_train_s, y_train_s = extract_sensor_features_from_split(train_meta)
    X_val_s, y_val_s = extract_sensor_features_from_split(val_meta)
    X_test_s, y_test_s = extract_sensor_features_from_split(test_meta)

    sensor_models = {}
    sensor_test_results = {}

    for s_name in SENSOR_CONFIG["models"]:
        clf = SensorClassifier(model_type=s_name, random_state=42)
        clf.train(X_train_s, y_train_s, X_val_s, y_val_s)
        test_preds, test_probs = clf.predict(X_test_s)
        s_metrics = compute_metrics(y_test_s, test_preds, test_probs)
        sensor_models[s_name] = clf
        sensor_test_results[s_name] = s_metrics
        print(f"  [{s_name.upper()}] Acc: {s_metrics['accuracy']*100:.1f}% | Prec: {s_metrics['precision']*100:.1f}% | Rec: {s_metrics['recall']*100:.1f}% | F1: {s_metrics['f1']:.3f}")

    primary_sensor_clf = sensor_models[SENSOR_CONFIG["primary_model"]]

    # =========================================================================
    # PART 2: VIDEO BRANCH TRAINING ON REAL DASHCAM FRAMES
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> VIDEO BRANCH: Frame Preprocessing & Vision Training (CCD Real Frames)")
    print("-" * 60)
    
    preprocessor = VideoPreprocessor()
    train_frames, train_frame_labels = extract_frame_samples_from_split(train_meta, preprocessor, samples_per_clip=8)
    val_frames, val_frame_labels = extract_frame_samples_from_split(val_meta, preprocessor, samples_per_clip=8)
    test_frames, test_frame_labels = extract_frame_samples_from_split(test_meta, preprocessor, samples_per_clip=8)

    video_pipelines = {}
    video_test_results = {}

    for v_arch in ["mobilenet_v3_small", "resnet18"]:
        pipe = VideoPipeline(architecture=v_arch)
        pipe.train_model(
            train_frames, train_frame_labels,
            val_frames, val_frame_labels,
            epochs=3,
            batch_size=16,
            lr=1e-4
        )
        
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.stack(test_frames), torch.tensor(test_frame_labels)),
            batch_size=16, shuffle=False
        )
        pipe.model.eval()
        test_preds_v = []
        test_probs_v = []
        with torch.no_grad():
            for b_frames, _ in test_loader:
                b_frames = b_frames.to(pipe.device)
                logits = pipe.model(b_frames)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_probs_v.extend(probs)
                test_preds_v.extend((probs >= 0.5).astype(int))

        v_metrics = compute_metrics(test_frame_labels, test_preds_v, test_probs_v)
        video_pipelines[v_arch] = pipe
        video_test_results[v_arch] = v_metrics
        print(f"  [{v_arch.upper()}] Acc: {v_metrics['accuracy']*100:.1f}% | Prec: {v_metrics['precision']*100:.1f}% | Rec: {v_metrics['recall']*100:.1f}% | F1: {v_metrics['f1']:.3f}")

    primary_video_pipe = video_pipelines[VIDEO_CONFIG["primary_model"]]

    # =========================================================================
    # PART 3: MULTIMODAL FUSION & EVALUATION
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> MULTIMODAL FUSION & TEMPORAL EVALUATION (CCD)")
    print("-" * 60)

    def evaluate_paired_clips(meta_split: pd.DataFrame, v_pipe: VideoPipeline, s_clf: SensorClassifier):
        clip_pv_list = []
        clip_ps_list = []
        clip_labels = []

        for _, row in meta_split.iterrows():
            vid_path = PROJECT_ROOT / row['video_path']
            csv_path = PROJECT_ROOT / row['sensor_path']
            
            pv_arr, _, _ = v_pipe.predict_video(vid_path)
            pv_clip = float(np.max(pv_arr)) if len(pv_arr) > 0 else 0.0

            ps_arr, _, _, _ = s_clf.predict_timeseries_file(csv_path)
            ps_clip = float(np.max(ps_arr)) if len(ps_arr) > 0 else 0.0

            clip_pv_list.append(pv_clip)
            clip_ps_list.append(ps_clip)
            clip_labels.append(row['label'])

        return np.array(clip_pv_list), np.array(clip_ps_list), np.array(clip_labels)

    val_pv, val_ps, val_y = evaluate_paired_clips(val_meta, primary_video_pipe, primary_sensor_clf)
    test_pv, test_ps, test_y = evaluate_paired_clips(test_meta, primary_video_pipe, primary_sensor_clf)

    fusion_engine = WeightedProbabilityFusion()
    fusion_engine.optimize_parameters(val_pv, val_ps, val_y)

    stacking_engine = StackingMetaFusion()
    stacking_engine.fit(val_pv, val_ps, val_y)

    # E1: Video-only baseline
    e1_preds = (test_pv >= 0.50).astype(int)
    e1_metrics = compute_metrics(test_y, e1_preds, test_pv)

    # E2: Sensor-only baseline
    e2_preds = (test_ps >= 0.50).astype(int)
    e2_metrics = compute_metrics(test_y, e2_preds, test_ps)

    # E3: Multimodal Fusion
    e3_preds, e3_probs = fusion_engine.predict(test_pv, test_ps)
    e3_metrics = compute_metrics(test_y, e3_preds, e3_probs)

    # E4: Multimodal + Temporal Decision Rule
    temporal_engine = TemporalDecisionEngine(smoothing_window_size=3, persistence_required=2, threshold=fusion_engine.threshold)
    e4_clip_preds = []
    e4_clip_probs = []

    for _, row in test_meta.iterrows():
        vid_path = PROJECT_ROOT / row['video_path']
        csv_path = PROJECT_ROOT / row['sensor_path']
        
        pv_seq, ts_v, _ = primary_video_pipe.predict_video(vid_path)
        ps_seq, ts_s, _, _ = primary_sensor_clf.predict_timeseries_file(csv_path)
        
        min_len = min(len(pv_seq), len(ps_seq))
        if min_len == 0:
            e4_clip_preds.append(0)
            e4_clip_probs.append(0.0)
            continue
            
        pv_aligned = pv_seq[:min_len]
        ps_aligned = ps_seq[:min_len]
        ts_aligned = ts_v[:min_len]
        
        fused_seq = fusion_engine.fuse(pv_aligned, ps_aligned)
        alert_seq, smoothed_seq = temporal_engine.process_sequence(fused_seq, ts_aligned)
        
        is_clip_alert = int(np.any(alert_seq == 1))
        e4_clip_preds.append(is_clip_alert)
        e4_clip_probs.append(float(np.max(smoothed_seq)))

    e4_metrics = compute_metrics(test_y, e4_clip_preds, e4_clip_probs)

    core_experiments = {
        "E1: Video-Only": e1_metrics,
        "E2: Sensor-Only": e2_metrics,
        "E3: Video+Sensor Fusion": e3_metrics,
        "E4: Temporal Decision": e4_metrics
    }

    # =========================================================================
    # PART 4: VISUALIZATIONS & METRICS EXPORT
    # =========================================================================
    plot_confusion_matrices(core_experiments, RESULTS_DIR / "confusion_matrices_ccd.png")
    plot_experiment_comparisons(core_experiments, RESULTS_DIR / "experiment_comparison_chart_ccd.png")
    
    roc_data = {
        "E1: Video-Only (CCD Dashcam)": (test_y, test_pv),
        "E2: Sensor-Only (Kinematic)": (test_y, test_ps),
        "E3: Multimodal Fusion": (test_y, e3_probs),
        "E4: Temporal Confirmation": (test_y, np.array(e4_clip_probs))
    }
    plot_roc_curves(roc_data, RESULTS_DIR / "roc_curves_ccd.png")

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "Car Crash Dataset (CCD - User Archive)",
        "dataset_type": "Real-World Dashcam Video & Kinematic Telemetry",
        "sample_count": len(df_meta),
        "core_experiments": core_experiments,
        "video_models_comparison": video_test_results,
        "sensor_models_comparison": sensor_test_results,
        "optimal_alpha": fusion_engine.alpha,
        "optimal_threshold": fusion_engine.threshold
    }

    out_json = RESULTS_DIR / "metrics_summary_ccd.json"
    with open(out_json, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print("             USER REAL DASHCAM (CCD) EXPERIMENT EVALUATION TABLE")
    print("=" * 80)
    header = f"| {'Experiment':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FAR (FP Rate)':<12} |"
    sep = "|" + "-"*27 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|"
    print(header)
    print(sep)
    for exp_k, m in core_experiments.items():
        row_str = f"| {exp_k:<25} | {m['accuracy']*100:>8.1f}% | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1']:>10.3f} | {m['false_alarm_rate']*100:>10.1f}% |"
        print(row_str)
    print(sep)
    print(f"\n[CCD Benchmark Complete] Saved results to: {out_json}")
    return summary_report


if __name__ == '__main__':
    run_ccd_benchmark()
