"""
Safe Road AI - Phase 1 Experiments Runner
Automates training and comprehensive evaluation of:
  - Video models (MobileNetV3-Small vs ResNet18)
  - Sensor models (Random Forest vs Gradient Boosting vs Extra Trees)
  - Fusion strategies (Weighted Probability Fusion vs Stacking Meta-Classifier)
  - Temporal confirmation rule
  - The 4 core benchmark experiments (E1, E2, E3, E4)
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

from src.config import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, VIDEO_CONFIG, SENSOR_CONFIG, FUSION_CONFIG
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
    samples_per_clip: int = 10
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Extracts balanced sampled frames and frame labels from video clips.
    """
    frame_tensors = []
    frame_labels = []

    for _, row in df_meta_split.iterrows():
        vid_path = PROJECT_ROOT / row['video_path']
        if not vid_path.exists():
            continue
        frames, _ = preprocessor.sample_frames_from_video(vid_path)
        if not frames:
            continue
        
        # Subsample up to `samples_per_clip` frames per video
        step = max(1, len(frames) // samples_per_clip)
        selected_frames = frames[::step][:samples_per_clip]
        
        tensors = [preprocessor.preprocess_frame(f) for f in selected_frames]
        # Frame label matches clip category
        labels = [row['label']] * len(tensors)
        
        frame_tensors.extend(tensors)
        frame_labels.extend(labels)

    return frame_tensors, frame_labels


def extract_sensor_features_from_split(df_meta_split: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts windowed tabular features from all sensor CSVs in a split.
    """
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


def run_phase1_benchmark():
    print("=" * 75)
    print("      SAFE ROAD AI — PHASE 1 RESEARCH & BENCHMARK SUITE")
    print("=" * 75)

    meta_path = DATA_DIR / "dataset_metadata.csv"
    if not meta_path.exists():
        print("[Error] dataset_metadata.csv not found! Generating dataset first...")
        from data.dataset_generator import generate_paired_dataset
        generate_paired_dataset(num_samples_per_category=6)

    df_meta = pd.read_csv(meta_path)
    train_meta = df_meta[df_meta['split'] == 'train']
    val_meta = df_meta[df_meta['split'] == 'val']
    test_meta = df_meta[df_meta['split'] == 'test']

    print(f"Loaded Dataset Manifest: {len(df_meta)} samples total.")
    print(f"  - Train: {len(train_meta)} clips | Val: {len(val_meta)} clips | Test: {len(test_meta)} clips")

    # =========================================================================
    # PART 1: SENSOR BRANCH TRAINING & MULTI-MODEL COMPARISON
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> SENSOR BRANCH: Feature Extraction & Model Comparison")
    print("-" * 60)
    
    X_train_s, y_train_s = extract_sensor_features_from_split(train_meta)
    X_val_s, y_val_s = extract_sensor_features_from_split(val_meta)
    X_test_s, y_test_s = extract_sensor_features_from_split(test_meta)

    print(f"Sensor Feature Matrix: {X_train_s.shape[0]} train windows, {X_test_s.shape[0]} test windows.")
    print(f"Features Extracted ({X_train_s.shape[1]}): {list(X_train_s.columns[:5])} ...")

    sensor_models = {}
    sensor_test_results = {}

    for s_name in SENSOR_CONFIG["models"]:
        print(f"\nTraining Sensor Model: {s_name.upper()}...")
        clf = SensorClassifier(model_type=s_name, random_state=42)
        clf.train(X_train_s, y_train_s, X_val_s, y_val_s)
        
        test_preds, test_probs = clf.predict(X_test_s)
        s_metrics = compute_metrics(y_test_s, test_preds, test_probs)
        sensor_models[s_name] = clf
        sensor_test_results[s_name] = s_metrics
        print(f"  [{s_name}] Test Acc: {s_metrics['accuracy']*100:.1f}% | Precision: {s_metrics['precision']*100:.1f}% | Recall: {s_metrics['recall']*100:.1f}% | F1: {s_metrics['f1']:.3f}")

    primary_sensor_clf = sensor_models[SENSOR_CONFIG["primary_model"]]

    # =========================================================================
    # PART 2: VIDEO BRANCH TRAINING & MULTI-MODEL COMPARISON
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> VIDEO BRANCH: Frame Preprocessing & Model Comparison")
    print("-" * 60)
    
    preprocessor = VideoPreprocessor()
    train_frames, train_frame_labels = extract_frame_samples_from_split(train_meta, preprocessor, samples_per_clip=8)
    val_frames, val_frame_labels = extract_frame_samples_from_split(val_meta, preprocessor, samples_per_clip=8)
    test_frames, test_frame_labels = extract_frame_samples_from_split(test_meta, preprocessor, samples_per_clip=8)

    print(f"Extracted Video Frames: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test.")

    video_pipelines = {}
    video_test_results = {}

    for v_arch in VIDEO_CONFIG["models"]:
        print(f"\nTraining Vision Model: {v_arch.upper()}...")
        pipe = VideoPipeline(architecture=v_arch)
        pipe.train_model(
            train_frames, train_frame_labels,
            val_frames, val_frame_labels,
            epochs=4,
            batch_size=16,
            lr=1e-4
        )
        # Evaluate on test frames
        test_preds_v = []
        test_probs_v = []
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.stack(test_frames), torch.tensor(test_frame_labels)),
            batch_size=16, shuffle=False
        )
        pipe.model.eval()
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
        print(f"  [{v_arch}] Test Acc: {v_metrics['accuracy']*100:.1f}% | Precision: {v_metrics['precision']*100:.1f}% | Recall: {v_metrics['recall']*100:.1f}% | F1: {v_metrics['f1']:.3f}")

    primary_video_pipe = video_pipelines[VIDEO_CONFIG["primary_model"]]

    # =========================================================================
    # PART 3: MULTIMODAL SYNCHRONIZATION & FUSION TUNING (Validation Split)
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> MULTIMODAL FUSION: Parameter Optimization (Alpha Search)")
    print("-" * 60)

    # Collect paired test clip predictions
    def evaluate_paired_clips(meta_split: pd.DataFrame, v_pipe: VideoPipeline, s_clf: SensorClassifier):
        clip_pv_list = []
        clip_ps_list = []
        clip_labels = []
        clip_scenarios = []

        for _, row in meta_split.iterrows():
            vid_path = PROJECT_ROOT / row['video_path']
            csv_path = PROJECT_ROOT / row['sensor_path']
            
            # Predict video
            pv_arr, _, _ = v_pipe.predict_video(vid_path)
            pv_clip = float(np.max(pv_arr)) if len(pv_arr) > 0 else 0.0

            # Predict sensor
            ps_arr, _, _, _ = s_clf.predict_timeseries_file(csv_path)
            ps_clip = float(np.max(ps_arr)) if len(ps_arr) > 0 else 0.0

            clip_pv_list.append(pv_clip)
            clip_ps_list.append(ps_clip)
            clip_labels.append(row['label'])
            clip_scenarios.append(row['scenario'])

        return np.array(clip_pv_list), np.array(clip_ps_list), np.array(clip_labels), clip_scenarios

    val_pv, val_ps, val_y, _ = evaluate_paired_clips(val_meta, primary_video_pipe, primary_sensor_clf)
    test_pv, test_ps, test_y, test_scenarios = evaluate_paired_clips(test_meta, primary_video_pipe, primary_sensor_clf)

    fusion_engine = WeightedProbabilityFusion()
    tuning_res = fusion_engine.optimize_parameters(val_pv, val_ps, val_y)

    # Train Stacking Meta-Classifier for comparison
    stacking_engine = StackingMetaFusion()
    stacking_engine.fit(val_pv, val_ps, val_y)

    # =========================================================================
    # PART 4: THE 4 BENCHMARK EXPERIMENTS (E1 to E4)
    # =========================================================================
    print("\n" + "=" * 75)
    print(">>> EVALUATING THE 4 CORE EXPERIMENTAL COMPARISONS (Section 15)")
    print("=" * 75)

    # E1: Video-only baseline
    e1_preds = (test_pv >= 0.50).astype(int)
    e1_metrics = compute_metrics(test_y, e1_preds, test_pv)

    # E2: Sensor-only baseline
    e2_preds = (test_ps >= 0.50).astype(int)
    e2_metrics = compute_metrics(test_y, e2_preds, test_ps)

    # E3: Video + Sensor (Weighted Fusion)
    e3_preds, e3_probs = fusion_engine.predict(test_pv, test_ps)
    e3_metrics = compute_metrics(test_y, e3_preds, e3_probs)

    # Stacking Fusion comparison
    stacking_preds, stacking_probs = stacking_engine.predict(test_pv, test_ps)
    stacking_metrics = compute_metrics(test_y, stacking_preds, stacking_probs)

    # E4: Multimodal + Temporal Decision Rule
    # Evaluate frame-level temporal smoothing and persistence on full test clips
    temporal_engine = TemporalDecisionEngine(smoothing_window_size=3, persistence_required=2, threshold=fusion_engine.threshold)
    
    e4_clip_preds = []
    e4_clip_probs = []
    stress_test_analysis = []

    for _, row in test_meta.iterrows():
        vid_path = PROJECT_ROOT / row['video_path']
        csv_path = PROJECT_ROOT / row['sensor_path']
        
        pv_seq, ts_v, _ = primary_video_pipe.predict_video(vid_path)
        ps_seq, ts_s, _, _ = primary_sensor_clf.predict_timeseries_file(csv_path)
        
        # Interpolate/align to common timestamps
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
        
        # If any alert confirmed in clip
        is_clip_alert = int(np.any(alert_seq == 1))
        e4_clip_preds.append(is_clip_alert)
        e4_clip_probs.append(float(np.max(smoothed_seq)))
        
        # Log false alarm resistance on edge cases
        if row['scenario'] in ['normal_speedbump', 'normal_hard_brake']:
            stress_test_analysis.append({
                "scenario": row['scenario'],
                "sensor_only_acc": int(np.max(ps_aligned) >= 0.5),
                "fused_raw_acc": int(np.max(fused_seq) >= fusion_engine.threshold),
                "temporal_alert": is_clip_alert,
                "suppressed_false_alarm": (is_clip_alert == 0)
            })

    e4_metrics = compute_metrics(test_y, e4_clip_preds, e4_clip_probs)

    # Package 4 Core Experiments
    core_experiments = {
        "E1: Video-Only": e1_metrics,
        "E2: Sensor-Only": e2_metrics,
        "E3: Video+Sensor Fusion": e3_metrics,
        "E4: Temporal Decision": e4_metrics
    }

    # =========================================================================
    # PART 5: VISUALIZATIONS & REPORT GENERATION
    # =========================================================================
    print("\n>>> Generating publication-quality plots...")
    plot_confusion_matrices(core_experiments, RESULTS_DIR / "confusion_matrices.png")
    plot_experiment_comparisons(core_experiments, RESULTS_DIR / "experiment_comparison_chart.png")
    
    roc_data = {
        "E1: Video-Only (MobileNetV3)": (test_y, test_pv),
        "E2: Sensor-Only (Random Forest)": (test_y, test_ps),
        "E3: Weighted Fusion": (test_y, e3_probs),
        "E3 (Alt): Stacking Meta": (test_y, stacking_probs)
    }
    plot_roc_curves(roc_data, RESULTS_DIR / "roc_curves.png")

    # Save structured summary JSON
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "core_experiments": core_experiments,
        "video_models_comparison": video_test_results,
        "sensor_models_comparison": sensor_test_results,
        "fusion_methods_comparison": {
            "weighted_probability_fusion": e3_metrics,
            "stacking_meta_fusion": stacking_metrics,
            "optimal_alpha": fusion_engine.alpha,
            "optimal_threshold": fusion_engine.threshold
        },
        "false_alarm_stress_test": stress_test_analysis
    }

    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # Print Formatted Markdown Table
    print("\n" + "=" * 80)
    print("                    PHASE 1 EXPERIMENT EVALUATION TABLE")
    print("=" * 80)
    header = f"| {'Experiment':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FAR (FP Rate)':<12} |"
    sep = "|" + "-"*27 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|"
    print(header)
    print(sep)
    for exp_k, m in core_experiments.items():
        row_str = f"| {exp_k:<25} | {m['accuracy']*100:>8.1f}% | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1']:>10.3f} | {m['false_alarm_rate']*100:>10.1f}% |"
        print(row_str)
    print(sep)

    print("\n" + "=" * 80)
    print("              MODEL ARCHITECTURE COMPARISON (Ablation Analysis)")
    print("=" * 80)
    print("VISION MODELS:")
    for arch_k, m in video_test_results.items():
        print(f"  • {arch_k.upper():<20} -> Acc: {m['accuracy']*100:.1f}% | Prec: {m['precision']*100:.1f}% | Rec: {m['recall']*100:.1f}% | F1: {m['f1']:.3f}")
    
    print("\nSENSOR MODELS:")
    for sm_k, m in sensor_test_results.items():
        print(f"  • {sm_k.upper():<20} -> Acc: {m['accuracy']*100:.1f}% | Prec: {m['precision']*100:.1f}% | Rec: {m['recall']*100:.1f}% | F1: {m['f1']:.3f}")

    print("\nFUSION COMPARISON:")
    print(f"  • Weighted Fusion (alpha={fusion_engine.alpha:.2f}) -> F1: {e3_metrics['f1']:.3f} | Acc: {e3_metrics['accuracy']*100:.1f}%")
    print(f"  • Stacking Meta-Classifier        -> F1: {stacking_metrics['f1']:.3f} | Acc: {stacking_metrics['accuracy']*100:.1f}%")

    print("\nResults and figures successfully exported to: results/")
    print("=" * 80)
    return summary_report


if __name__ == '__main__':
    run_phase1_benchmark()
