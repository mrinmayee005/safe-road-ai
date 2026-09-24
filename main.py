"""
Safe Road AI - Master Command Line Interface (CLI)
Entrypoint for dataset generation, training, evaluation benchmarks, and live inference.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Safe Road AI: Smartphone-Based Four-Wheeler Accident Detection (Phase 1)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate dataset clips (synthetic, ccd, or telematics)")
    p_gen.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "ccd", "telematics", "all"], help="Dataset to build")
    p_gen.add_argument("--samples", type=int, default=6, help="Samples per category")

    # Command: benchmark / evaluate
    p_bench = subparsers.add_parser("benchmark", help="Train all models and run E1, E2, E3, and E4 experimental benchmarks")
    p_bench.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "ccd", "telematics", "all"], help="Dataset benchmark to run")

    # Command: compare
    subparsers.add_parser("compare", help="Run comprehensive cross-dataset accuracy mapping and deep comparison")

    # Command: infer
    p_infer = subparsers.add_parser("infer", help="Run synchronized multimodal inference on a paired test clip")
    p_infer.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4)")
    p_infer.add_argument("--sensor", type=str, required=True, help="Path to synchronized sensor file (.csv)")
    p_infer.add_argument("--video-model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"], help="Visual model")
    p_infer.add_argument("--sensor-model", type=str, default="random_forest", choices=["random_forest", "gradient_boosting", "extra_trees"], help="Sensor model")
    p_infer.add_argument("--alpha", type=float, default=0.55, help="Video weight alpha for fusion (default: 0.55)")
    p_infer.add_argument("--threshold", type=float, default=0.50, help="Decision threshold T (default: 0.50)")
    p_infer.add_argument("--render", action="store_true", help="Render output video with real-time HUD telemetry")
    p_infer.add_argument("--output", type=str, default=None, help="Path to save annotated video")

    # Command: demo / app
    subparsers.add_parser("demo", help="Launch interactive Streamlit evaluation dashboard")

    args = parser.parse_args()

    if args.command == "generate-data":
        if args.dataset in ["synthetic", "all"]:
            from data.dataset_generator import generate_paired_dataset
            generate_paired_dataset(num_samples_per_category=args.samples)
        if args.dataset in ["ccd", "all"]:
            from data.ccd_dataset_builder import build_ccd_dataset
            build_ccd_dataset(num_samples=100)
        if args.dataset in ["telematics", "all"]:
            from data.telematics_dataset_builder import build_telematics_dataset
            build_telematics_dataset(num_samples_per_category=15)

    elif args.command == "benchmark":
        if args.dataset == "synthetic":
            from experiments.run_experiments import run_phase1_benchmark
            run_phase1_benchmark()
        elif args.dataset == "ccd":
            from experiments.run_experiments_ccd import run_ccd_benchmark
            run_ccd_benchmark()
        elif args.dataset == "telematics":
            from experiments.run_experiments_telematics import run_telematics_benchmark
            run_telematics_benchmark()
        elif args.dataset == "all":
            from experiments.run_experiments import run_phase1_benchmark
            from experiments.run_experiments_ccd import run_ccd_benchmark
            from experiments.run_experiments_telematics import run_telematics_benchmark
            from src.evaluation.cross_dataset_comparison import generate_cross_dataset_comparison
            print("\n>>> Running Benchmark on Synthetic Dataset...")
            run_phase1_benchmark()
            print("\n>>> Running Benchmark on User Real Dashcam (CCD) Dataset...")
            run_ccd_benchmark()
            print("\n>>> Running Benchmark on Real-World Telematics Dataset...")
            run_telematics_benchmark()
            print("\n>>> Generating Cross-Dataset Accuracy Mapping...")
            generate_cross_dataset_comparison()

    elif args.command == "compare":
        from src.evaluation.cross_dataset_comparison import generate_cross_dataset_comparison
        generate_cross_dataset_comparison()

    elif args.command == "infer":
        from src.pipeline.inference_engine import MultimodalInferenceEngine
        print(f"\n[Safe Road AI] Initializing Multimodal Inference Engine...")
        
        v_path = Path(args.video)
        if not v_path.exists() and (PROJECT_ROOT / args.video).exists():
            v_path = PROJECT_ROOT / args.video
            
        s_path = Path(args.sensor)
        if not s_path.exists() and (PROJECT_ROOT / args.sensor).exists():
            s_path = PROJECT_ROOT / args.sensor

        engine = MultimodalInferenceEngine(
            video_model_arch=args.video_model,
            sensor_model_type=args.sensor_model,
            alpha=args.alpha,
            threshold=args.threshold
        )
        print(f"[Safe Road AI] Running inference on:\n  Video:  {v_path}\n  Sensor: {s_path}")
        result = engine.run_synchronized_inference(
            video_path=v_path,
            sensor_csv_path=s_path,
            render_annotated_video=args.render,
            output_video_path=args.output
        )
        print("\n" + "=" * 50)
        print("               INFERENCE RESULT")
        print("=" * 50)
        print(f"Accident Detected:      {'YES - HIGH RISK COLLISION' if result['accident_detected'] else 'NO - NORMAL DRIVING'}")
        if result['accident_detected']:
            print(f"Detection Timestamp:    {result['first_alert_time_sec']} seconds")
            print(f"Emergency GPS Dispatch: Lat {result['gps_coordinates']['latitude']}, Lon {result['gps_coordinates']['longitude']}")
            print(f"Location Details:       {result['gps_coordinates']['location_name']}")
        print(f"Total Video Duration:   {result['total_duration_sec']} seconds")
        print(f"Processing Throughput:  {result['processed_fps']} FPS")
        if result['annotated_video_path']:
            print(f"Annotated Video Saved:  {result['annotated_video_path']}")
        print("=" * 50)

    elif args.command == "demo":
        import subprocess
        print("[Safe Road AI] Starting Streamlit dashboard on http://localhost:8501 ...")
        subprocess.run(["streamlit", "run", "app.py"])

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
