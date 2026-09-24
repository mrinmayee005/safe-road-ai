"""
Safe Road AI - Project 3: Real-World Multimodal Telematics Benchmark CLI
Entrypoint for training and evaluating on authentic 50Hz mobile phone IMU and road dynamics (Phase 2 candidate).
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Safe Road AI: Project 3 - Real-World Telematics Benchmark")
    parser.add_argument("command", choices=["build", "benchmark"], help="Action to execute")
    parser.add_argument("--samples", type=int, default=15, help="Samples per scenario")
    args = parser.parse_args()

    if args.command == "build":
        from data.telematics_dataset_builder import build_telematics_dataset
        build_telematics_dataset(num_samples_per_category=args.samples)
    elif args.command == "benchmark":
        from experiments.run_experiments_telematics import run_telematics_benchmark
        run_telematics_benchmark()


if __name__ == "__main__":
    main()
