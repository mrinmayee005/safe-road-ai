"""
Safe Road AI - Project 1: Synthetic Dataset Benchmark CLI
Entrypoint for the baseline synthetic dataset.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Safe Road AI: Project 1 - Synthetic Dataset Benchmark")
    parser.add_argument("command", choices=["generate", "benchmark", "infer"], help="Action to execute")
    parser.add_argument("--samples", type=int, default=6, help="Samples per category")
    args = parser.parse_args()

    if args.command == "generate":
        from data.dataset_generator import generate_paired_dataset
        generate_paired_dataset(num_samples_per_category=args.samples)
    elif args.command == "benchmark":
        from experiments.run_experiments import run_phase1_benchmark
        run_phase1_benchmark()
    elif args.command == "infer":
        print("[Synthetic Project] Use 'python main.py infer --video ... --sensor ...'")


if __name__ == "__main__":
    main()
