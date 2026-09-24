"""
Safe Road AI - Project 2: User Real-World Dashcam Dataset (CCD) CLI
Entrypoint for training and evaluating on the user-provided 75,000 real crash frames.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Safe Road AI: Project 2 - User Real Dashcam Dataset (CCD)")
    parser.add_argument("command", choices=["build", "benchmark"], help="Action to execute")
    parser.add_argument("--samples", type=int, default=100, help="Clips to compile from archive")
    args = parser.parse_args()

    if args.command == "build":
        from data.ccd_dataset_builder import build_ccd_dataset
        build_ccd_dataset(num_samples=args.samples)
    elif args.command == "benchmark":
        from experiments.run_experiments_ccd import run_ccd_benchmark
        run_ccd_benchmark()


if __name__ == "__main__":
    main()
