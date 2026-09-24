"""
Safe Road AI - Cross-Dataset Accuracy Mapping & Deep Comparison CLI
Runs the unified comparative analysis across:
  - Synthetic Dataset
  - User Real Dashcam Dataset (CCD)
  - Real Multimodal Telematics Benchmark
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cross_dataset_comparison import generate_cross_dataset_comparison


def main():
    print("\n" + "=" * 80)
    print("       SAFE ROAD AI: CROSS-DATASET ACCURACY MAPPING & DEEP COMPARISON")
    print("=" * 80)
    generate_cross_dataset_comparison()


if __name__ == "__main__":
    main()
