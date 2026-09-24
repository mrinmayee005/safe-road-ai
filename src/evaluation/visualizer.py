"""
Safe Road AI - Visualization Module
Generates publication-quality charts: 4-panel confusion matrices,
experiment comparison bar charts, multi-model ablations, and ROC curves.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from src.config import RESULTS_DIR

# Set clean aesthetic style using standard matplotlib
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


def plot_confusion_matrices(
    experiments_data: Dict[str, Dict[str, Any]],
    output_path: Path = None
):
    """
    Plots a multi-panel grid of confusion matrices for E1, E2, E3, and E4 using pure Matplotlib.
    """
    output_path = output_path or RESULTS_DIR / "confusion_matrices.png"
    n_exp = len(experiments_data)
    fig, axes = plt.subplots(1, n_exp, figsize=(4.2 * n_exp, 3.8), dpi=200)
    
    if n_exp == 1:
        axes = [axes]
        
    labels = ["Normal", "Accident"]

    for ax, (exp_name, res) in zip(axes, experiments_data.items()):
        cm = np.array(res["confusion_matrix"])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Annotate cells
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:d}",
                        ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(f"{exp_name}\nAcc: {res['accuracy']*100:.1f}% | F1: {res['f1']:.3f}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.grid(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Saved confusion matrices to: {output_path}")


def plot_experiment_comparisons(
    experiments_data: Dict[str, Dict[str, Any]],
    output_path: Path = None
):
    """
    Renders grouped bar chart comparing Accuracy, Precision, Recall, and F1 across E1 to E4.
    """
    output_path = output_path or RESULTS_DIR / "experiment_comparison_chart.png"
    
    exp_names = list(experiments_data.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(exp_names))
    width = 0.18
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    for i, (m_key, m_label, color) in enumerate(zip(metrics_names, metric_labels, colors)):
        vals = [experiments_data[e][m_key] * 100 for e in exp_names]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=m_label, color=color, alpha=0.9, edgecolor="none")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Safe Road AI — Phase 1 Experimental Comparison (E1 to E4)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Saved experiment comparison chart to: {output_path}")


def plot_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path = None
):
    """
    Plots multi-model ROC Curves.
    curves_data: Dict mapping name -> (y_true, y_probs)
    """
    output_path = output_path or RESULTS_DIR / "roc_curves.png"
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    
    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]

    for (name, (y_true, y_prob)), color in zip(curves_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (False Alarms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Detection Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Safe Road AI — Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Saved ROC curves to: {output_path}")
