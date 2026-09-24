"""
Safe Road AI - Evaluation Metrics Module
Computes Accuracy, Precision, Recall, F1-score, Specificity, False Alarm Rate,
and Confusion Matrix for accident detection models.
"""

from typing import Dict, Any, List, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard metrics required by the research specification:
    Accuracy, Precision, Recall, F1-Score, Specificity, False Alarm Rate,
    Confusion Matrix (TP, TN, FP, FN), and ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate (FPR)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "false_alarm_rate": float(far),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist()
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["roc_auc"] = float(auc)
        except Exception:
            metrics["roc_auc"] = 0.5
            
    return metrics
