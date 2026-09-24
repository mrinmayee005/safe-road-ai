"""
Safe Road AI - Multimodal Fusion Module
Implements Weighted Probability Fusion (Pfinal = alpha * Pv + (1 - alpha) * Ps)
with optimal alpha tuning, and a comparison Stacking Meta-Classifier.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


class WeightedProbabilityFusion:
    """
    Weighted Probability Fusion:
        P_final = alpha * P_v + (1 - alpha) * P_s
    """
    def __init__(self, alpha: float = 0.55, threshold: float = 0.50):
        self.alpha = float(alpha)
        self.threshold = float(threshold)

    def fuse(self, pv: np.ndarray or float, ps: np.ndarray or float, alpha: Optional[float] = None) -> np.ndarray:
        """
        Combines visual probability Pv and sensor probability Ps.
        """
        a = self.alpha if alpha is None else float(alpha)
        pv_arr = np.asarray(pv)
        ps_arr = np.asarray(ps)
        p_final = a * pv_arr + (1.0 - a) * ps_arr
        return np.clip(p_final, 0.0, 1.0)

    def predict(self, pv: np.ndarray or float, ps: np.ndarray or float,
                threshold: Optional[float] = None, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            binary_decisions: 1 (accident) if P_final >= threshold else 0
            fused_probabilities: P_final
        """
        t = self.threshold if threshold is None else float(threshold)
        p_final = self.fuse(pv, ps, alpha=alpha)
        decisions = (p_final >= t).astype(int)
        return decisions, p_final

    def optimize_parameters(
        self,
        val_pv: np.ndarray,
        val_ps: np.ndarray,
        val_y: np.ndarray,
        alpha_steps: int = 21,
        threshold_steps: int = 19
    ) -> Dict[str, float]:
        """
        Grid search on validation set to find the optimal alpha and threshold T
        that maximizes the F1-score on the multimodal validation set.
        """
        best_f1 = -1.0
        best_alpha = 0.5
        best_thresh = 0.5
        best_metrics = {}

        alphas = np.linspace(0.0, 1.0, alpha_steps)
        thresholds = np.linspace(0.1, 0.9, threshold_steps)

        for a in alphas:
            p_final = a * val_pv + (1.0 - a) * val_ps
            for t in thresholds:
                preds = (p_final >= t).astype(int)
                f1 = f1_score(val_y, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = a
                    best_thresh = t
                    best_metrics = {
                        "optimal_alpha": round(float(a), 3),
                        "optimal_threshold": round(float(t), 3),
                        "val_f1": round(float(f1), 4),
                        "val_precision": round(float(precision_score(val_y, preds, zero_division=0)), 4),
                        "val_recall": round(float(recall_score(val_y, preds, zero_division=0)), 4),
                        "val_accuracy": round(float(accuracy_score(val_y, preds)), 4)
                    }

        self.alpha = best_alpha
        self.threshold = best_thresh
        print(f"[Fusion Optimizer] Optimal Alpha: {self.alpha:.2f}, Threshold: {self.threshold:.2f} (Val F1: {best_f1:.4f})")
        return best_metrics


class StackingMetaFusion:
    """
    Comparison Candidate: Supervised Logistic Regression Stacking Meta-Classifier.
    Learns a non-linear decision boundary over [Pv, Ps, |Pv - Ps|, Pv * Ps].
    """
    def __init__(self):
        self.meta_model = LogisticRegression(class_weight="balanced", random_state=42)
        self.is_fitted = False

    def _extract_meta_features(self, pv: np.ndarray, ps: np.ndarray) -> np.ndarray:
        pv_col = np.asarray(pv).reshape(-1, 1)
        ps_col = np.asarray(ps).reshape(-1, 1)
        diff_col = np.abs(pv_col - ps_col)
        prod_col = pv_col * ps_col
        return np.hstack([pv_col, ps_col, diff_col, prod_col])

    def fit(self, pv: np.ndarray, ps: np.ndarray, y: np.ndarray):
        X_meta = self._extract_meta_features(pv, ps)
        self.meta_model.fit(X_meta, y)
        self.is_fitted = True

    def predict(self, pv: np.ndarray, ps: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            # Fallback to simple mean if not fitted
            p_final = 0.5 * pv + 0.5 * ps
            return (p_final >= threshold).astype(int), p_final
            
        X_meta = self._extract_meta_features(pv, ps)
        probs = self.meta_model.predict_proba(X_meta)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs
