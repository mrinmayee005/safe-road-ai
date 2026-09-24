"""
Safe Road AI - Sensor Models Module
Implements Random Forest (primary candidate), Gradient Boosting, and Extra Trees
classifiers for tabular sensor features extracted from accelerometer and gyroscope.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import CHECKPOINTS_DIR, SENSOR_CONFIG
from src.sensor.features import process_sensor_timeseries, extract_window_features, compute_magnitudes


class SensorClassifier:
    """
    Wrapper for ML classifiers on tabular IMU features.
    Supports 'random_forest', 'gradient_boosting', and 'extra_trees'.
    """
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.checkpoint_path = CHECKPOINTS_DIR / f"sensor_{model_type}.joblib"
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        elif model_type == "extra_trees":
            self.model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'random_forest', 'gradient_boosting', or 'extra_trees'.")

    def train(self, X_train: pd.DataFrame or np.ndarray, y_train: List[int] or np.ndarray,
              X_val: Optional[pd.DataFrame or np.ndarray] = None,
              y_val: Optional[List[int] or np.ndarray] = None) -> Dict[str, float]:
        """
        Fits the scaler and classifier on training features.
        Returns validation metrics.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train_mat = np.nan_to_num(X_train.values, nan=0.0, posinf=1e5, neginf=-1e5)
        else:
            X_train_mat = np.nan_to_num(np.array(X_train), nan=0.0, posinf=1e5, neginf=-1e5)

        X_train_scaled = self.scaler.fit_transform(X_train_mat)
        self.model.fit(X_train_scaled, y_train)

        metrics = {}
        if X_val is not None and y_val is not None:
            val_preds, val_probs = self.predict(X_val)
            metrics["accuracy"] = float(accuracy_score(y_val, val_preds))
            metrics["precision"] = float(precision_score(y_val, val_preds, zero_division=0))
            metrics["recall"] = float(recall_score(y_val, val_preds, zero_division=0))
            metrics["f1"] = float(f1_score(y_val, val_preds, zero_division=0))
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_val, val_probs))
            except Exception:
                metrics["roc_auc"] = 0.5
                
            print(f"[Sensor {self.model_type}] Val Acc: {metrics['accuracy']*100:.1f}%, F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")

        self.save_model()
        return metrics

    def predict(self, X: pd.DataFrame or np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            binary_predictions: 0 (normal) or 1 (accident)
            accident_probabilities: P(sensor = accident)
        """
        if isinstance(X, pd.DataFrame):
            X_mat = np.nan_to_num(X.values, nan=0.0, posinf=1e5, neginf=-1e5)
        else:
            X_mat = np.nan_to_num(np.array(X), nan=0.0, posinf=1e5, neginf=-1e5)
            
        X_scaled = self.scaler.transform(X_mat)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def predict_timeseries_file(self, csv_path: Path or str) -> Tuple[np.ndarray, List[float], List[int], float]:
        """
        Reads a raw IMU sensor CSV file, segments it into windows, extracts features,
        and produces accident probabilities for each window.
        
        Returns:
            probs: array of Ps for each window
            timestamps: list of window center timestamps
            labels: ground-truth labels if available
            avg_latency_ms: latency per window in milliseconds
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Sensor file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        features_df, timestamps, labels = process_sensor_timeseries(df)
        
        if len(features_df) == 0:
            return np.array([]), [], [], 0.0

        t0 = time.perf_counter()
        _, probs = self.predict(features_df)
        total_time_ms = (time.perf_counter() - t0) * 1000
        avg_latency_ms = total_time_ms / len(features_df)

        return probs, timestamps, labels, avg_latency_ms

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance scores sorted descending.
        """
        if not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [f"f_{i}" for i in range(len(importances))]
        sorted_pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in sorted_pairs}

    def save_model(self, path: Optional[Path] = None):
        target_path = path or self.checkpoint_path
        joblib.dump({
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }, str(target_path))
        print(f"  Saved sensor model to: {target_path}")

    def load_model(self, path: Optional[Path] = None):
        target_path = path or self.checkpoint_path
        if not target_path.exists():
            raise FileNotFoundError(f"Sensor model checkpoint not found: {target_path}")
        data = joblib.load(str(target_path))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", [])
        self.model_type = data.get("model_type", self.model_type)
        print(f"  Loaded sensor model from: {target_path}")
