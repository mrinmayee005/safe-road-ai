"""
Safe Road AI - Temporal Confirmation & Smoothing Module
Applies a sliding smoothing window and persistence rule over sequential fused predictions
to suppress isolated false positives (e.g. potholes, road bumps, sudden hard braking).
"""

from typing import List, Tuple, Optional
from collections import deque
import numpy as np

from src.config import TEMPORAL_CONFIG


class TemporalDecisionEngine:
    """
    Stateful and batch temporal decision filter:
    1. Moving Average / Exponential Smoothing over past W windows.
    2. Persistence Rule: requires at least K out of W windows to exceed threshold T.
    3. Cooldown: prevents repeated alerts within cooldown_sec.
    """
    def __init__(
        self,
        smoothing_window_size: int = None,
        persistence_required: int = None,
        threshold: float = 0.50,
        cooldown_sec: float = None
    ):
        self.window_size = smoothing_window_size or TEMPORAL_CONFIG["smoothing_window_size"]
        self.persistence_required = persistence_required or TEMPORAL_CONFIG["persistence_required"]
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec or TEMPORAL_CONFIG["cooldown_seconds"]
        
        self.history = deque(maxlen=self.window_size)
        self.last_alert_time = -999.0

    def reset(self):
        """Clears state buffer."""
        self.history.clear()
        self.last_alert_time = -999.0

    def update(self, p_final: float, current_timestamp_sec: float) -> Tuple[bool, float]:
        """
        Processes a single incoming fused probability in real-time.
        Returns:
            is_confirmed_alert: True if temporal criteria met and cooldown elapsed
            smoothed_prob: Moving average probability
        """
        self.history.append(float(p_final))
        smoothed_prob = float(np.mean(self.history))
        
        # Check persistence condition: count elements exceeding threshold
        high_risk_hits = sum(1 for p in self.history if p >= self.threshold)
        meets_persistence = (high_risk_hits >= self.persistence_required) and (smoothed_prob >= self.threshold)
        
        # Check cooldown
        time_since_last = current_timestamp_sec - self.last_alert_time
        in_cooldown = time_since_last < self.cooldown_sec

        if meets_persistence and not in_cooldown:
            self.last_alert_time = current_timestamp_sec
            return True, smoothed_prob
            
        return False, smoothed_prob

    def process_sequence(
        self,
        p_final_seq: np.ndarray or List[float],
        timestamps_sec: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch processing over a sequence of predictions for offline evaluation (E4).
        Returns:
            alert_flags: binary array (1 if confirmed alert at timestamp, else 0)
            smoothed_probs: array of smoothed probabilities
        """
        self.reset()
        alert_flags = []
        smoothed_probs = []

        for p, t in zip(p_final_seq, timestamps_sec):
            alert, s_prob = self.update(p, t)
            alert_flags.append(1 if alert else 0)
            smoothed_probs.append(s_prob)

        return np.array(alert_flags), np.array(smoothed_probs)
