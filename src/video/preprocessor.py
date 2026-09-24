"""
Safe Road AI - Video Preprocessing Module
Extracts and samples frames using OpenCV, applies image resizing,
normalization, and prepares PyTorch tensors.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from torchvision import transforms

from src.config import VIDEO_CONFIG


class VideoPreprocessor:
    """
    Handles video reading, efficient frame sampling, resizing,
    and PyTorch tensor normalization.
    """
    def __init__(self, target_size: Tuple[int, int] = None, sample_fps: int = None):
        self.target_size = target_size or VIDEO_CONFIG["target_size"]
        self.sample_fps = sample_fps or VIDEO_CONFIG["sample_fps"]
        
        # Standard torchvision normalization for ImageNet pre-trained backbones
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=VIDEO_CONFIG["normalize_mean"],
                std=VIDEO_CONFIG["normalize_std"]
            )
        ])

    def sample_frames_from_video(
        self,
        video_path: Path or str,
        max_duration_sec: Optional[float] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Samples frames uniformly at `sample_fps` rate.
        Returns:
            frames: List of BGR numpy arrays (original resolution)
            timestamps_sec: List of timestamp in seconds for each sampled frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(source_fps / self.sample_fps)))
        
        frames = []
        timestamps = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / source_fps
            if max_duration_sec and current_time > max_duration_sec:
                break

            if frame_idx % frame_interval == 0:
                frames.append(frame)
                timestamps.append(current_time)

            frame_idx += 1

        cap.release()
        return frames, timestamps

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Converts a BGR frame to RGB, resizes, normalizes, and returns
        a PyTorch Tensor shaped (C, H, W).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        return tensor

    def preprocess_batch(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses a list of BGR frames into a batch tensor (B, C, H, W).
        """
        tensors = [self.preprocess_frame(f) for f in frames_bgr]
        if not tensors:
            return torch.empty((0, 3, self.target_size[0], self.target_size[1]))
        return torch.stack(tensors, dim=0)
