"""
Safe Road AI - Video Models Module
Implements MobileNetV3-Small (primary mobile candidate) and ResNet18 (comparison candidate)
for binary accident classification from video frames.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from src.config import CHECKPOINTS_DIR, VIDEO_CONFIG
from src.video.preprocessor import VideoPreprocessor


class VideoClassificationModel(nn.Module):
    """
    Wrapper around lightweight CNN backbones for binary accident detection.
    Supports 'mobilenet_v3_small' and 'resnet18'.
    """
    def __init__(self, architecture: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        
        if architecture == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            # Replace classifier head for binary accident probability
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(128, 2)  # [P(normal), P(accident)]
            )
            self.model = backbone
            
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 2)
            )
            self.model = backbone
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'mobilenet_v3_small' or 'resnet18'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrameDataset(Dataset):
    """
    PyTorch Dataset of video frames and corresponding labels.
    """
    def __init__(self, frame_tensors: List[torch.Tensor], labels: List[int]):
        self.frame_tensors = frame_tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.frame_tensors[idx], self.labels[idx]


class VideoPipeline:
    """
    High-level manager for training, evaluating, and running inference with video models.
    """
    def __init__(self, architecture: str = "mobilenet_v3_small", device: str = None):
        self.architecture = architecture
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VideoClassificationModel(architecture=architecture, pretrained=True).to(self.device)
        self.preprocessor = VideoPreprocessor()
        self.checkpoint_path = CHECKPOINTS_DIR / f"video_{architecture}.pth"

    def predict_frames(self, frames_bgr: List[torch.Tensor or any]) -> np.ndarray:
        """
        Runs inference on a list of BGR OpenCV frames.
        Returns:
            np.ndarray of shape (N,) containing P(accident) for each frame.
        """
        if not frames_bgr:
            return np.array([])

        self.model.eval()
        batch_tensor = self.preprocessor.preprocess_batch(frames_bgr).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            accident_probs = probs[:, 1].cpu().numpy()
            
        return accident_probs

    def predict_video(self, video_path: Path or str) -> Tuple[np.ndarray, List[float], float]:
        """
        Samples frames from video, runs inference, and measures inference latency.
        Returns:
            probs: array of accident probabilities Pv for sampled frames
            timestamps: list of timestamps (seconds)
            avg_latency_ms: average inference time per frame in milliseconds
        """
        frames, timestamps = self.preprocessor.sample_frames_from_video(video_path)
        if not frames:
            return np.array([]), [], 0.0

        t0 = time.perf_counter()
        probs = self.predict_frames(frames)
        total_time_ms = (time.perf_counter() - t0) * 1000
        avg_latency_ms = total_time_ms / len(frames) if frames else 0.0

        return probs, timestamps, avg_latency_ms

    def train_model(
        self,
        train_frames: List[torch.Tensor],
        train_labels: List[int],
        val_frames: List[torch.Tensor],
        val_labels: List[int],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Trains the video classifier with validation monitoring.
        """
        train_dataset = FrameDataset(train_frames, train_labels)
        val_dataset = FrameDataset(val_frames, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")

        print(f"\n[Video Training - {self.architecture}] Params: {self.model.count_parameters():,}")
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            
            for frames, labels in train_loader:
                frames, labels = frames.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(labels)

            train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0

            # Validation step
            self.model.eval()
            val_loss_total = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames, labels = frames.to(self.device), labels.to(self.device)
                    outputs = self.model(frames)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item() * len(labels)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)

            val_loss = val_loss_total / total if total > 0 else 0.0
            val_acc = correct / total if total > 0 else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()

        return history

    def save_checkpoint(self, path: Optional[Path] = None):
        target_path = path or self.checkpoint_path
        torch.save({
            "architecture": self.architecture,
            "state_dict": self.model.state_dict()
        }, str(target_path))
        print(f"  Saved model checkpoint to: {target_path}")

    def load_checkpoint(self, path: Optional[Path] = None):
        target_path = path or self.checkpoint_path
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {target_path}")
        checkpoint = torch.load(str(target_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print(f"  Loaded model checkpoint from: {target_path}")
