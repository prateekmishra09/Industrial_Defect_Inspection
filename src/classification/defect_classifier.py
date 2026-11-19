from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn


class LightweightDefectClassifier(nn.Module):
    """
    Lightweight CNN to classify detected defect patches into:
    ['bent', 'color', 'flip', 'scratch']
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


class HeuristicDefectClassifier:
    """
    Rule-based geometric and photometric classifier for MVTec Metal Nut defects.
    Uses aspect ratio, color variance, and radial position.
    """
    CLASSES = ["bent", "color", "flip", "scratch"]

    @staticmethod
    def classify_defect(
        image_np: np.ndarray,
        mask_np: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        image_np: (H, W, 3) in [0, 1]
        mask_np: (H, W) boolean
        bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        patch_img = image_np[y:y+h, x:x+w]
        patch_mask = mask_np[y:y+h, x:x+w]

        if not np.any(patch_mask):
            return "unknown"

        aspect_ratio = max(w, h) / max(1, min(w, h))

        # 1. Scratch: high aspect ratio (elongated line)
        if aspect_ratio >= 2.5:
            return "scratch"

        # 2. Color: high color variance (red/blue delta) in the defect region
        pixels = patch_img[patch_mask]
        r_mean = np.mean(pixels[:, 0])
        b_mean = np.mean(pixels[:, 2])
        color_diff = abs(r_mean - b_mean)
        if color_diff > 0.08:
            return "color"

        # 3. Position relative to image center
        img_h, img_w, _ = image_np.shape
        center_x, center_y = img_w // 2, img_h // 2
        cx, cy = x + w / 2, y + h / 2
        dist_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        # Outer rim vs inner hole
        if dist_from_center > (img_w * 0.32):
            return "bent"
        else:
            return "flip"
