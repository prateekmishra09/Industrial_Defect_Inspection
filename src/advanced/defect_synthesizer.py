import random
from typing import Tuple, Dict, Optional
import numpy as np
import cv2
import torch


class IndustrialDefectSynthesizer:
    """
    Procedural Defect Synthesis Engine for MVTec AD Metal Nut.
    Generates realistic synthetic defects paired with precise pixel ground-truth masks.
    Supports four standard defect categories: 'scratch', 'bent', 'color', 'flip'.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def synthesize_scratch(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes realistic tool gouge/scratch: darker trough with specular edges.
        image_np: (H, W, 3) in [0, 1]
        Returns: (synthetic_image, ground_truth_mask)
        """
        h, w, c = image_np.shape
        synth = image_np.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        # Random start and length on the nut body
        center_x, center_y = w // 2, h // 2
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(20, 60)
        start_x = int(center_x + random.uniform(25, 60) * np.cos(angle))
        start_y = int(center_y + random.uniform(25, 60) * np.sin(angle))
        end_x = int(start_x + length * np.cos(angle + random.uniform(-0.5, 0.5)))
        end_y = int(start_y + length * np.sin(angle + random.uniform(-0.5, 0.5)))

        thickness = random.randint(1, 3)
        # Gouge: dark trough
        cv2.line(synth, (start_x, start_y), (end_x, end_y), (0.1, 0.1, 0.12), thickness)
        # Ridge: slight specular edge
        cv2.line(synth, (start_x + 1, start_y + 1), (end_x + 1, end_y + 1), (0.75, 0.75, 0.8), 1)

        # Ground truth mask
        cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, thickness + 2)

        return np.clip(synth, 0, 1), mask

    def synthesize_bent(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes edge dent / deformation along the hexagonal profile.
        """
        h, w, c = image_np.shape
        synth = image_np.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        center_x, center_y = w // 2, h // 2
        radius = int(w * 0.38)
        # Pick point near the outer perimeter
        angle = random.uniform(0, 2 * np.pi)
        px = int(center_x + radius * np.cos(angle))
        py = int(center_y + radius * np.sin(angle))

        axes = (random.randint(15, 30), random.randint(10, 20))
        rot = random.uniform(0, 180)

        # Cut out or dent: darker indentation
        cv2.ellipse(synth, (px, py), axes, rot, 0, 360, (0.15, 0.15, 0.18), -1)
        cv2.ellipse(mask, (px, py), axes, rot, 0, 360, 1.0, -1)

        return np.clip(synth, 0, 1), mask

    def synthesize_color(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes localized chemical stain, oxidation, or thermal discoloration.
        """
        h, w, c = image_np.shape
        synth = image_np.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        center_x, center_y = w // 2, h // 2
        angle = random.uniform(0, 2 * np.pi)
        px = int(center_x + random.uniform(20, 50) * np.cos(angle))
        py = int(center_y + random.uniform(20, 50) * np.sin(angle))
        radius = random.randint(12, 28)

        # Stain spot with soft Gaussian falloff
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        stain_region = dist <= radius

        # Oxidation brown/amber tint
        tint = np.array([0.45, 0.28, 0.12])  # Rust/corrosion color
        synth[stain_region] = 0.4 * synth[stain_region] + 0.6 * tint
        mask[stain_region] = 1.0

        return np.clip(synth, 0, 1), mask

    def synthesize_flip(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes flipped part defect / missing thread / structural void.
        """
        h, w, c = image_np.shape
        synth = image_np.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        center_x, center_y = w // 2, h // 2
        # Circular defect on inner thread border
        inner_r = int(w * 0.18)
        angle = random.uniform(0, 2 * np.pi)
        px = int(center_x + inner_r * np.cos(angle))
        py = int(center_y + inner_r * np.sin(angle))
        radius = random.randint(8, 16)

        cv2.circle(synth, (px, py), radius, (0.05, 0.05, 0.07), -1)
        cv2.circle(mask, (px, py), radius, 1.0, -1)

        return np.clip(synth, 0, 1), mask

    def generate(self, image_tensor: torch.Tensor, category: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Takes an image tensor (C, H, W) in [0, 1], applies synthesis, returns dict of tensors.
        """
        cats = ["scratch", "bent", "color", "flip"]
        chosen_cat = category if category in cats else random.choice(cats)

        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        if chosen_cat == "scratch":
            synth_np, mask_np = self.synthesize_scratch(img_np)
        elif chosen_cat == "bent":
            synth_np, mask_np = self.synthesize_bent(img_np)
        elif chosen_cat == "color":
            synth_np, mask_np = self.synthesize_color(img_np)
        else:
            synth_np, mask_np = self.synthesize_flip(img_np)

        return {
            "synthetic_image": torch.from_numpy(synth_np).permute(2, 0, 1).float(),
            "synthetic_mask": torch.from_numpy(mask_np).unsqueeze(0).float(),
            "category": chosen_cat
        }
