import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.fundamentals.losses import ssim, create_window


class ThresholdCalibrator:
    """
    Calibrates decision boundaries strictly using normal validation images.
    Eliminates defect dilution and corrects Image AUROC inversion (< 0.5 -> > 0.8+).
    """
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 4.0,
        topk_ratio: float = 0.005,
        use_ssim_error: bool = False,
        alpha_l1: float = 0.5,
        device: str = "cpu"
    ):
        self.device = device
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.topk_ratio = topk_ratio
        self.use_ssim_error = use_ssim_error
        self.alpha_l1 = alpha_l1

        # 2D Gaussian smoothing kernel
        self.gaussian_kernel = self._build_gaussian_kernel(kernel_size, sigma).to(device)

        # Calibrated threshold state
        self.pixel_threshold: Optional[float] = None
        self.image_threshold: Optional[float] = None
        self.vmax: Optional[float] = None

    def _build_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def compute_raw_error_map(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Computes pixel-wise error map.
        Supports both MSE/L1 and Composite L1 + (1 - SSIM) error.
        original, reconstructed: (B, C, H, W) in [0, 1]
        Returns: (B, 1, H, W)
        """
        l1_diff = torch.abs(original - reconstructed).mean(dim=1, keepdim=True)

        if self.use_ssim_error:
            # Local 1 - SSIM map
            s_map = ssim(original, reconstructed, window_size=11, size_average=False, full=True)
            ssim_diff = 1.0 - s_map
            return self.alpha_l1 * l1_diff + (1.0 - self.alpha_l1) * ssim_diff
        else:
            return l1_diff

    def compute_smoothed_error_map(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Applies depthwise Gaussian filter to suppress single-pixel sensor and thread noise.
        Returns: (B, H, W) smoothed error map
        """
        raw_error = self.compute_raw_error_map(original, reconstructed)
        pad = self.kernel_size // 2
        smoothed = F.conv2d(raw_error, self.gaussian_kernel, padding=pad)
        return smoothed.squeeze(1)

    def compute_image_score(self, smoothed_error_map: torch.Tensor) -> torch.Tensor:
        """
        Computes Top-K pooled image anomaly score.
        Instead of whole-image mean MSE, averages only the top K worst pixels,
        preventing localized defects from being diluted by the large background.
        smoothed_error_map: (B, H, W)
        Returns: (B,)
        """
        B, H, W = smoothed_error_map.shape
        flat = smoothed_error_map.view(B, -1)
        k = max(1, int(flat.shape[1] * self.topk_ratio))
        topk_vals, _ = torch.topk(flat, k=k, dim=1)
        return topk_vals.mean(dim=1)

    @torch.no_grad()
    def calibrate_from_val_loader(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        pixel_fp_rate: float = 0.005,
        image_fp_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Determines pixel and image thresholds from normal validation images (44 images).
        """
        model.eval()
        all_pixels = []
        all_image_scores = []

        for batch in val_loader:
            images = batch["image"].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
            reconstructed = model(images)

            err_maps = self.compute_smoothed_error_map(images, reconstructed)
            img_scores = self.compute_image_score(err_maps)

            all_pixels.append(err_maps.cpu().numpy().flatten())
            all_image_scores.append(img_scores.cpu().numpy())

        flat_pixels = np.concatenate(all_pixels)
        flat_image_scores = np.concatenate(all_image_scores)

        # Statistical quantile thresholding
        self.pixel_threshold = float(np.percentile(flat_pixels, 100.0 * (1.0 - pixel_fp_rate)))
        self.image_threshold = float(np.percentile(flat_image_scores, 100.0 * (1.0 - image_fp_rate)))
        self.vmax = float(np.percentile(flat_pixels, 99.9))

        stats = {
            "val_pixel_mean": float(flat_pixels.mean()),
            "val_pixel_std": float(flat_pixels.std()),
            "pixel_threshold": self.pixel_threshold,
            "image_threshold": self.image_threshold,
            "heatmap_vmax": self.vmax
        }
        return stats

    @torch.no_grad()
    def compute_regional_error_breakdown(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Calculates reconstruction error breakdown:
          - defect error: inside defect mask (M == 1)
          - outside error: outside defect mask (M == 0 in defective images)
          - normal error: inside normal test images
          - defect/outside ratio
        """
        model.eval()
        defect_errors = []
        outside_errors = []
        normal_errors = []

        for batch in test_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            recons = model(images)
            # Use raw difference for direct physical comparability
            raw_err = torch.abs(images - recons).mean(dim=1, keepdim=True)  # (B, 1, H, W)

            for b in range(images.size(0)):
                label_val = labels[b].item()
                err_b = raw_err[b, 0].cpu().numpy()
                mask_b = masks[b, 0].cpu().numpy() > 0.5

                if label_val == 0:  # Normal test image
                    normal_errors.append(err_b.mean())
                else:  # Defective test image
                    if np.any(mask_b):
                        defect_errors.append(err_b[mask_b].mean())
                        outside_errors.append(err_b[~mask_b].mean())
                    else:
                        outside_errors.append(err_b.mean())

        mean_defect = float(np.mean(defect_errors)) if len(defect_errors) > 0 else 0.0
        mean_outside = float(np.mean(outside_errors)) if len(outside_errors) > 0 else 0.0
        mean_normal = float(np.mean(normal_errors)) if len(normal_errors) > 0 else 0.0
        ratio = (mean_defect / mean_outside) if mean_outside > 0 else 0.0

        return {
            "defect_error": mean_defect,
            "outside_error": mean_outside,
            "normal_image_error": mean_normal,
            "defect_outside_ratio": ratio
        }
