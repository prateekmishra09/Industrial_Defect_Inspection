import os
from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.calibrator import ThresholdCalibrator


class InspectionVisualizer:
    """
    Standardized visualizer for industrial defect inspection.
    Enforces common heatmap scales across all images and models, eliminating
    the visual distortion of per-image 99.5th percentile normalization.
    """
    def __init__(self, calibrator: ThresholdCalibrator, output_dir: str = "./reports/figures"):
        self.calibrator = calibrator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_single_sample(
        self,
        original: torch.Tensor,
        gt_mask: torch.Tensor,
        reconstructed: torch.Tensor,
        error_map: torch.Tensor,
        sample_title: str = "Inspection Sample",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Generates a 5-panel standardized comparison:
        1. Original Image
        2. Ground Truth Defect Mask
        3. Model Reconstruction
        4. Calibrated Error Heatmap (LOCKED COMMON SCALE)
        5. Predicted Defect Overlay (where Error > tau_pixel)
        """
        orig_np = original.permute(1, 2, 0).cpu().numpy()
        recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
        gt_np = gt_mask.squeeze().cpu().numpy()
        err_np = error_map.cpu().numpy()

        # Binary prediction using calibrated threshold
        pred_mask = err_np > self.calibrator.pixel_threshold

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        # 1. Original Image
        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title(f"Input: {sample_title}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth Mask
        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. Reconstructed Image
        axes[2].imshow(np.clip(recon_np, 0, 1))
        axes[2].set_title("Reconstruction", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # 4. Calibrated Error Heatmap
        vmax = self.calibrator.vmax if self.calibrator.vmax is not None else (err_np.max() + 1e-5)
        im = axes[3].imshow(err_np, cmap="inferno", vmin=0.0, vmax=vmax)
        axes[3].set_title(f"Calibrated Heatmap\n[0.0, {vmax:.4f}]", fontsize=11, fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        # 5. Anomaly Overlay
        overlay = np.clip(orig_np.copy(), 0, 1)
        # Highlight predicted defect pixels in red with alpha blending
        red_tint = np.zeros_like(overlay)
        red_tint[..., 0] = 1.0  # Red
        alpha = 0.55
        overlay[pred_mask] = (1 - alpha) * overlay[pred_mask] + alpha * red_tint[pred_mask]

        axes[4].imshow(overlay)
        axes[4].set_title(f"Predicted Overlay\n(τ_pixel={self.calibrator.pixel_threshold:.4f})", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        plt.tight_layout()
        if save_name:
            out_path = os.path.join(self.output_dir, save_name)
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            print(f"Figure saved to {out_path}")

        return fig

    @torch.no_grad()
    def plot_defect_category_gallery(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        save_name: str = "defect_category_gallery.png"
    ):
        """
        Collects one representative sample from each category:
        ['good', 'bent', 'color', 'flip', 'scratch'] and visualizes them on a unified grid.
        """
        model.eval()
        found_categories: Dict[str, Dict] = {}
        target_cats = ["good", "bent", "color", "flip", "scratch"]

        for batch in test_loader:
            cat = batch["category"][0]
            if cat in target_cats and cat not in found_categories:
                images = batch["image"].to(self.calibrator.device)
                recons = model(images)
                smoothed_err = self.calibrator.compute_smoothed_error_map(images, recons)

                found_categories[cat] = {
                    "image": batch["image"][0],
                    "mask": batch["mask"][0],
                    "recon": recons[0],
                    "error": smoothed_err[0],
                    "category": cat
                }
            if len(found_categories) == len(target_cats):
                break

        # Plot 5 rows x 5 columns
        fig, axes = plt.subplots(len(found_categories), 5, figsize=(22, 4.2 * len(found_categories)))
        if len(found_categories) == 1:
            axes = np.expand_dims(axes, 0)

        vmax = self.calibrator.vmax or 0.005

        for i, cat in enumerate(target_cats):
            if cat not in found_categories:
                continue
            item = found_categories[cat]
            orig_np = item["image"].permute(1, 2, 0).cpu().numpy()
            recon_np = item["recon"].permute(1, 2, 0).cpu().numpy()
            gt_np = item["mask"].squeeze().cpu().numpy()
            err_np = item["error"].cpu().numpy()
            pred_mask = err_np > self.calibrator.pixel_threshold

            # Col 1: Original
            axes[i, 0].imshow(np.clip(orig_np, 0, 1))
            axes[i, 0].set_title(f"Class: {cat.upper()}", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            # Col 2: Ground Truth Mask
            axes[i, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground Truth", fontsize=11)
            axes[i, 1].axis("off")

            # Col 3: Reconstruction
            axes[i, 2].imshow(np.clip(recon_np, 0, 1))
            axes[i, 2].set_title("Reconstructed", fontsize=11)
            axes[i, 2].axis("off")

            # Col 4: Calibrated Heatmap
            im = axes[i, 3].imshow(err_np, cmap="inferno", vmin=0.0, vmax=vmax)
            axes[i, 3].set_title(f"Heatmap [0, {vmax:.4f}]", fontsize=11)
            axes[i, 3].axis("off")
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

            # Col 5: Overlay
            overlay = np.clip(orig_np.copy(), 0, 1)
            red_tint = np.zeros_like(overlay)
            red_tint[..., 0] = 1.0
            overlay[pred_mask] = 0.5 * overlay[pred_mask] + 0.5 * red_tint[pred_mask]

            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title("Calibrated Overlay", fontsize=11)
            axes[i, 4].axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Category gallery saved to: {out_path}")
        return fig
