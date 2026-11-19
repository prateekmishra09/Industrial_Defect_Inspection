import os
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, jaccard_score
import pandas as pd

from src.evaluation.calibrator import ThresholdCalibrator


def compute_comprehensive_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    calibrator: ThresholdCalibrator,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates image-level and pixel-level performance for one MVTec category.
    Returns quantitative metrics and per-defect subtype breakdowns.
    """
    model.eval()

    gt_masks_flat = []
    pred_pixel_scores_flat = []

    gt_image_labels = []
    pred_image_scores_topk = []
    pred_image_scores_mean = []

    category_records: List[Dict] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            categories = batch["category"]
            filenames = batch["filename"]

            recons = model(images)
            smoothed_err = calibrator.compute_smoothed_error_map(images, recons)
            raw_err = calibrator.compute_raw_error_map(images, recons).squeeze(1)

            # Compute both Top-K score and global mean score
            topk_scores = calibrator.compute_image_score(smoothed_err)
            mean_scores = smoothed_err.view(images.size(0), -1).mean(dim=1)

            for b in range(images.size(0)):
                lbl = labels[b].item()
                cat = categories[b]
                fn = filenames[b]
                topk_s = topk_scores[b].item()
                mean_s = mean_scores[b].item()

                gt_image_labels.append(lbl)
                pred_image_scores_topk.append(topk_s)
                pred_image_scores_mean.append(mean_s)

                mask_b = masks[b, 0].cpu().numpy().astype(bool)
                err_b = smoothed_err[b].cpu().numpy()

                gt_masks_flat.append(mask_b.flatten())
                pred_pixel_scores_flat.append(err_b.flatten())

                category_records.append({
                    "filename": fn,
                    "category": cat,
                    "true_label": lbl,
                    "topk_score": topk_s,
                    "mean_score": mean_s,
                    "pred_label": int(topk_s > calibrator.image_threshold)
                })

    y_true_pixels = np.concatenate(gt_masks_flat)
    y_score_pixels = np.concatenate(pred_pixel_scores_flat)

    y_true_images = np.array(gt_image_labels)
    y_score_topk = np.array(pred_image_scores_topk)
    y_score_mean = np.array(pred_image_scores_mean)

    # 1. Image-Level AUROC
    image_auroc_calibrated = float(roc_auc_score(y_true_images, y_score_topk))
    image_auroc_uncalibrated = float(roc_auc_score(y_true_images, y_score_mean))

    # Binary Image Metrics at calibrated threshold
    y_pred_images = (y_score_topk > calibrator.image_threshold).astype(int)
    img_prec, img_rec, img_f1, _ = precision_recall_fscore_support(
        y_true_images, y_pred_images, average="binary", zero_division=0
    )

    # 2. Pixel-Level AUROC
    pixel_auroc = float(roc_auc_score(y_true_pixels, y_score_pixels))

    # Binary Pixel Metrics at calibrated threshold
    y_pred_pixels = (y_score_pixels > calibrator.pixel_threshold).astype(bool)
    pix_prec, pix_rec, pix_f1, _ = precision_recall_fscore_support(
        y_true_pixels, y_pred_pixels, average="binary", zero_division=0
    )
    pix_iou = float(jaccard_score(y_true_pixels, y_pred_pixels, zero_division=0))

    # 3. Regional Error Breakdown
    breakdown = calibrator.compute_regional_error_breakdown(model, test_loader)

    # 4. Category breakdown table
    df_cat = pd.DataFrame(category_records)
    category_summary = {}
    for cat_name, group in df_cat.groupby("category"):
        acc = (group["true_label"] == group["pred_label"]).mean()
        category_summary[cat_name] = {
            "total_samples": len(group),
            "accuracy": float(acc),
            "mean_topk_score": float(group["topk_score"].mean())
        }

    results = {
        "Image AUROC (Calibrated Top-K)": image_auroc_calibrated,
        "Image AUROC (Uncalibrated Mean)": image_auroc_uncalibrated,
        "Image Precision": float(img_prec),
        "Image Recall": float(img_rec),
        "Image F1": float(img_f1),
        "Pixel AUROC": pixel_auroc,
        "Pixel Precision": float(pix_prec),
        "Pixel Recall": float(pix_rec),
        "Pixel F1": float(pix_f1),
        "Pixel IoU": pix_iou,
        "Defect Region Error": breakdown["defect_error"],
        "Outside Region Error": breakdown["outside_error"],
        "Normal Image Error": breakdown["normal_image_error"],
        "Defect/Outside Ratio": breakdown["defect_outside_ratio"],
        "Category Summary": category_summary
    }

    return results


def print_comparison_table(results_dict: Dict[str, Dict[str, Any]]):
    """
    Prints a formatted Markdown table comparing multiple model iterations.
    """
    headers = [
        "Metric",
        *results_dict.keys()
    ]
    metrics = [
        "Image AUROC (Calibrated Top-K)",
        "Image AUROC (Uncalibrated Mean)",
        "Image Precision",
        "Image Recall",
        "Image F1",
        "Pixel AUROC",
        "Pixel Precision",
        "Pixel Recall",
        "Pixel F1",
        "Pixel IoU",
        "Defect/Outside Ratio"
    ]

    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION SUMMARY: MVTEC AD")
    print("=" * 80)
    print(f"{'Metric':<35} | " + " | ".join([f"{k:<18}" for k in results_dict.keys()]))
    print("-" * 80)

    for m in metrics:
        vals = []
        for model_name, res in results_dict.items():
            val = res.get(m, 0.0)
            vals.append(f"{val:.5f}" if isinstance(val, (int, float)) else str(val))
        print(f"{m:<35} | " + " | ".join([f"{v:<18}" for v in vals]))
    print("=" * 80 + "\n")
