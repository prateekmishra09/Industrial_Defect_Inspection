# Industrial Surface Defect Inspection Platform (MVTec AD)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD-blue.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)](https://colab.research.google.com/)

Deep Learning-based automated visual inspection platform for manufacturing surface defect detection, precise pixel localization, and defect categorization across MVTec AD categories.

---

## 🔍 How It Works

The project applies unsupervised visual anomaly detection to all 15 MVTec AD object and texture
categories. A separate reconstruction model is trained for each category using only its normal
`train/good` images. Normal training images are split into training and validation sets; defective
test images and their ground-truth masks are reserved for evaluation.

At inference, the system compares an input image with its reconstruction, smooths the pixel-wise
error map, and uses top-k pooling for an image-level anomaly score. Thresholds are calibrated from
normal validation images. The dashboard can select a category, load its checkpoint, and display
the reconstruction, anomaly heatmap, and localized regions.

---

## 🏗️ Repository Architecture

```
Industrial_Defect_Inspection/
├── app/
│   └── app.py                     # Streamlit / Web Industrial Inspection Dashboard
├── configs/
│   └── config.yaml                # Dataset paths, model hyperparams & thresholds
├── data/                          # Local MVTec AD data (ignored by Git)
├── notebooks/
│   └── 01_mvtec_metal_nut_inspection.ipynb  # Metal Nut-specific Colab walkthrough
├── scripts/
│   ├── train_all_categories.py    # Train/evaluate one model per MVTec category
│   └── run_inspection.py          # Training workflow compatibility entry point
├── src/
│   ├── fundamentals/
│   │   ├── dataset.py             # Category-aware MVTec AD split loader
│   │   └── losses.py              # Differentiable SSIM, Composite L1+SSIM, RoI Masker
│   ├── detection/
│   │   └── models.py              # BasicAutoencoderV2, SkipAutoencoderV2, GatedSkipAutoencoder
│   ├── evaluation/
│   │   ├── calibrator.py          # Validation-based threshold calibrator & Top-K scorer
│   │   ├── metrics.py             # Image/Pixel AUROC, F1, IoU, regional error breakdown
│   │   └── visualizer.py          # Standardized 5-panel canvas with locked colorbars
│   ├── segmentation/
│   │   └── post_processing.py     # Morphological filtering & connected component extraction
│   ├── classification/
│   │   └── defect_classifier.py   # Defect category classifier (bent, color, flip, scratch)
│   ├── advanced/
│   │   ├── vae.py                 # Variational Autoencoder (VAE V2) for latent modeling
│   │   └── defect_synthesizer.py  # Procedural defect synthesizer (paired image + mask)
│   └── pipeline.py                # Operational Industrial Inspection Pipeline
├── tests/
│   └── test_pipeline.py           # Unit & integration test suite
├── requirements.txt               # Dependencies
└── README.md
```

---

## 📊 Evaluation Status

The repository includes per-category test evaluation and reporting. Full 15-category training and
evaluation have not yet been run, so no aggregate MVTec AD results are claimed here. The included
Colab notebook is a legacy Metal Nut-specific walkthrough; use the training script below for the
full dataset.

---

## 🏋️ Train on the Full MVTec AD Dataset

The project supports the 15 MVTec AD categories. The full dataset is stored under
`data\mvtec_anomaly_detection`; change `dataset.root_dir` in
[`configs/config.yaml`](configs/config.yaml) or pass `--data-root` if your dataset is elsewhere.

Train a separate Skip Autoencoder V2 for every discovered category:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m scripts.train_all_categories
```

Training runs for 80 epochs per category by default. Each model trains only on that category's
`train/good` images. Twenty percent of those normal images are held out for validation and
calibration; defective `test` images and their `ground_truth` masks are reserved for evaluation.
Checkpoints and calibrated thresholds are saved under
`checkpoints\<category>\skip_v2_trained.pth`.

Useful options:

```powershell
py -3.11 -m scripts.train_all_categories --category bottle
py -3.11 -m scripts.train_all_categories --epochs 80 --data-root "C:\path\to\mvtec_anomaly_detection"
```

The previous Metal Nut runner remains available as an entry point and now delegates to the
category-wise training workflow:

```powershell
py -3.11 -m scripts.run_inspection
```

## 🚀 Quickstart: Google Colab Execution

The notebook below is a Metal Nut-specific walkthrough, not the full 15-category workflow. Use
the local training command above for all categories.

1. Open Google Colab and clone the repository:
   ```bash
   !git clone https://github.com/prateekmishra09/Industrial_Defect_Inspection.git
   %cd Industrial_Defect_Inspection
   !pip install -r requirements.txt
   ```
2. Open [`notebooks/01_mvtec_metal_nut_inspection.ipynb`](notebooks/01_mvtec_metal_nut_inspection.ipynb) in Colab.
3. Run all cells:
   - Downloads/loads the 176/44/115 MVTec AD Metal Nut split.
   - Loads your saved checkpoints (or trains Basic V2 and Skip V2).
   - Calibrates $\tau_{\text{pixel}}$ and $\tau_{\text{image}}$ strictly on the 44 normal validation images.
   - Computes all quantitative metrics and displays the side-by-side 5-panel inspection gallery.
   - Runs synthetic anomaly generation with the VAE module.

---

## 🖥️ Running the Inspection Web Dashboard

Launch the interactive production dashboard:
```bash
streamlit run app/app.py
```
- Select a category; the app loads its checkpoint from `checkpoints/<category>/skip_v2_trained.pth`.
- Upload an uninspected image from the selected category or choose a preloaded test sample.
- View real-time **PASS / FAIL** verdicts.
- Inspect localized bounding boxes, defect area statistics, and calibrated heatmaps.

---

## 📜 Citation & License
This project is licensed under the MIT License. Developed for industrial surface defect inspection using the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).
