import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.detection.models import SkipConnectionAutoencoderV2
from src.fundamentals.dataset import discover_categories
from src.evaluation.calibrator import ThresholdCalibrator
from src.pipeline import IndustrialInspectionPipeline


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="Industrial Defect Inspection Platform",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Industrial Surface Defect Inspection Platform")
    st.subheader("MVTec AD Category-Specific Anomaly Detection & Localization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    DATASET_DIR = config["dataset"]["root_dir"]
    if not os.path.isabs(DATASET_DIR):
        DATASET_DIR = os.path.join(project_root, DATASET_DIR)
    try:
        categories = discover_categories(DATASET_DIR)
    except (FileNotFoundError, OSError) as exc:
        st.error(f"Cannot load MVTec AD dataset: {exc}")
        return

    st.sidebar.header("1. Category")
    category = st.sidebar.selectbox("MVTec AD category", categories)
    category_root = (
        DATASET_DIR
        if os.path.basename(os.path.normpath(DATASET_DIR)) == category
        else os.path.join(DATASET_DIR, category)
    )
    model_config = config.get("models", {}).get("skip_v2", {})
    model = SkipConnectionAutoencoderV2(
        in_channels=int(config["dataset"].get("channels", 3)),
        base_channels=int(model_config.get("base_channels", 32)),
        latent_dim=int(model_config.get("latent_dim", 128))
    ).to(device)

    st.sidebar.subheader("2. Checkpoint")
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "./checkpoints")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, category, "skip_v2_trained.pth")
    if category == "metal_nut" and not os.path.isfile(checkpoint_path):
        legacy_checkpoint = os.path.join(checkpoint_dir, "skip_v2_trained.pth")
        if os.path.isfile(legacy_checkpoint):
            checkpoint_path = legacy_checkpoint

    checkpoint_file = st.sidebar.file_uploader(
        "Or upload trained weights (.pth / .pt)",
        type=["pth", "pt"],
        key=f"{category}_checkpoint_upload"
    )
    is_model_trained = False
    checkpoint_data = None
    checkpoint_source = checkpoint_file if checkpoint_file is not None else checkpoint_path
    if checkpoint_file is not None or os.path.isfile(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_source, map_location=device, weights_only=True)
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            checkpoint_category = checkpoint_data.get("category")
            if checkpoint_category and checkpoint_category != category:
                st.sidebar.error(
                    f"This checkpoint is for `{checkpoint_category}`, not `{category}`."
                )
                st.stop()
            model.load_state_dict(checkpoint_data["model_state_dict"])
            is_model_trained = True
        else:
            model.load_state_dict(checkpoint_data)
            is_model_trained = True
        st.sidebar.success(f"Loaded checkpoint for `{category}`")
    else:
        st.sidebar.warning(f"No trained checkpoint found for `{category}`.")
        st.sidebar.info("Run `py -3.11 -m scripts.train_all_categories` to train all categories.")

    calibration_stats = (
        checkpoint_data.get("calibration", {})
        if isinstance(checkpoint_data, dict)
        else {}
    )
    calibration_config = config.get("calibration", {})
    calibrator = ThresholdCalibrator(
        kernel_size=int(calibration_config.get("gaussian_kernel_size", 11)),
        sigma=float(calibration_config.get("gaussian_sigma", 4.0)),
        topk_ratio=float(calibration_config.get("topk_ratio", 0.005)),
        device=device
    )

    st.sidebar.subheader("3. Calibration Thresholds")
    default_pixel_th = float(calibration_stats.get("pixel_threshold", 0.025 if is_model_trained else 0.42))
    default_image_th = float(calibration_stats.get("image_threshold", 0.022 if is_model_trained else 0.41))
    default_vmax = float(calibration_stats.get("heatmap_vmax", 0.06 if is_model_trained else 0.5))

    pixel_threshold = st.sidebar.number_input(
        f"Pixel Anomaly Threshold (τ_pixel) - {category}",
        key=f"{category}_pixel_threshold",
        value=default_pixel_th, format="%.5f", step=0.001
    )
    image_threshold = st.sidebar.number_input(
        f"Image Anomaly Threshold (τ_image) - {category}",
        key=f"{category}_image_threshold",
        value=default_image_th, format="%.5f", step=0.001
    )
    vmax_val = st.sidebar.number_input(
        f"Heatmap Scale Max (vmax) - {category}",
        key=f"{category}_heatmap_vmax",
        value=default_vmax, format="%.5f", step=0.005
    )

    calibrator.pixel_threshold = pixel_threshold
    calibrator.image_threshold = image_threshold
    calibrator.vmax = vmax_val

    pipeline = IndustrialInspectionPipeline(
        model=model,
        calibrator=calibrator,
        device=device,
        classify_defects=(category == "metal_nut")
    )

    # Input Image Selection
    st.subheader("Select Part for Inspection")
    input_method = st.radio("Input Source", ["Preloaded Test Samples", "Upload Image"], horizontal=True)

    input_image = None
    sample_info = ""

    if input_method == "Preloaded Test Samples":
        test_samples = []
        test_dir = os.path.join(category_root, "test")
        for defect_type in sorted(os.listdir(test_dir)):
            cat_dir = os.path.join(test_dir, defect_type)
            if os.path.isdir(cat_dir):
                files = sorted(
                    path for path in glob.glob(os.path.join(cat_dir, "*"))
                    if os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                )
                for path in files[:4]:
                    test_samples.append((f"{defect_type.upper()}: {os.path.basename(path)}", path))

        if test_samples:
            sample_labels = [s[0] for s in test_samples]
            chosen_label = st.selectbox(f"Choose a {category} test sample:", sample_labels)
            chosen_path = dict(test_samples)[chosen_label]
            input_image = Image.open(chosen_path).convert("RGB")
            sample_info = chosen_label
        else:
            st.info(f"No test samples found for {category} in {test_dir}")
    else:
        uploaded_file = st.file_uploader(
            f"Upload {category.replace('_', ' ').title()} image",
            type=["png", "jpg", "jpeg", "bmp"]
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            sample_info = uploaded_file.name

    if input_image is None:
        st.info("👈 Please select or upload an image to begin inspection.")
        return

    # Run Pipeline
    with st.spinner("Running calibrated inspection pipeline..."):
        result = pipeline.inspect(input_image)

    # Model status notice
    if not is_model_trained:
        st.warning(
            f"No trained checkpoint is loaded for `{category}`. "
            "Inspection results from the randomly initialized model are not meaningful."
        )

    # Verdict Banner
    st.markdown("---")
    col_verdict, col_meta = st.columns([2, 1])
    with col_verdict:
        if result["status"] == "PASS":
            st.success(
                f"### Verdict: PASS (Normal Part)  \n"
                f"**Anomaly Score**: `{result['anomaly_score']:.6f}` (Threshold: `{result['image_threshold']:.6f}`)  \n"
                f"**Margin**: `{result['score_margin']:+.6f}`"
            )
        else:
            st.error(
                f"### Verdict: FAIL (Defective Part Detected)  \n"
                f"**Anomaly Score**: `{result['anomaly_score']:.6f}` (Threshold: `{result['image_threshold']:.6f}`)  \n"
                f"**Defect Type**: **{result['primary_defect_type'].upper()}** | **Defect Blobs**: {result['defect_count']}"
            )

    with col_meta:
        st.metric(
            label="Image Anomaly Score",
            value=f"{result['anomaly_score']:.5f}",
            delta=f"{result['score_margin']:+.5f} vs threshold",
            delta_color="inverse"
        )
        st.metric(label="Detected Defect Blobs", value=result["defect_count"])

    # 4-Panel Standardized Visual Breakdown
    st.markdown("### Visual Inspection Breakdown")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # 1. Input
    axes[0].imshow(input_image)
    axes[0].set_title(f"Input: {sample_info}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Reconstruction
    axes[1].imshow(np.clip(result["reconstructed_image"], 0, 1))
    axes[1].set_title("Reconstruction (Skip Autoencoder V2)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Calibrated Heatmap (Dynamic / Fixed VMAX)
    current_vmax = max(calibrator.vmax, result["smoothed_error_map"].max() * 0.95 + 1e-4)
    im = axes[2].imshow(result["smoothed_error_map"], cmap="inferno", vmin=0.0, vmax=current_vmax)
    axes[2].set_title(f"Calibrated Heatmap [0, {current_vmax:.4f}]", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Defect Localization
    axes[3].imshow(result["annotated_image"])
    axes[3].set_title(f"Localization (τ={calibrator.pixel_threshold:.4f})", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
