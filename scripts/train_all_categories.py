import argparse
import os
from typing import Dict, Any, List, Optional

import torch
import yaml

from src.detection.models import SkipConnectionAutoencoderV2
from src.evaluation.calibrator import ThresholdCalibrator
from src.evaluation.metrics import compute_comprehensive_metrics
from src.fundamentals.dataset import discover_categories, get_dataloaders
from src.fundamentals.losses import CompositeLoss


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def train_category(
    category: str,
    data_root: str,
    checkpoint_dir: str,
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    calibration_config: Dict[str, Any],
    loss_alpha: float,
    device: str,
    evaluate: bool = True
) -> str:
    seed = int(dataset_config.get("random_seed", 42))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    image_size = tuple(dataset_config.get("image_size", [256, 256]))
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=data_root,
        category=category,
        batch_size=int(training_config.get("batch_size", 16)),
        image_size=image_size,
        num_workers=int(training_config.get("num_workers", 2)),
        seed=seed
    )
    model = SkipConnectionAutoencoderV2(
        in_channels=int(dataset_config.get("channels", 3)),
        base_channels=int(model_config.get("base_channels", 32)),
        latent_dim=int(model_config.get("latent_dim", 128))
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-5))
    )
    criterion = CompositeLoss(alpha=float(loss_alpha)).to(device)

    epochs = int(training_config.get("epochs", 80))
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                val_loss += criterion(model(images), images).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(
            f"[{category}] Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

    calibrator = ThresholdCalibrator(
        kernel_size=int(calibration_config.get("gaussian_kernel_size", 11)),
        sigma=float(calibration_config.get("gaussian_sigma", 4.0)),
        topk_ratio=float(calibration_config.get("topk_ratio", 0.005)),
        device=device
    )
    calibration_stats = calibrator.calibrate_from_val_loader(
        model,
        val_loader,
        pixel_fp_rate=float(calibration_config.get("pixel_false_positive_rate", 0.005)),
        image_fp_rate=float(calibration_config.get("image_false_positive_rate", 0.02))
    )

    category_checkpoint_dir = os.path.join(checkpoint_dir, category)
    os.makedirs(category_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(category_checkpoint_dir, "skip_v2_trained.pth")
    torch.save({
        "category": category,
        "architecture": "SkipConnectionAutoencoderV2",
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "calibration": calibration_stats
    }, checkpoint_path)
    print(f"[{category}] Saved checkpoint: {checkpoint_path}")

    if evaluate:
        metrics = compute_comprehensive_metrics(model, test_loader, calibrator, device=device)
        print(f"[{category}] Test metrics: {metrics}")

    return checkpoint_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a separate MVTec AD model for each category."
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_default = os.path.join(project_root, "configs", "config.yaml")
    parser.add_argument("--config", default=config_default, help="Path to YAML configuration.")
    parser.add_argument("--data-root", help="Override the MVTec AD dataset root.")
    parser.add_argument("--checkpoint-dir", help="Override the checkpoint output directory.")
    parser.add_argument("--category", help="Train only this category; default: all discovered categories.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip test-set metrics.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    training_config = dict(config.get("training", {}))
    model_config = config.get("models", {}).get("skip_v2", {})
    calibration_config = config.get("calibration", {})
    loss_alpha = config.get("loss", {}).get("alpha_l1", 0.5)

    data_root = args.data_root or dataset_config.get("root_dir")
    if not data_root:
        raise ValueError("Set dataset.root_dir in the config or pass --data-root.")
    if not os.path.isabs(data_root):
        data_root = os.path.join(project_root, data_root)
    checkpoint_dir = args.checkpoint_dir or training_config.get("checkpoint_dir", "./checkpoints")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    if args.epochs is not None:
        training_config["epochs"] = args.epochs

    available_categories = discover_categories(data_root)
    categories = [args.category] if args.category else available_categories
    unknown_categories = sorted(set(categories) - set(available_categories))
    if unknown_categories:
        raise ValueError(
            f"Unknown category/categories {unknown_categories}. "
            f"Available categories: {available_categories}"
        )
    if int(training_config.get("epochs", 80)) < 1:
        raise ValueError("The training epoch count must be at least one.")

    device = config.get("project", {}).get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Dataset root: {data_root}")
    print(f"Device: {device}")
    print(f"Categories: {', '.join(categories)}")

    for category in categories:
        train_category(
            category=category,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            training_config=training_config,
            dataset_config=dataset_config,
            model_config=model_config,
            calibration_config=calibration_config,
            loss_alpha=loss_alpha,
            device=device,
            evaluate=not args.skip_evaluation
        )


if __name__ == "__main__":
    main()
