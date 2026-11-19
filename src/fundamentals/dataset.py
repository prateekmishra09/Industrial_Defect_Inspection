import os
import glob
from typing import Tuple, List, Optional, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


MVTEC_METAL_NUT_URL = (
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938145-1629952514/metal_nut.tar.xz"
)

DEFECT_CATEGORIES = ["bent", "color", "flip", "scratch"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def discover_categories(root_dir: str) -> List[str]:
    """Return dataset folders that contain an MVTec ``train/good`` split."""
    if os.path.isdir(os.path.join(root_dir, "train", "good")):
        return [os.path.basename(os.path.normpath(root_dir))]
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"MVTec dataset root not found: {root_dir}")

    categories = []
    for name in sorted(os.listdir(root_dir)):
        category_dir = os.path.join(root_dir, name)
        if os.path.isdir(os.path.join(category_dir, "train", "good")):
            categories.append(name)
    if not categories:
        raise FileNotFoundError(
            f"No MVTec categories with a train/good directory found in {root_dir}"
        )
    return categories


class MVTecADDataset(Dataset):
    """
    Loads one MVTec AD category, training and validating only on normal images.
    """
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.category = category
        normalized_root = os.path.normpath(root_dir)
        self.category_dir = (
            root_dir if os.path.basename(normalized_root) == category
            else os.path.join(root_dir, category)
        )
        self.split = split.lower()
        self.image_size = image_size
        self.seed = seed

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        if target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.target_transform = target_transform

        self.samples = self._load_samples()

    @staticmethod
    def _image_files(directory: str) -> List[str]:
        if not os.path.isdir(directory):
            return []
        return sorted(
            path
            for path in glob.glob(os.path.join(directory, "*"))
            if os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS)
        )

    def _load_samples(self) -> List[Dict]:
        samples = []

        if self.split in ["train", "val"]:
            train_good_dir = os.path.join(self.category_dir, "train", "good")
            if not os.path.isdir(train_good_dir):
                raise FileNotFoundError(
                    f"Train directory not found at {train_good_dir}. "
                    f"Expected category {self.category!r} in {self.root_dir}."
                )

            all_train_files = self._image_files(train_good_dir)
            if len(all_train_files) == 0:
                raise RuntimeError(f"No supported images found in {train_good_dir}")
            if len(all_train_files) < 2:
                raise RuntimeError(
                    f"At least two normal training images are required to create "
                    f"train/validation splits for {self.category!r}."
                )

            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(all_train_files))
            rng.shuffle(indices)

            split_boundary = min(max(int(len(all_train_files) * 0.8), 1), len(all_train_files) - 1)
            if self.split == "train":
                selected_indices = indices[:split_boundary]
            else:
                selected_indices = indices[split_boundary:]

            for idx in selected_indices:
                fpath = all_train_files[idx]
                samples.append({
                    "image_path": fpath,
                    "mask_path": None,
                    "label": 0,
                    "category": "good",
                    "filename": os.path.basename(fpath)
                })

        elif self.split == "test":
            test_dir = os.path.join(self.category_dir, "test")
            gt_dir = os.path.join(self.category_dir, "ground_truth")

            if not os.path.isdir(test_dir):
                raise FileNotFoundError(
                    f"Test directory not found at {test_dir}. "
                    f"Expected category {self.category!r} in {self.root_dir}."
                )

            good_test_dir = os.path.join(test_dir, "good")
            for fpath in self._image_files(good_test_dir):
                samples.append({
                    "image_path": fpath,
                    "mask_path": None,
                    "label": 0,
                    "category": "good",
                    "filename": os.path.basename(fpath)
                })

            defect_categories = sorted(
                name for name in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, name)) and name != "good"
            )
            for defect in defect_categories:
                defect_test_dir = os.path.join(test_dir, defect)
                defect_gt_dir = os.path.join(gt_dir, defect)

                for fpath in self._image_files(defect_test_dir):
                    fname = os.path.basename(fpath)
                    base_id = os.path.splitext(fname)[0]
                    mask_path = None
                    if os.path.isdir(defect_gt_dir):
                        mask_candidates = sorted(
                            glob.glob(os.path.join(defect_gt_dir, f"{base_id}_mask.*"))
                        )
                        if not mask_candidates:
                            mask_candidates = sorted(
                                glob.glob(os.path.join(defect_gt_dir, f"{base_id}.*"))
                            )
                        if mask_candidates:
                            mask_path = mask_candidates[0]

                    samples.append({
                        "image_path": fpath,
                        "mask_path": mask_path,
                        "label": 1,
                        "category": defect,
                        "filename": fname
                    })
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_meta = self.samples[idx]

        # Load RGB image
        img = Image.open(sample_meta["image_path"]).convert("RGB")
        img_tensor = self.transform(img)

        # Load or generate binary mask (1, H, W)
        if sample_meta["mask_path"] is not None and os.path.exists(sample_meta["mask_path"]):
            mask = Image.open(sample_meta["mask_path"]).convert("L")
            mask_tensor = self.target_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Normal image -> all-zero mask
            mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample_meta["label"], dtype=torch.long),
            "category": sample_meta["category"],
            "filename": sample_meta["filename"]
        }


class MVTecMetalNutDataset(MVTecADDataset):
    """Backward-compatible Metal Nut dataset loader."""
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            category="metal_nut",
            split=split,
            image_size=image_size,
            seed=seed,
            transform=transform,
            target_transform=target_transform
        )


def get_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 2,
    seed: int = 42,
    category: str = "metal_nut"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates category-specific train, validation, and test DataLoaders.
    """
    train_dataset = MVTecADDataset(root_dir, category, split="train", image_size=image_size, seed=seed)
    val_dataset = MVTecADDataset(root_dir, category, split="val", image_size=image_size, seed=seed)
    test_dataset = MVTecADDataset(root_dir, category, split="test", image_size=image_size, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_synthetic_mvtec_directory(target_dir: str, image_size: Tuple[int, int] = (256, 256)):
    """
    Generates a synthetic MVTec AD Metal Nut mock directory for local tests and debugging
    when the complete 2GB dataset has not yet been downloaded.
    Matches exact counts: 220 train good, 22 test good, 93 test defective.
    """
    import cv2
    os.makedirs(os.path.join(target_dir, "train", "good"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", "good"), exist_ok=True)

    for defect in DEFECT_CATEGORIES:
        os.makedirs(os.path.join(target_dir, "test", defect), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "ground_truth", defect), exist_ok=True)

    def draw_metal_nut(has_defect: Optional[str] = None):
        h, w = image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Outer hexagonal body
        center = (w // 2, h // 2)
        radius = int(w * 0.38)
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        pts = np.array([[int(center[0] + radius * np.cos(a)), 
                         int(center[1] + radius * np.sin(a))] for a in angles], np.int32)
        cv2.fillPoly(img, [pts], (180, 185, 190))

        # Inner hole with threads
        inner_radius = int(w * 0.18)
        cv2.circle(img, center, inner_radius, (30, 30, 35), -1)
        for r in range(inner_radius - 8, inner_radius + 4, 3):
            cv2.circle(img, center, r, (70, 75, 80), 1)

        # Draw defects
        if has_defect == "scratch":
            pt1 = (center[0] - 30, center[1] - 40)
            pt2 = (center[0] + 40, center[1] - 20)
            cv2.line(img, pt1, pt2, (40, 40, 40), 2)
            cv2.line(mask, pt1, pt2, 255, 6)
        elif has_defect == "bent":
            cv2.ellipse(img, (center[0] + radius - 15, center[1]), (25, 15), 30, 0, 360, (60, 60, 60), -1)
            cv2.ellipse(mask, (center[0] + radius - 15, center[1]), (25, 15), 30, 0, 360, 255, -1)
        elif has_defect == "color":
            cv2.circle(img, (center[0] - 25, center[1] + 25), 18, (120, 80, 40), -1)
            cv2.circle(mask, (center[0] - 25, center[1] + 25), 18, 255, -1)
        elif has_defect == "flip":
            cv2.circle(img, (center[0], center[1] - inner_radius - 10), 12, (90, 95, 100), -1)
            cv2.circle(mask, (center[0], center[1] - inner_radius - 10), 12, 255, -1)

        return img, mask

    # 1. 220 train normal images
    for i in range(220):
        img, _ = draw_metal_nut()
        Image.fromarray(img).save(os.path.join(target_dir, "train", "good", f"{i:03d}.png"))

    # 2. 22 test normal images
    for i in range(22):
        img, _ = draw_metal_nut()
        Image.fromarray(img).save(os.path.join(target_dir, "test", "good", f"{i:03d}.png"))

    # 3. 93 defective images across 4 categories (approx 23-24 each)
    counts = {"bent": 25, "color": 22, "flip": 23, "scratch": 23}
    for defect, cnt in counts.items():
        for i in range(cnt):
            img, mask = draw_metal_nut(has_defect=defect)
            Image.fromarray(img).save(os.path.join(target_dir, "test", defect, f"{i:03d}.png"))
            Image.fromarray(mask).save(os.path.join(target_dir, "ground_truth", defect, f"{i:03d}_mask.png"))

    print(f"Synthetic MVTec AD Metal Nut dataset generated at: {target_dir}")
