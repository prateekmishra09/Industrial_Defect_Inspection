from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.evaluation.calibrator import ThresholdCalibrator
from src.segmentation.post_processing import DefectSegmenter
from src.classification.defect_classifier import HeuristicDefectClassifier


class IndustrialInspectionPipeline:
    """
    End-to-End Industrial Surface Defect Inspection Platform.
    Integrates model inference, calibrated thresholding, defect segmentation,
    and defect classification into an operational inspection engine.
    """
    def __init__(
        self,
        model: nn.Module,
        calibrator: ThresholdCalibrator,
        segmenter: Optional[DefectSegmenter] = None,
        image_size: Tuple[int, int] = (256, 256),
        device: str = "cpu",
        classify_defects: bool = True
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.calibrator = calibrator
        self.segmenter = segmenter or DefectSegmenter(min_area_pixels=20)
        self.image_size = image_size
        self.classify_defects = classify_defects

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def inspect(self, input_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Runs real-time inspection on a single part image.
        Returns Pass/Fail status, confidence score, bounding boxes, and annotated image.
        """
        # 1. Image loading & formatting
        if isinstance(input_image, str):
            pil_img = Image.open(input_image).convert("RGB")
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_image, Image.Image):
            pil_img = input_image.convert("RGB")
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_image, np.ndarray):
            if input_image.max() > 1.0:
                input_image = input_image / 255.0
            pil_img = Image.fromarray((input_image * 255).astype(np.uint8))
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_image, torch.Tensor):
            if input_image.ndim == 3:
                img_tensor = input_image.unsqueeze(0).to(self.device)
            else:
                img_tensor = input_image.to(self.device)
            pil_img = None
        else:
            raise TypeError("Unsupported image input type.")

        # 2. Forward inference & Anomaly scoring
        with torch.no_grad():
            reconstruction = self.model(img_tensor)
            smoothed_err = self.calibrator.compute_smoothed_error_map(img_tensor, reconstruction)
            anomaly_score = self.calibrator.compute_image_score(smoothed_err)[0].item()

        # 3. Decision Logic
        is_defective = anomaly_score > self.calibrator.image_threshold
        status = "FAIL" if is_defective else "PASS"

        # 4. Defect Localization & Segmentation
        err_np = smoothed_err[0].cpu().numpy()
        raw_binary_mask = err_np > self.calibrator.pixel_threshold
        seg_result = self.segmenter.process_binary_map(raw_binary_mask)

        cleaned_mask = seg_result["cleaned_mask"]
        bounding_boxes = seg_result["bounding_boxes"]
        defect_count = seg_result["defect_count"]

        # 5. Defect Classification (if defective)
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        defect_categories = []
        if is_defective and defect_count > 0:
            if self.classify_defects:
                for bbox in bounding_boxes:
                    cat = HeuristicDefectClassifier.classify_defect(img_np, cleaned_mask, bbox)
                    defect_categories.append(cat)
            else:
                defect_categories = ["anomaly"] * defect_count
            primary_defect = defect_categories[0]
        elif is_defective:
            primary_defect = "unlocalized_anomaly"
        else:
            primary_defect = "none"

        # 6. Generate Annotated Production Image
        annotated = (np.clip(img_np.copy(), 0, 1) * 255).astype(np.uint8).copy()
        for bbox, cat in zip(bounding_boxes, defect_categories):
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                annotated, cat.upper(), (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
            )

        # Margin relative to threshold
        margin = anomaly_score - self.calibrator.image_threshold

        return {
            "status": status,
            "anomaly_score": anomaly_score,
            "image_threshold": self.calibrator.image_threshold,
            "score_margin": margin,
            "is_defective": is_defective,
            "defect_count": defect_count,
            "primary_defect_type": primary_defect,
            "all_defect_types": defect_categories,
            "bounding_boxes": bounding_boxes,
            "defect_details": seg_result["defect_details"],
            "cleaned_mask": cleaned_mask,
            "smoothed_error_map": err_np,
            "annotated_image": annotated,
            "reconstructed_image": reconstruction[0].permute(1, 2, 0).cpu().numpy()
        }
