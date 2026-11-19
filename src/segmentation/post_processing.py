import cv2
import numpy as np
from typing import List, Dict, Tuple


class DefectSegmenter:
    """
    Industrial Post-Processing and Defect Segmentation.
    Applies morphological filtering, connected component analysis,
    and bounding-box extraction to filter out false alarms.
    """
    def __init__(self, min_area_pixels: int = 20, morph_kernel_size: int = 3):
        self.min_area_pixels = min_area_pixels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def process_binary_map(self, raw_binary_mask: np.ndarray) -> Dict:
        """
        raw_binary_mask: 2D boolean or uint8 array (H, W) where True/1 indicates anomaly.
        Returns:
          - cleaned_mask: uint8 (H, W) with false positives suppressed
          - defect_count: number of discrete defects
          - bounding_boxes: list of (x, y, w, h)
          - defect_stats: list of dicts with area, centroid, bounding box
        """
        mask_uint8 = (raw_binary_mask.astype(np.uint8)) * 255

        # 1. Morphological Opening to remove isolated noise
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.morph_kernel)

        # 2. Morphological Closing to bridge micro-fractures in continuous defects
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)

        # 3. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        cleaned_mask = np.zeros_like(closed)
        valid_defects: List[Dict] = []
        bounding_boxes: List[Tuple[int, int, int, int]] = []

        # Label 0 is background; iterate through 1 to num_labels-1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area_pixels:
                cleaned_mask[labels == i] = 255
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[i]

                bbox = (x, y, w, h)
                bounding_boxes.append(bbox)
                valid_defects.append({
                    "defect_id": len(valid_defects) + 1,
                    "area_pixels": int(area),
                    "bounding_box": bbox,
                    "centroid": (float(cx), float(cy))
                })

        return {
            "cleaned_mask": cleaned_mask > 0,
            "defect_count": len(valid_defects),
            "bounding_boxes": bounding_boxes,
            "defect_details": valid_defects,
            "total_defect_area": sum(d["area_pixels"] for d in valid_defects)
        }
