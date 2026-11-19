import os
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from src.detection.models import BasicAutoencoderV2, SkipConnectionAutoencoderV2, GatedSkipAutoencoder
from src.fundamentals.losses import CompositeLoss, ssim
from src.fundamentals.dataset import MVTecADDataset, discover_categories
from src.evaluation.calibrator import ThresholdCalibrator
from src.segmentation.post_processing import DefectSegmenter
from src.advanced.defect_synthesizer import IndustrialDefectSynthesizer
from src.pipeline import IndustrialInspectionPipeline


class TestDefectInspectionPipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 2
        self.img_size = (256, 256)
        self.dummy_images = torch.rand((self.batch_size, 3, *self.img_size), dtype=torch.float32)

    def test_basic_autoencoder_shape(self):
        model = BasicAutoencoderV2(in_channels=3, base_channels=16, latent_dim=64)
        out = model(self.dummy_images)
        self.assertEqual(out.shape, self.dummy_images.shape)
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_skip_autoencoder_shape(self):
        model = SkipConnectionAutoencoderV2(in_channels=3, base_channels=16, latent_dim=64)
        out = model(self.dummy_images)
        self.assertEqual(out.shape, self.dummy_images.shape)
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_gated_skip_autoencoder_shape(self):
        model = GatedSkipAutoencoder(in_channels=3, base_channels=16, latent_dim=64)
        out = model(self.dummy_images)
        self.assertEqual(out.shape, self.dummy_images.shape)

    def test_composite_loss(self):
        criterion = CompositeLoss(alpha=0.5)
        loss = criterion(self.dummy_images, self.dummy_images)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_calibrator_topk_ranking(self):
        calibrator = ThresholdCalibrator(kernel_size=5, sigma=1.5, topk_ratio=0.01, device=self.device)
        
        # Create normal image (flat low noise)
        normal_orig = torch.zeros((1, 3, 64, 64))
        normal_recon = torch.full((1, 3, 64, 64), 0.001)

        # Create defective image (mostly perfect, but localized high spike)
        defect_orig = torch.zeros((1, 3, 64, 64))
        defect_recon = torch.zeros((1, 3, 64, 64))
        defect_recon[0, :, 30:35, 30:35] = 0.5  # Localized defect

        # Compute smoothed maps
        err_norm = calibrator.compute_smoothed_error_map(normal_orig, normal_recon)
        err_def = calibrator.compute_smoothed_error_map(defect_orig, defect_recon)

        # Top-K pooling vs global mean
        score_norm_topk = calibrator.compute_image_score(err_norm).item()
        score_def_topk = calibrator.compute_image_score(err_def).item()

        # Defective sample must score significantly higher than normal under Top-K
        self.assertGreater(score_def_topk, score_norm_topk)

    def test_defect_segmenter(self):
        segmenter = DefectSegmenter(min_area_pixels=10)
        # 64x64 mask with 1 valid blob and 1 tiny noise pixel
        raw_mask = np.zeros((64, 64), dtype=bool)
        raw_mask[20:25, 20:25] = True  # 25 pixels -> valid defect
        raw_mask[5, 5] = True          # 1 pixel -> noise to filter

        res = segmenter.process_binary_map(raw_mask)
        self.assertEqual(res["defect_count"], 1)
        self.assertFalse(res["cleaned_mask"][5, 5])
        self.assertTrue(res["cleaned_mask"][22, 22])
        self.assertEqual(len(res["bounding_boxes"]), 1)

    def test_pipeline_uses_generic_label_without_metal_nut_classifier(self):
        calibrator = ThresholdCalibrator(kernel_size=1, sigma=1.0, device=self.device)
        calibrator.pixel_threshold = -1.0
        calibrator.image_threshold = -1.0
        pipeline = IndustrialInspectionPipeline(
            torch.nn.Identity(),
            calibrator,
            device=self.device,
            classify_defects=False
        )

        result = pipeline.inspect(torch.rand(3, 64, 64))

        self.assertEqual(result["primary_defect_type"], "anomaly")
        self.assertEqual(result["all_defect_types"], ["anomaly"])

    def test_defect_synthesizer(self):
        synthesizer = IndustrialDefectSynthesizer(seed=42)
        sample = torch.rand((3, 256, 256))
        for cat in ["scratch", "bent", "color", "flip"]:
            res = synthesizer.generate(sample, category=cat)
            self.assertEqual(res["synthetic_image"].shape, (3, 256, 256))
            self.assertEqual(res["synthetic_mask"].shape, (1, 256, 256))
            self.assertTrue(res["synthetic_mask"].sum().item() > 0)

    def test_mvtec_dataset_category_splits_and_masks(self):
        with tempfile.TemporaryDirectory() as root:
            category_root = os.path.join(root, "bottle")
            train_good = os.path.join(category_root, "train", "good")
            test_good = os.path.join(category_root, "test", "good")
            test_broken = os.path.join(category_root, "test", "broken")
            ground_truth = os.path.join(category_root, "ground_truth", "broken")
            for directory in (train_good, test_good, test_broken, ground_truth):
                os.makedirs(directory)

            image = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.full((16, 16), 255, dtype=np.uint8)
            for index in range(5):
                Image.fromarray(image).save(os.path.join(train_good, f"{index:03d}.png"))
            Image.fromarray(image).save(os.path.join(test_good, "000.png"))
            Image.fromarray(image).save(os.path.join(test_broken, "001.png"))
            Image.fromarray(mask).save(os.path.join(ground_truth, "001_mask.png"))

            self.assertEqual(discover_categories(root), ["bottle"])
            train = MVTecADDataset(root, "bottle", split="train", image_size=(16, 16))
            val = MVTecADDataset(root, "bottle", split="val", image_size=(16, 16))
            test = MVTecADDataset(root, "bottle", split="test", image_size=(16, 16))

            self.assertEqual(len(train), 4)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 2)
            defective_sample = next(sample for sample in test.samples if sample["label"] == 1)
            loaded_defect = test[test.samples.index(defective_sample)]
            self.assertEqual(loaded_defect["category"], "broken")
            self.assertEqual(loaded_defect["label"].item(), 1)
            self.assertEqual(loaded_defect["mask"].sum().item(), 16 * 16)


if __name__ == "__main__":
    unittest.main()
