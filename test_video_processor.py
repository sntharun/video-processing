import os
import unittest
import cv2
import numpy as np
from video_processor import (
    crop_center,
    calculate_exposure_stats,
    calculate_masked_laplacian,
    calculate_tenengrad,
    calculate_exposure_quality_multiplier,
    select_top_k_indices,
    calculate_frame_metrics,
)


class TestVideoProcessor(unittest.TestCase):
    def test_crop_center(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_center(frame, fraction=0.5)
        self.assertEqual(cropped.shape, (50, 50, 3))

        # Edge case: fraction <= 0 or >= 1.0 returns original
        self.assertEqual(crop_center(frame, fraction=1.0).shape, (100, 100, 3))
        self.assertEqual(crop_center(frame, fraction=0.0).shape, (100, 100, 3))

    def test_exposure_stats(self):
        img = np.full((100, 100), 50, dtype=np.uint8)
        img[:50, :50] = 250

        stats = calculate_exposure_stats(img)
        self.assertEqual(stats["overexposure_pct"], 25.0)
        self.assertTrue(90.0 < stats["underexposure_mean"] < 110.0)
        self.assertTrue(stats["contrast_std"] > 0)

    def test_glare_masked_laplacian(self):
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        masked_score, raw_score = calculate_masked_laplacian(uniform)
        self.assertEqual(masked_score, 0.0)
        self.assertEqual(raw_score, 0.0)

        glare_img = np.full((100, 100), 128, dtype=np.uint8)
        glare_img[40:60, 40:60] = 255

        masked_score, raw_score = calculate_masked_laplacian(glare_img)
        self.assertTrue(raw_score > masked_score)

    def test_tenengrad(self):
        # Step edge image produces strong gradient energy
        pattern = np.zeros((50, 50), dtype=np.uint8)
        pattern[:, 25:] = 255
        score = calculate_tenengrad(pattern)
        self.assertTrue(score > 1000.0)

    def test_exposure_quality_multiplier(self):
        self.assertEqual(calculate_exposure_quality_multiplier(120.0), 1.0)
        self.assertEqual(calculate_exposure_quality_multiplier(35.0), 0.5)
        self.assertEqual(calculate_exposure_quality_multiplier(220.0), 0.5)

    def test_select_top_k_indices_with_nms(self):
        scored_items = [
            (0, 100.0, {}),
            (1, 95.0, {}),
            (2, 90.0, {}),
            (10, 85.0, {}),
            (25, 80.0, {}),
        ]
        selected = select_top_k_indices(scored_items, top_k=2, min_frame_gap=5)
        selected_fns = [item[0] for item in selected]
        self.assertEqual(selected_fns, [0, 10])


if __name__ == "__main__":
    unittest.main()
