"""Tests for ``pcspot.features.cropper``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.features.cropper import (
    CropperConfig,
    PaddedPlayerCropper,
    pad_and_clamp_box,
    scale_roi_box,
)


class ScaleRoiTests(unittest.TestCase):
    def test_scaling_to_smaller_video(self) -> None:
        # FullHD ROI at (192, 108) of 1920x1080 maps to (64, 36) of 640x360.
        out = scale_roi_box(
            (192.0, 108.0, 96.0, 192.0),
            video_width=640,
            video_height=360,
        )
        self.assertIsNotNone(out)
        x, y, w, h = out  # type: ignore[misc]
        self.assertAlmostEqual(x, 64.0, places=5)
        self.assertAlmostEqual(y, 36.0, places=5)
        self.assertAlmostEqual(w, 32.0, places=5)
        self.assertAlmostEqual(h, 64.0, places=5)

    def test_nan_returns_none(self) -> None:
        self.assertIsNone(
            scale_roi_box(
                (float("nan"), 0.0, 10.0, 10.0),
                video_width=1280,
                video_height=720,
            )
        )


class PadClampTests(unittest.TestCase):
    def test_padded_box_squares_up_and_clamps(self) -> None:
        # Tall ROI: w=20, h=80 centered at (100, 100); pad_factor=2 -> 160.
        x1, y1, x2, y2 = pad_and_clamp_box(
            (90.0, 60.0, 20.0, 80.0),
            pad_factor=2.0,
            min_box_size=10,
            frame_width=200,
            frame_height=200,
        )
        # Center should be (100, 100); side = 80 * 2 = 160; half = 80.
        self.assertEqual(x1, 20)
        self.assertEqual(y1, 20)
        self.assertEqual(x2, 180)
        self.assertEqual(y2, 180)

    def test_clamps_to_frame(self) -> None:
        # Box near the corner should be clamped without going negative.
        x1, y1, x2, y2 = pad_and_clamp_box(
            (0.0, 0.0, 10.0, 10.0),
            pad_factor=2.0,
            min_box_size=20,
            frame_width=50,
            frame_height=50,
        )
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 50)
        self.assertLessEqual(y2, 50)
        # Width / height must be positive after clamping.
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_min_box_size_inflates_tiny_rois(self) -> None:
        x1, y1, x2, y2 = pad_and_clamp_box(
            (100.0, 100.0, 1.0, 1.0),
            pad_factor=1.0,
            min_box_size=40,
            frame_width=300,
            frame_height=300,
        )
        # Side becomes 40 (min_box_size * pad_factor=1.0).
        self.assertEqual(x2 - x1, 40)
        self.assertEqual(y2 - y1, 40)


class CropperTests(unittest.TestCase):
    def _frame(self, w: int = 256, h: int = 256) -> np.ndarray:
        # Simple gradient image so resized crops are not all zero.
        ys = np.arange(h, dtype=np.uint8).reshape(h, 1)
        xs = np.arange(w, dtype=np.uint8).reshape(1, w)
        return np.broadcast_to((xs + ys).astype(np.uint8), (h, w))[..., None].repeat(
            3, axis=-1
        ).copy()

    def test_extract_visible_player(self) -> None:
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32, pad_factor=1.5))
        frame = self._frame(256, 256)
        # FullHD ROI scaled to 256x256 frame:
        # 256 / 1920 * x_full == x_video, so use ROI such that center is roughly (128, 128).
        # ROI center = (96 / 1920 * 256 + 4 / 1920 * 256 / 2, ...)
        roi = np.array([[920.0, 480.0, 80.0, 160.0]], dtype=np.float32)
        crops, mask = cropper.extract(frame, roi)
        self.assertEqual(crops.shape, (1, 32, 32, 3))
        self.assertTrue(bool(mask[0]))
        self.assertGreater(int(crops.sum()), 0)

    def test_invisible_player_yields_zero_crop(self) -> None:
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=16))
        frame = self._frame(128, 128)
        roi = np.array([[float("nan"), 0.0, 0.0, 0.0]], dtype=np.float32)
        crops, mask = cropper.extract(frame, roi)
        self.assertFalse(bool(mask[0]))
        self.assertEqual(int(crops.sum()), 0)

    def test_multiple_players(self) -> None:
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=16))
        frame = self._frame(256, 256)
        rois = np.array(
            [
                [100.0, 100.0, 50.0, 100.0],
                [float("nan")] * 4,
                [800.0, 400.0, 80.0, 160.0],
            ],
            dtype=np.float32,
        )
        crops, mask = cropper.extract(frame, rois)
        self.assertEqual(crops.shape, (3, 16, 16, 3))
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))
        self.assertTrue(bool(mask[2]))
        # Invisible row must be all-zero.
        self.assertEqual(int(crops[1].sum()), 0)

    def test_invalid_frame_raises(self) -> None:
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=16))
        bad = np.zeros((10, 10), dtype=np.uint8)  # missing channel axis
        with self.assertRaises(ValueError):
            cropper.extract(bad, np.zeros((1, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
