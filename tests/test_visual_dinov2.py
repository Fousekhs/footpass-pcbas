"""Tests for the DINOv2 wrapper (stub backbone)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.features.cropper import CropperConfig, PaddedPlayerCropper
from pcspot.features.dinov2 import DinoV2Config, DinoV2Extractor


class StubExtractorTests(unittest.TestCase):
    def test_stub_features_have_expected_shape(self) -> None:
        cfg = DinoV2Config(
            crop_size=32,
            feature_dim=16,
            use_stub=True,
            batch_size=4,
        )
        ex = DinoV2Extractor(cfg)
        crops = (np.random.rand(5, 32, 32, 3) * 255).astype(np.uint8)
        feats = ex.extract_features(crops)
        self.assertEqual(feats.shape, (5, 16))
        self.assertEqual(feats.dtype, np.float32)

    def test_stub_is_deterministic(self) -> None:
        cfg = DinoV2Config(crop_size=16, feature_dim=8, use_stub=True)
        ex = DinoV2Extractor(cfg)
        rng = np.random.default_rng(0)
        crops = (rng.random((3, 16, 16, 3)) * 255).astype(np.uint8)
        f1 = ex.extract_features(crops)
        f2 = ex.extract_features(crops)
        self.assertTrue(np.allclose(f1, f2))

    def test_fp16_flag_no_crash_on_cpu(self) -> None:
        """fp16=True should be a no-op on CPU (autocast only activates on CUDA)."""
        cfg = DinoV2Config(crop_size=16, feature_dim=8, use_stub=True, fp16=True)
        ex = DinoV2Extractor(cfg)
        crops = (np.random.rand(3, 16, 16, 3) * 255).astype(np.uint8)
        feats = ex.extract_features(crops)
        self.assertEqual(feats.shape, (3, 8))
        self.assertEqual(feats.dtype, np.float32)

    def test_compile_model_flag_with_stub(self) -> None:
        """compile_model=True should be skipped gracefully for stub backbones."""
        cfg = DinoV2Config(
            crop_size=16, feature_dim=8, use_stub=True, compile_model=True
        )
        ex = DinoV2Extractor(cfg)
        crops = (np.random.rand(2, 16, 16, 3) * 255).astype(np.uint8)
        feats = ex.extract_features(crops)
        self.assertEqual(feats.shape, (2, 8))
        self.assertEqual(feats.dtype, np.float32)

    def test_pipeline_with_cropper(self) -> None:
        cfg_extract = DinoV2Config(crop_size=32, feature_dim=8, use_stub=True)
        ex = DinoV2Extractor(cfg_extract)
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32, pad_factor=1.5))
        # Synthetic 64x64 frame.
        ys = np.arange(64, dtype=np.uint8).reshape(64, 1)
        xs = np.arange(64, dtype=np.uint8).reshape(1, 64)
        frame = (
            np.broadcast_to((xs + ys).astype(np.uint8), (64, 64))[..., None]
            .repeat(3, axis=-1)
            .copy()
        )
        # Two visible players.
        rois = np.array(
            [
                [200.0, 200.0, 80.0, 160.0],
                [800.0, 400.0, 100.0, 200.0],
            ],
            dtype=np.float32,
        )
        crops, mask = cropper.extract(frame, rois)
        self.assertTrue(bool(mask.all()))
        feats = ex.extract_features(crops)
        self.assertEqual(feats.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
