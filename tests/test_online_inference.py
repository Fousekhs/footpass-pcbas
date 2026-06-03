"""Tests for the memory-bounded online inference wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.features.cropper import CropperConfig, PaddedPlayerCropper
from pcspot.inference.online import (
    OnlineInferenceConfig,
    OnlineSpotter,
    SlidingWindowBuffer,
    _FrameRecord,
)
from pcspot.models.pipeline import PlayerCentricSpottingModel


class _FakeExtractor:
    """Deterministic extractor for unit tests.

    Returns crop-mean as a feature so we can verify the wrapper is
    actually plumbing crops through to features.
    """

    def __init__(self, feature_dim: int = 8) -> None:
        self.feature_dim = int(feature_dim)
        self.calls: int = 0

    def extract_features(self, crops: np.ndarray) -> np.ndarray:
        self.calls += 1
        if crops.size == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        N = crops.shape[0]
        feats = np.zeros((N, self.feature_dim), dtype=np.float32)
        # First channel = mean intensity, rest = constant slot for player.
        for i in range(N):
            feats[i, 0] = float(crops[i].astype(np.float32).mean()) / 255.0
            feats[i, 1:] = float(i)
        return feats


def _gradient_frame(h: int = 128, w: int = 128) -> np.ndarray:
    ys = np.arange(h, dtype=np.uint8).reshape(h, 1)
    xs = np.arange(w, dtype=np.uint8).reshape(1, w)
    return (
        np.broadcast_to((xs + ys).astype(np.uint8), (h, w))[..., None]
        .repeat(3, axis=-1)
        .copy()
    )


def _tactical_row(
    frame: int,
    pid: int,
    *,
    visible: bool = True,
) -> np.ndarray:
    if visible:
        roi = (920.0, 480.0, 80.0, 160.0)
    else:
        roi = (float("nan"),) * 4
    row = np.array(
        [
            frame,
            pid,
            1.0,  # left_to_right
            pid % 100,  # shirt
            1,  # role
            0.5,  # x
            0.5,  # y
            0.0,  # speed_x
            0.0,  # speed_y
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            0,  # class
        ],
        dtype=np.float32,
    )
    return row


class SlidingWindowBufferTests(unittest.TestCase):
    def test_evicts_when_full(self) -> None:
        buf = SlidingWindowBuffer(window_size=3)
        for f in range(5):
            buf.push(
                _FrameRecord(
                    frame_index=f,
                    snapshots=[],
                    events=[],
                )
            )
        self.assertEqual(buf.frames(), [2, 3, 4])
        self.assertTrue(buf.is_full())

    def test_eviction_clears_inner_arrays(self) -> None:
        buf = SlidingWindowBuffer(window_size=2)
        rec0 = _FrameRecord(
            frame_index=0,
            snapshots=["dummy"],  # type: ignore[list-item]
            events=["e"],  # type: ignore[list-item]
            visual_features={101: np.zeros(4, dtype=np.float32)},
        )
        buf.push(rec0)
        buf.push(_FrameRecord(frame_index=1, snapshots=[], events=[]))
        buf.push(_FrameRecord(frame_index=2, snapshots=[], events=[]))
        # rec0 was evicted; its inner arrays must be cleared.
        self.assertEqual(rec0.snapshots, [])
        self.assertEqual(rec0.events, [])
        self.assertEqual(rec0.visual_features, {})


class OnlineSpotterTests(unittest.TestCase):
    def _make_model(self, F_visual: int = 8, hidden_dim: int = 16) -> PlayerCentricSpottingModel:
        torch.manual_seed(0)
        return PlayerCentricSpottingModel(
            hidden_dim=hidden_dim,
            visual_dim=F_visual,
            visual_proj_dim=4,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        ).eval()

    def test_step_returns_empty_until_window_fills(self) -> None:
        F_visual = 8
        model = self._make_model(F_visual=F_visual)
        extractor = _FakeExtractor(feature_dim=F_visual)
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32, pad_factor=1.5))
        spotter = OnlineSpotter(
            model,
            extractor,
            cropper,
            OnlineInferenceConfig(window_size=4, stride=1, score_threshold=0.0, fps=25.0),
        )
        frame = _gradient_frame(128, 128)
        for f in range(3):
            preds = spotter.step(
                f,
                frame,
                np.stack([_tactical_row(f, 101), _tactical_row(f, 201)]),
            )
            self.assertEqual(preds, [])
        # Fourth call fills the window.
        preds = spotter.step(
            3,
            frame,
            np.stack([_tactical_row(3, 101), _tactical_row(3, 201)]),
        )
        self.assertIsInstance(preds, list)

    def test_buffer_size_is_bounded(self) -> None:
        F_visual = 8
        model = self._make_model(F_visual=F_visual)
        extractor = _FakeExtractor(feature_dim=F_visual)
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32, pad_factor=1.5))
        spotter = OnlineSpotter(
            model,
            extractor,
            cropper,
            OnlineInferenceConfig(window_size=3, stride=1, score_threshold=0.0),
        )
        frame = _gradient_frame(128, 128)
        for f in range(10):
            spotter.step(
                f,
                frame,
                np.stack([_tactical_row(f, 101), _tactical_row(f, 201)]),
            )
        self.assertLessEqual(len(spotter.buffer), 3)
        self.assertEqual(spotter.buffer.frames()[-1], 9)

    def test_extractor_called_only_for_visible_players(self) -> None:
        F_visual = 8
        model = self._make_model(F_visual=F_visual)
        extractor = _FakeExtractor(feature_dim=F_visual)
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32, pad_factor=1.5))
        spotter = OnlineSpotter(
            model,
            extractor,
            cropper,
            OnlineInferenceConfig(window_size=2, stride=1, score_threshold=0.0),
        )
        frame = _gradient_frame(128, 128)
        rows = np.stack(
            [
                _tactical_row(0, 101, visible=True),
                _tactical_row(0, 201, visible=False),
            ]
        )
        spotter.step(0, frame, rows)
        # Extractor was called once with one visible player.
        self.assertEqual(extractor.calls, 1)
        record = spotter.buffer._records[-1]
        self.assertIn(101, record.visual_features)
        self.assertNotIn(201, record.visual_features)

    def test_reset_clears_state(self) -> None:
        F_visual = 8
        model = self._make_model(F_visual=F_visual)
        extractor = _FakeExtractor(feature_dim=F_visual)
        cropper = PaddedPlayerCropper(CropperConfig(crop_size=32))
        spotter = OnlineSpotter(
            model,
            extractor,
            cropper,
            OnlineInferenceConfig(window_size=3, stride=1, score_threshold=0.0),
        )
        frame = _gradient_frame(128, 128)
        for f in range(2):
            spotter.step(f, frame, np.stack([_tactical_row(f, 101)]))
        self.assertGreater(len(spotter.buffer), 0)
        spotter.reset()
        self.assertEqual(len(spotter.buffer), 0)


if __name__ == "__main__":
    unittest.main()
