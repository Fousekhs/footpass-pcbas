"""Tests for ``pcspot.features.cache``."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.schema import (
    PlayerSnapshot,
    Sample,
    SampleMeta,
    stack_sample,
)
from pcspot.features.cache import (
    VisualFeatureCache,
    VisualFeatureMetadata,
    VisualFeatureStore,
    align_features_to_stacked,
    _ShardIndex,
)


def _meta(F: int = 8) -> VisualFeatureMetadata:
    return VisualFeatureMetadata(
        backbone_name="dinov2_vits14_test",
        feature_dim=F,
        crop_size=32,
        pad_factor=1.5,
        fullhd_width=1920,
        fullhd_height=1080,
    )


class ShardIndexTests(unittest.TestCase):
    def test_lookup_returns_visible_row(self) -> None:
        frames = np.array([5, 5, 6, 6], dtype=np.int64)
        pids = np.array([101, 201, 101, 201], dtype=np.int64)
        feats = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        )
        visible = np.array([True, True, False, True], dtype=bool)
        shard = _ShardIndex.from_arrays(frames, pids, feats, visible)
        self.assertTrue(np.allclose(shard.lookup(5, 101), [1.0, 0.0]))
        self.assertTrue(np.allclose(shard.lookup(6, 201), [4.0, 0.0]))
        # Missing visibility -> None.
        self.assertIsNone(shard.lookup(6, 101))
        # Missing key -> None.
        self.assertIsNone(shard.lookup(99, 101))

    def test_lookup_window_aligns_to_grid(self) -> None:
        frames = np.array([1, 1, 2], dtype=np.int64)
        pids = np.array([101, 201, 101], dtype=np.int64)
        feats = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )
        visible = np.array([True, True, True], dtype=bool)
        shard = _ShardIndex.from_arrays(frames, pids, feats, visible)
        out, valid = shard.lookup_window([1, 2], [101, 201])
        self.assertEqual(out.shape, (2, 2, 2))
        self.assertEqual(valid.shape, (2, 2))
        # Frame 1 has both players.
        self.assertTrue(bool(valid[0, 0]))
        self.assertTrue(bool(valid[0, 1]))
        # Frame 2 has only 101.
        self.assertTrue(bool(valid[1, 0]))
        self.assertFalse(bool(valid[1, 1]))
        # Missing rows are zero.
        self.assertEqual(int(out[1, 1].sum()), 0)


class StoreCacheRoundtripTests(unittest.TestCase):
    def test_write_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = VisualFeatureStore(td, metadata=_meta(F=4))
            feats0 = np.random.randn(2, 4).astype(np.float32)
            feats1 = np.random.randn(2, 4).astype(np.float32)
            store.add_frame("m1", "h1", frame=10, player_ids=[101, 201], features=feats0)
            store.add_frame("m1", "h1", frame=11, player_ids=[101, 201], features=feats1)
            written = store.flush()
            self.assertEqual(len(written), 1)

            cache = VisualFeatureCache(td, backbone_name="dinov2_vits14_test")
            self.assertTrue(cache.has_shard("m1", "h1"))
            self.assertEqual(cache.feature_dim(), 4)
            out, valid = cache.get_window("m1", "h1", [10, 11], [101, 201])
            self.assertEqual(out.shape, (2, 2, 4))
            self.assertTrue(np.allclose(out[0, 0], feats0[0]))
            self.assertTrue(np.allclose(out[0, 1], feats0[1]))
            self.assertTrue(np.allclose(out[1, 0], feats1[0]))
            self.assertTrue(np.allclose(out[1, 1], feats1[1]))
            self.assertTrue(bool(valid.all()))

    def test_metadata_sidecar_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = VisualFeatureStore(td, metadata=_meta(F=3))
            store.add_frame(
                "mA",
                "h1",
                frame=0,
                player_ids=[101],
                features=np.zeros((1, 3), dtype=np.float32),
            )
            store.flush()
            cache = VisualFeatureCache(td, backbone_name="dinov2_vits14_test")
            md = cache.metadata
            self.assertIsNotNone(md)
            assert md is not None
            self.assertEqual(md.feature_dim, 3)
            self.assertEqual(md.crop_size, 32)


def _snap(pid: int, team: int = 0) -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=pid,
        team=team,
        shirt_number=pid % 100,
        role_id=1,
        x=0.5,
        y=0.5,
        speed_x=0.0,
        speed_y=0.0,
        bbox_xywh=(0.0, 0.0, 10.0, 10.0),
        visible=True,
    )


class AlignToStackedTests(unittest.TestCase):
    def test_align_attaches_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            F = 4
            store = VisualFeatureStore(td, metadata=_meta(F=F))
            store.add_frame(
                "m",
                "h",
                frame=0,
                player_ids=[101, 201],
                features=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            )
            store.add_frame(
                "m",
                "h",
                frame=1,
                player_ids=[101],
                features=np.array([[0, 0, 1, 0]], dtype=np.float32),
            )
            store.flush()
            cache = VisualFeatureCache(td, backbone_name="dinov2_vits14_test")

            sample = Sample(
                frames=np.array([0, 1], dtype=np.int64),
                players_per_step=[
                    [_snap(101), _snap(201, team=1)],
                    [_snap(101), _snap(201, team=1)],
                ],
                events=[],
                meta=SampleMeta(match_id="m", half_id="h"),
            )
            stacked = stack_sample(sample)
            aligned = align_features_to_stacked(stacked, cache)
            self.assertIsNotNone(aligned.visual_features)
            assert aligned.visual_features is not None
            self.assertEqual(aligned.visual_features.shape, (2, 2, F))
            # Player 101 at frame 0.
            self.assertTrue(np.allclose(aligned.visual_features[0, 0], [1, 0, 0, 0]))
            # Player 201 at frame 1 should be missing -> zero row.
            self.assertEqual(int(aligned.visual_features[1, 1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
