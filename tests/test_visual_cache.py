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

    def test_lookup_window_matches_scalar_lookup(self) -> None:
        # The vectorised lookup_window must agree cell-for-cell with the
        # scalar lookup() across present/missing frames, present/missing
        # players, and invisible rows.
        rng = np.random.default_rng(7)
        n_frames = 12
        players = [101, 202, 303, 404]
        rows_f: list[int] = []
        rows_p: list[int] = []
        rows_vis: list[bool] = []
        for f in range(n_frames):
            for pid in players:
                # Randomly drop some (frame, player) rows entirely.
                if rng.random() < 0.25:
                    continue
                rows_f.append(f)
                rows_p.append(pid)
                rows_vis.append(bool(rng.random() > 0.3))
        frames = np.array(rows_f, dtype=np.int64)
        pids = np.array(rows_p, dtype=np.int64)
        feats = rng.standard_normal((len(rows_f), 5)).astype(np.float32)
        visible = np.array(rows_vis, dtype=bool)
        shard = _ShardIndex.from_arrays(frames, pids, feats, visible)

        # Query a window that includes a frame with no rows at all (99).
        q_frames = list(range(n_frames)) + [99]
        q_players = [404, 101, 999, 202]  # includes an absent player (999)
        out, valid = shard.lookup_window(q_frames, q_players)

        for ti, f in enumerate(q_frames):
            for pi, pid in enumerate(q_players):
                vec = shard.lookup(int(f), int(pid))
                if vec is None:
                    self.assertFalse(bool(valid[ti, pi]))
                    self.assertEqual(int(out[ti, pi].sum() != 0), 0)
                else:
                    self.assertTrue(bool(valid[ti, pi]))
                    self.assertTrue(np.allclose(out[ti, pi], vec))


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


class MmapShardLayoutTests(unittest.TestCase):
    def test_from_mmap_matches_from_arrays(self) -> None:
        # The lazy memmap index must agree cell-for-cell with the eager one,
        # including when the on-disk rows are NOT in frame-sorted order.
        rng = np.random.default_rng(11)
        n = 40
        frames = rng.integers(0, 8, size=n).astype(np.int64)
        pids = rng.choice([101, 202, 303, 404], size=n).astype(np.int64)
        # Make (frame, player) unique to match the shard invariant.
        seen: set[tuple[int, int]] = set()
        keep = []
        for i in range(n):
            key = (int(frames[i]), int(pids[i]))
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)
        frames, pids = frames[keep], pids[keep]
        feats = rng.standard_normal((len(keep), 6)).astype(np.float32)
        visible = rng.random(len(keep)) > 0.3

        eager = _ShardIndex.from_arrays(frames, pids, feats, visible)
        lazy = _ShardIndex.from_mmap(
            features=feats, frames=frames, player_ids=pids, visible=visible
        )
        q_frames = list(range(8)) + [99]
        q_players = [404, 101, 999, 202]
        oe, ve = eager.lookup_window(q_frames, q_players)
        ol, vl = lazy.lookup_window(q_frames, q_players)
        self.assertTrue(np.array_equal(ve, vl))
        self.assertTrue(np.allclose(oe, ol))

    def test_converted_cache_reads_identically(self) -> None:
        import importlib

        sys.path.insert(0, str(ROOT / "scripts"))
        conv = importlib.import_module("convert_visual_cache_mmap")

        with tempfile.TemporaryDirectory() as td:
            F = 5
            store = VisualFeatureStore(td, metadata=_meta(F=F))
            f0 = np.random.randn(2, F).astype(np.float32)
            f1 = np.random.randn(2, F).astype(np.float32)
            store.add_frame("m1", "h1", frame=10, player_ids=[101, 201], features=f0)
            store.add_frame("m1", "h1", frame=11, player_ids=[101, 201], features=f1)
            store.flush()
            backbone_dir = Path(td) / "dinov2_vits14_test"

            # Baseline read from the legacy compressed shard.
            legacy = VisualFeatureCache(td, backbone_name="dinov2_vits14_test")
            base_out, base_valid = legacy.get_window("m1", "h1", [10, 11], [101, 201])

            # Convert in place, then read again -- the reader must now prefer
            # the memmap layout and return identical values.
            rc = conv.main([
                "--cache-root", td,
                "--backbone", "dinov2_vits14_test",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((backbone_dir / "m1__h1.features.npy").exists())
            self.assertTrue((backbone_dir / "m1__h1.meta.npz").exists())

            converted = VisualFeatureCache(td, backbone_name="dinov2_vits14_test")
            self.assertTrue(converted.has_shard("m1", "h1"))
            out, valid = converted.get_window("m1", "h1", [10, 11], [101, 201])
            self.assertTrue(np.array_equal(base_valid, valid))
            self.assertTrue(np.allclose(base_out, out))
            self.assertTrue(np.allclose(out[0, 0], f0[0]))
            self.assertTrue(np.allclose(out[1, 1], f1[1]))

            # Release the memmap so the TemporaryDirectory can be cleaned on
            # Windows (open mmaps lock the file there; harmless on POSIX).
            import gc

            converted._lru.clear()
            del converted, legacy, out, base_out
            gc.collect()


if __name__ == "__main__":
    unittest.main()
