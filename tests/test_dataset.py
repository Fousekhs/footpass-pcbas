"""Tests for the lazy dataset, target cache, samplers, and split discipline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.cache import TargetCache, config_hash
from pcspot.data.dataset import DatasetItemKey, PCBASDataset
from pcspot.data.loader import HalfArray
from pcspot.data.sampling import MixedEventSampler, UniformWindowSampler
from pcspot.data.splits import SplitManifest, ensure_no_overlap
from pcspot.data.targets import CalfConfig


def _synthetic_half(
    match_id: str = "m",
    half_id: str = "h",
    num_players: int = 4,
    num_frames: int = 40,
    events: list[tuple[int, int, int]] | None = None,
) -> HalfArray:
    """Build a small 14-column tactical array for testing."""
    events = events or []
    rows = []
    for frame in range(num_frames):
        for p in range(num_players):
            pid = 101 + p if p < 2 else 201 + (p - 2)
            cls = 0
            for ef, ep, ec in events:
                if frame == ef and pid == ep:
                    cls = ec
            rows.append(
                [
                    frame,
                    pid,
                    1.0,  # left_to_right
                    p % 11 + 1,  # shirt
                    1,  # role
                    0.5 + 0.01 * p,  # x
                    0.5,  # y
                    0.0,  # speed_x
                    0.0,  # speed_y
                    100.0,  # roi_x
                    100.0,  # roi_y
                    50.0,  # roi_w
                    50.0,  # roi_h
                    cls,
                ]
            )
    arr = np.asarray(rows, dtype=np.float32)
    return HalfArray(match_id=match_id, half_id=half_id, array=arr)


class SplitManifestTests(unittest.TestCase):
    def test_from_dict_and_halves_for(self) -> None:
        m = SplitManifest.from_dict(
            {
                "train": [("m1", "h1"), ("m1", "h2")],
                "val": [("m2", "h1")],
            }
        )
        self.assertEqual(m.split_of("m1", "h1"), "train")
        self.assertEqual(m.split_of("m2", "h1"), "val")
        self.assertIsNone(m.split_of("m9", "h9"))
        self.assertEqual(len(m.halves_for("train")), 2)
        self.assertEqual(len(m.halves_for("val")), 1)

    def test_conflicting_assignment_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SplitManifest.from_dict(
                {
                    "train": [("m1", "h1")],
                    "val": [("m1", "h1")],
                }
            )

    def test_ensure_no_overlap_across_manifests(self) -> None:
        a = SplitManifest.from_dict({"train": [("m1", "h1")]})
        b = SplitManifest.from_dict({"val": [("m1", "h1")]})
        with self.assertRaises(ValueError):
            ensure_no_overlap(a, b)


class TargetCacheTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        cfg = CalfConfig()
        with tempfile.TemporaryDirectory() as tmp:
            cache = TargetCache(tmp, config=cfg)
            ct = np.ones((4, 3, 12), dtype=np.float32)
            cw = np.full((4, 3, 12), 0.5, dtype=np.float32)
            ot = np.ones((4, 3), dtype=np.float32)
            ow = np.full((4, 3), 0.5, dtype=np.float32)
            self.assertFalse(cache.has("k"))
            cache.save("k", ct, cw, ot, ow)
            self.assertTrue(cache.has("k"))
            loaded = cache.load("k")
            self.assertIsNotNone(loaded)
            ct2, cw2, ot2, ow2 = loaded  # type: ignore[misc]
            np.testing.assert_array_equal(ct, ct2)
            np.testing.assert_array_equal(cw, cw2)
            np.testing.assert_array_equal(ot, ot2)
            np.testing.assert_array_equal(ow, ow2)

    def test_config_changes_invalidate_cache(self) -> None:
        cfg_a = CalfConfig(k1_default=2)
        cfg_b = CalfConfig(k1_default=5)
        self.assertNotEqual(config_hash(cfg_a), config_hash(cfg_b))
        with tempfile.TemporaryDirectory() as tmp:
            ca = TargetCache(tmp, config=cfg_a)
            cb = TargetCache(tmp, config=cfg_b)
            ca.save(
                "k",
                np.zeros((1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
            self.assertTrue(ca.has("k"))
            self.assertFalse(cb.has("k"))


class PCBASDatasetTests(unittest.TestCase):
    def test_dataset_iterates_windows_and_builds_targets(self) -> None:
        half = _synthetic_half(
            num_frames=40,
            events=[(10, 101, 2), (25, 202, 7)],
        )
        manifest = SplitManifest.from_dict({"train": [("m", "h")]})
        ds = PCBASDataset(
            manifest=manifest,
            split="train",
            window_size=16,
            stride=16,
            halves=[half],
        )
        self.assertGreater(len(ds), 0)
        stacked, targets = ds[0]
        self.assertIsNotNone(targets)
        self.assertEqual(targets.class_targets.shape[0], stacked.num_steps)
        self.assertEqual(
            targets.class_targets.shape[1], stacked.num_players
        )
        self.assertEqual(targets.objectness_targets.shape, (stacked.num_steps, stacked.num_players))

    def test_dataset_uses_cache_on_second_call(self) -> None:
        half = _synthetic_half(num_frames=40, events=[(10, 101, 2)])
        manifest = SplitManifest.from_dict({"train": [("m", "h")]})
        with tempfile.TemporaryDirectory() as tmp:
            cache = TargetCache(tmp, config=CalfConfig())
            ds = PCBASDataset(
                manifest=manifest,
                split="train",
                window_size=16,
                stride=16,
                halves=[half],
                target_cache=cache,
            )
            _, t1 = ds[0]
            # Cache file should now exist for that item.
            item_key = ds.items[0].cache_key()
            self.assertTrue(cache.has(item_key))
            _, t2 = ds[0]
            np.testing.assert_array_equal(t1.class_targets, t2.class_targets)
            np.testing.assert_array_equal(t1.class_weights, t2.class_weights)

    def test_dataset_rejects_unknown_half(self) -> None:
        # Manifest contains a half we did not provide; no loader either.
        manifest = SplitManifest.from_dict({"train": [("m", "missing")]})
        with self.assertRaises(ValueError):
            PCBASDataset(
                manifest=manifest,
                split="train",
                window_size=16,
                halves=[_synthetic_half()],
            )

    def test_dataset_attaches_visual_features_from_cache(self) -> None:
        from pcspot.features.cache import (
            VisualFeatureCache,
            VisualFeatureMetadata,
            VisualFeatureStore,
        )

        F = 4
        half = _synthetic_half(num_frames=16)
        manifest = SplitManifest.from_dict({"train": [("m", "h")]})
        with tempfile.TemporaryDirectory() as tmp:
            store = VisualFeatureStore(
                tmp,
                metadata=VisualFeatureMetadata(
                    backbone_name="dinov2_test",
                    feature_dim=F,
                    crop_size=16,
                    pad_factor=1.5,
                    fullhd_width=1920,
                    fullhd_height=1080,
                ),
            )
            for f in range(16):
                pids = [101, 102, 201, 202]
                feats = np.tile(np.arange(F, dtype=np.float32), (len(pids), 1))
                store.add_frame("m", "h", frame=f, player_ids=pids, features=feats)
            store.flush()

            cache = VisualFeatureCache(tmp, backbone_name="dinov2_test")
            ds = PCBASDataset(
                manifest=manifest,
                split="train",
                window_size=8,
                stride=8,
                halves=[half],
                visual_feature_cache=cache,
                compute_targets=False,
            )
            stacked, _ = ds[0]
            self.assertIsNotNone(stacked.visual_features)
            assert stacked.visual_features is not None
            self.assertEqual(
                stacked.visual_features.shape,
                (stacked.num_steps, stacked.num_players, F),
            )
            # Padded / invalid columns are zeroed by align_features_to_stacked.
            zero_mask = ~stacked.valid_mask
            if bool(zero_mask.any()):
                self.assertEqual(
                    int(stacked.visual_features[zero_mask].sum()), 0
                )

    def test_event_counts_align_with_windows(self) -> None:
        half = _synthetic_half(
            num_frames=64,
            events=[(5, 101, 2), (40, 202, 7), (50, 102, 4)],
        )
        manifest = SplitManifest.from_dict({"train": [("m", "h")]})
        ds = PCBASDataset(
            manifest=manifest,
            split="train",
            window_size=16,
            stride=16,
            halves=[half],
        )
        self.assertEqual(len(ds.event_counts), len(ds))
        total_events = sum(ds.event_counts)
        self.assertEqual(total_events, 3)


class SamplerTests(unittest.TestCase):
    def test_mixed_sampler_oversamples_positives(self) -> None:
        counts = [0] * 90 + [1] * 10  # 10% positives
        sampler = MixedEventSampler(
            counts,
            num_samples=1000,
            positive_ratio=0.7,
            seed=0,
            replacement=True,
        )
        drawn = list(sampler)
        self.assertEqual(len(drawn), 1000)
        positives = sum(1 for i in drawn if counts[i] > 0)
        # Should be close to 700 with low variance.
        self.assertGreater(positives, 600)
        self.assertLess(positives, 800)

    def test_mixed_sampler_handles_pure_negative(self) -> None:
        counts = [0] * 50
        sampler = MixedEventSampler(counts, num_samples=20, seed=1)
        drawn = list(sampler)
        self.assertEqual(len(drawn), 20)
        for i in drawn:
            self.assertEqual(counts[i], 0)

    def test_linear_event_weighting_prefers_busy_windows(self) -> None:
        counts = [1] * 8 + [5] * 2  # 2 very-busy windows
        sampler = MixedEventSampler(
            counts,
            num_samples=2000,
            positive_ratio=1.0,
            event_weighting="linear",
            cap_events_per_window=5,
            seed=0,
        )
        drawn = list(sampler)
        busy_hits = sum(1 for i in drawn if counts[i] == 5)
        light_hits = sum(1 for i in drawn if counts[i] == 1)
        # Probability of busy = 2 * 5 / (8 * 1 + 2 * 5) = 10/18.
        self.assertGreater(busy_hits, light_hits)

    def test_uniform_window_sampler_visits_every_index(self) -> None:
        sampler = UniformWindowSampler(7)
        self.assertEqual(list(sampler), list(range(7)))


if __name__ == "__main__":
    unittest.main()
