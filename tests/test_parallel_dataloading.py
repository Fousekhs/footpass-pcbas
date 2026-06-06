"""Tests for the multi-worker / collated training path.

Covers:
- ``collate_with_targets`` produces a batch and targets equivalent to the
  legacy ``stacked_to_batch`` + ``make_full_targets_for_batch`` pair.
- ``build_collated_dataloader_provider`` yields ``(batch, targets)`` pairs and
  carries the ``yields_collated`` marker.
- ``Trainer.train_epoch_from_collated`` and ``Trainer.fit`` route the collated
  provider correctly and increment ``global_step`` once per batch.
- ``Trainer.train_step`` and ``Trainer.train_step_collated`` agree.
- ``compile=True`` constructs without raising and checkpoints stay free of the
  ``_orig_mod.`` prefix torch.compile would otherwise add.

These run with ``num_workers=0`` so they exercise the collate + provider logic
deterministically without depending on the platform's multiprocessing start
method.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import HalfArray
from pcspot.data.splits import SplitManifest
from pcspot.data.targets import CalfConfig
from pcspot.models.pipeline import PlayerCentricSpottingModel, stacked_to_batch
from pcspot.train.trainer import (
    Trainer,
    build_collated_dataloader_provider,
    collate_with_targets,
    make_full_targets_for_batch,
)


def _synthetic_half(num_frames: int = 32, num_players: int = 4) -> HalfArray:
    rows: list[list[float]] = []
    for frame in range(num_frames):
        for p in range(num_players):
            pid = 101 + p if p < num_players // 2 else 201 + (p - num_players // 2)
            rows.append(
                [frame, pid, 1.0, p % 11 + 1, 1, 0.5 + 0.01 * p, 0.5,
                 0.0, 0.0, 100.0, 100.0, 50.0, 50.0, 0.0]
            )
    arr = np.asarray(rows, dtype=np.float32)
    return HalfArray(match_id="m", half_id="H1", array=arr)


def _dataset(num_frames: int = 32, window_size: int = 8) -> PCBASDataset:
    half = _synthetic_half(num_frames=num_frames)
    manifest = SplitManifest.from_dict({"train": [("m", "H1")]})
    return PCBASDataset(
        manifest=manifest,
        split="train",
        window_size=window_size,
        stride=window_size,
        halves=[half],
        calf_config=CalfConfig(),
        compute_targets=True,
    )


def _model() -> PlayerCentricSpottingModel:
    torch.manual_seed(0)
    return PlayerCentricSpottingModel(
        hidden_dim=8,
        num_hgt_layers=1,
        num_heads=2,
        num_mstcn_stages=1,
        num_mstcn_layers=2,
        knn=2,
    )


class CollateWithTargetsTests(unittest.TestCase):
    def test_matches_legacy_batch_and_targets(self) -> None:
        ds = _dataset()
        items = [ds[i] for i in range(4)]
        samples = [it[0] for it in items]

        batch, targets = collate_with_targets(items)
        ref_batch = stacked_to_batch(samples)
        ref_tgt = make_full_targets_for_batch(
            samples, config=CalfConfig(), num_classes=_model().num_classes
        )

        self.assertTrue(torch.equal(batch.pitch_xy, ref_batch.pitch_xy))
        self.assertTrue(torch.equal(batch.valid_mask, ref_batch.valid_mask))
        self.assertTrue(torch.allclose(targets.class_targets, ref_tgt.class_targets))
        self.assertTrue(torch.allclose(targets.class_weights, ref_tgt.class_weights))
        self.assertTrue(
            torch.allclose(targets.objectness_targets, ref_tgt.objectness_targets)
        )

    def test_empty_items_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_with_targets([])

    def test_missing_targets_rejected(self) -> None:
        ds = _dataset()
        items = [(ds[0][0], None), (ds[1][0], None)]
        with self.assertRaises(ValueError):
            collate_with_targets(items)


class CollatedProviderTests(unittest.TestCase):
    def test_provider_yields_collated_pairs(self) -> None:
        ds = _dataset(num_frames=40, window_size=8)  # 5 windows
        provider = build_collated_dataloader_provider(
            ds,
            sampler_factory=lambda e: range(len(ds)),
            batch_size=2,
            num_workers=0,
        )
        self.assertTrue(getattr(provider, "yields_collated", False))
        pairs = list(provider(0))
        self.assertEqual(len(pairs), 3)  # [2, 2, 1]
        batch, targets = pairs[0]
        self.assertEqual(batch.pitch_xy.shape[0], 2)
        self.assertEqual(targets.class_targets.shape[0], 2)

    def test_train_epoch_from_collated_increments_step(self) -> None:
        ds = _dataset(num_frames=32, window_size=8)  # 4 windows
        trainer = Trainer(_model(), learning_rate=1e-3)
        provider = build_collated_dataloader_provider(
            ds,
            sampler_factory=lambda e: range(len(ds)),
            batch_size=2,
            num_workers=0,
        )
        steps = list(trainer.train_epoch_from_collated(provider(0)))
        self.assertEqual(len(steps), 2)
        self.assertEqual(trainer.global_step, 2)

    def test_fit_routes_collated_provider(self) -> None:
        ds = _dataset(num_frames=32, window_size=8)
        trainer = Trainer(_model(), learning_rate=1e-3)
        provider = build_collated_dataloader_provider(
            ds,
            sampler_factory=lambda e: range(len(ds)),
            batch_size=2,
            num_workers=0,
        )
        logs = trainer.fit(provider, epochs=2, batch_size=2)
        self.assertEqual(len(logs), 2)
        self.assertEqual(trainer.global_step, 4)  # 2 batches x 2 epochs


class TrainStepEquivalenceTests(unittest.TestCase):
    def test_collated_and_raw_train_step_agree(self) -> None:
        ds = _dataset(num_frames=16, window_size=8)
        samples = [ds[i][0] for i in range(2)]

        # Raw path.
        t_raw = Trainer(_model(), learning_rate=0.0, grad_clip_norm=None)
        log_raw = t_raw.train_step(samples)

        # Collated path on a fresh, identically-seeded trainer.
        t_col = Trainer(_model(), learning_rate=0.0, grad_clip_norm=None)
        batch, targets = collate_with_targets([ds[0], ds[1]])
        log_col = t_col.train_step_collated(batch, targets)

        self.assertAlmostEqual(log_raw.total_loss, log_col.total_loss, places=5)


class CompileOptionTests(unittest.TestCase):
    def test_compile_true_constructs_and_checkpoints_clean(self) -> None:
        # torch.compile is lazy (no compilation until first forward), so
        # construction is cheap. We only assert it does not raise and that
        # checkpoint keys carry no _orig_mod. prefix.
        trainer = Trainer(_model(), learning_rate=1e-3, compile=True)
        self.assertIsInstance(trainer.compiled, bool)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            trainer.save_checkpoint(path)
            payload = torch.load(str(path), weights_only=True)
            keys = list(payload["model_state"].keys())
            self.assertTrue(keys)
            self.assertFalse(any(k.startswith("_orig_mod.") for k in keys))
            # Round-trips back into a plain (uncompiled) trainer.
            plain = Trainer(_model(), learning_rate=1e-3)
            plain.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
