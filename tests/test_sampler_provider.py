"""Tests for the BatchProvider plumbing in ``pcspot.data.sampling``.

Covers:
- ``build_dataset_batch_provider`` yields batches of the configured size
  and returns the trailing partial batch unless ``drop_last`` is set.
- ``reseeded_mixed_sampler_factory`` produces deterministic sequences
  when seeded and yields different sequences in different epochs.
- ``Trainer.train_epoch_from_batches`` runs end-to-end and increments
  ``global_step`` once per batch.
- ``Trainer.fit`` dispatches to the provider path when given a
  callable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import HalfArray
from pcspot.data.sampling import (
    build_dataset_batch_provider,
    reseeded_mixed_sampler_factory,
)
from pcspot.data.splits import SplitManifest
from pcspot.data.targets import CalfConfig
from pcspot.models.pipeline import PlayerCentricSpottingModel
from pcspot.train.trainer import Trainer


def _synthetic_half(
    num_frames: int = 32,
    num_players: int = 4,
    events: list[tuple[int, int, int]] | None = None,
) -> HalfArray:
    events = events or []
    rows: list[list[float]] = []
    for frame in range(num_frames):
        for p in range(num_players):
            pid = 101 + p if p < num_players // 2 else 201 + (p - num_players // 2)
            cls = 0
            for ef, ep, ec in events:
                if frame == ef and pid == ep:
                    cls = ec
            rows.append(
                [
                    frame,
                    pid,
                    1.0,
                    p % 11 + 1,
                    1,
                    0.5 + 0.01 * p,
                    0.5,
                    0.0,
                    0.0,
                    100.0,
                    100.0,
                    50.0,
                    50.0,
                    cls,
                ]
            )
    arr = np.asarray(rows, dtype=np.float32)
    return HalfArray(match_id="m", half_id="H1", array=arr)


def _dataset(num_frames: int = 32, events: list[tuple[int, int, int]] | None = None,
             window_size: int = 8, compute_targets: bool = False) -> PCBASDataset:
    half = _synthetic_half(num_frames=num_frames, events=events)
    manifest = SplitManifest.from_dict({"train": [("m", "H1")]})
    return PCBASDataset(
        manifest=manifest,
        split="train",
        window_size=window_size,
        stride=window_size,
        halves=[half],
        calf_config=CalfConfig(),
        compute_targets=compute_targets,
    )


class BuildDatasetBatchProviderTests(unittest.TestCase):
    def test_batches_have_expected_size(self) -> None:
        ds = _dataset(num_frames=40, window_size=8)  # 5 windows
        provider = build_dataset_batch_provider(
            ds,
            sampler_factory=lambda epoch: range(len(ds)),
            batch_size=2,
        )
        batches = list(provider(epoch=0))
        sizes = [len(b) for b in batches]
        # Five samples in batches of 2 -> [2, 2, 1].
        self.assertEqual(sizes, [2, 2, 1])

    def test_drop_last_drops_trailing_partial_batch(self) -> None:
        ds = _dataset(num_frames=40, window_size=8)
        provider = build_dataset_batch_provider(
            ds,
            sampler_factory=lambda epoch: range(len(ds)),
            batch_size=2,
            drop_last=True,
        )
        batches = list(provider(epoch=0))
        self.assertEqual([len(b) for b in batches], [2, 2])

    def test_invalid_batch_size_rejected(self) -> None:
        ds = _dataset()
        with self.assertRaises(ValueError):
            build_dataset_batch_provider(
                ds, sampler_factory=lambda e: [], batch_size=0
            )


class ReseededMixedSamplerFactoryTests(unittest.TestCase):
    def test_same_seed_same_sequence(self) -> None:
        counts = [0, 0, 1, 0, 1]
        f1 = reseeded_mixed_sampler_factory(
            counts, base_seed=42, num_samples=10
        )
        f2 = reseeded_mixed_sampler_factory(
            counts, base_seed=42, num_samples=10
        )
        self.assertEqual(list(f1(0)), list(f2(0)))

    def test_different_epoch_diverges(self) -> None:
        counts = [0, 0, 1, 0, 1, 0, 0, 1]
        factory = reseeded_mixed_sampler_factory(
            counts, base_seed=7, num_samples=20
        )
        a = list(factory(0))
        b = list(factory(1))
        # With a base seed shift the two sequences should differ in at
        # least one position (statistically certain for length 20).
        self.assertNotEqual(a, b)


class TrainerProviderPathTests(unittest.TestCase):
    def _model(self) -> PlayerCentricSpottingModel:
        torch.manual_seed(0)
        return PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )

    def test_train_epoch_from_batches_increments_step(self) -> None:
        ds = _dataset(num_frames=32, window_size=8)  # 4 windows
        trainer = Trainer(self._model(), learning_rate=1e-3)
        provider = build_dataset_batch_provider(
            ds, sampler_factory=lambda e: range(len(ds)), batch_size=2
        )
        steps = list(trainer.train_epoch_from_batches(provider(0)))
        # 4 windows / batch_size 2 = 2 batches = 2 train steps.
        self.assertEqual(len(steps), 2)
        self.assertEqual(trainer.global_step, 2)

    def test_fit_accepts_provider_callable(self) -> None:
        ds = _dataset(num_frames=32, window_size=8)
        trainer = Trainer(self._model(), learning_rate=1e-3)
        provider = build_dataset_batch_provider(
            ds, sampler_factory=lambda e: range(len(ds)), batch_size=2
        )
        logs = trainer.fit(provider, epochs=2, batch_size=2)
        self.assertEqual(len(logs), 2)
        # 4 windows / batch_size 2 = 2 steps/epoch, 2 epochs.
        self.assertEqual(trainer.global_step, 4)


class TrainerGradAccumTests(unittest.TestCase):
    def test_optimizer_steps_every_n_calls(self) -> None:
        torch.manual_seed(0)
        ds = _dataset(num_frames=32, window_size=8)
        model = PlayerCentricSpottingModel(
            hidden_dim=8, num_hgt_layers=1, num_heads=2,
            num_mstcn_stages=1, num_mstcn_layers=2, knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3, grad_accum_steps=2)
        chunks = [
            [ds[i][0] for i in (0, 1)],
            [ds[i][0] for i in (2, 3)],
        ]
        # Capture parameter snapshot before the first micro-batch and
        # after the first/second micro-batches.
        p_before = next(model.parameters()).detach().clone()
        trainer.train_step(chunks[0])
        p_mid = next(model.parameters()).detach().clone()
        # After one micro-batch with grad_accum=2 the optimizer has NOT
        # stepped yet, so parameters should be identical.
        self.assertTrue(torch.allclose(p_before, p_mid))
        self.assertEqual(trainer._accum_counter, 1)
        trainer.train_step(chunks[1])
        p_after = next(model.parameters()).detach().clone()
        # Now the optimizer has stepped, so parameters should change.
        self.assertFalse(torch.allclose(p_before, p_after))
        self.assertEqual(trainer._accum_counter, 0)
        self.assertEqual(trainer.global_step, 2)


class TrainerLatestCheckpointTests(unittest.TestCase):
    def test_fit_writes_latest_pt(self) -> None:
        import tempfile

        torch.manual_seed(0)
        ds = _dataset(num_frames=16, window_size=8)
        model = PlayerCentricSpottingModel(
            hidden_dim=8, num_hgt_layers=1, num_heads=2,
            num_mstcn_stages=1, num_mstcn_layers=2, knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3)
        provider = build_dataset_batch_provider(
            ds, sampler_factory=lambda e: range(len(ds)), batch_size=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer.fit(
                provider,
                epochs=2,
                batch_size=2,
                checkpoint_dir=tmp,
            )
            self.assertTrue((Path(tmp) / "latest.pt").exists())
            self.assertTrue((Path(tmp) / "epoch_0000.pt").exists())
            self.assertTrue((Path(tmp) / "epoch_0001.pt").exists())


class TrainerStartEpochTests(unittest.TestCase):
    def test_start_epoch_overrides_filename(self) -> None:
        import tempfile

        torch.manual_seed(0)
        ds = _dataset(num_frames=16, window_size=8)
        model = PlayerCentricSpottingModel(
            hidden_dim=8, num_hgt_layers=1, num_heads=2,
            num_mstcn_stages=1, num_mstcn_layers=2, knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3)
        provider = build_dataset_batch_provider(
            ds, sampler_factory=lambda e: range(len(ds)), batch_size=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            logs = trainer.fit(
                provider, epochs=1, batch_size=2,
                checkpoint_dir=tmp, start_epoch=5,
            )
            self.assertEqual(logs[0].epoch, 5)
            self.assertTrue((Path(tmp) / "epoch_0005.pt").exists())


if __name__ == "__main__":
    unittest.main()
