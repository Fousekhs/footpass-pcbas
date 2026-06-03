"""Tests for the ``pcspot.train.trainer`` upgrades.

Covers:
- WarmupCosineSchedule shape (warmup then cosine decay).
- Trainer.train_step gradient clipping (no exception, gradients exist).
- save_checkpoint / load_checkpoint roundtrip.
- fit() validation hook + best-checkpoint persistence.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.schema import EventLabel, PlayerSnapshot, Sample, SampleMeta, stack_sample
from pcspot.models.pipeline import PlayerCentricSpottingModel
from pcspot.train.trainer import Trainer, WarmupCosineSchedule


def _make_stacked(num_steps: int = 8) -> list:
    snaps = []
    for t in range(num_steps):
        step = []
        for pid in (101, 102, 201, 202):
            step.append(
                PlayerSnapshot(
                    player_id=pid,
                    team=0 if pid < 200 else 1,
                    shirt_number=pid % 100,
                    role_id=1,
                    x=0.1 * (pid % 5),
                    y=0.1 * (pid % 7),
                    speed_x=0.0,
                    speed_y=0.0,
                    bbox_xywh=(10.0, 10.0, 20.0, 20.0),
                    visible=True,
                )
            )
        snaps.append(step)
    sample = Sample(
        frames=np.arange(num_steps, dtype=np.int64),
        players_per_step=snaps,
        events=[EventLabel(frame=num_steps // 2, player_id=101, class_id=2)],
        meta=SampleMeta(match_id="m"),
    )
    return [stack_sample(sample)]


class WarmupCosineScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine(self) -> None:
        sched = WarmupCosineSchedule(total_steps=20, warmup_steps=5, min_lr_ratio=0.1)
        # Warmup ramps from 1/5 at step 0 to 1.0 at step 4.
        self.assertAlmostEqual(sched(0), 0.2, places=6)
        self.assertAlmostEqual(sched(4), 1.0, places=6)
        # Past warmup, cosine decay starts at 1.0 (step 5).
        self.assertAlmostEqual(sched(5), 1.0, places=6)
        # At end, value approaches min_lr_ratio.
        self.assertAlmostEqual(sched(20), 0.1, places=2)
        # Monotonic non-increasing past warmup.
        for s in range(5, 20):
            self.assertGreaterEqual(sched(s) + 1e-6, sched(s + 1))


class TrainerStepTests(unittest.TestCase):
    def test_train_step_runs_with_grad_clip(self) -> None:
        torch.manual_seed(0)
        samples = _make_stacked(num_steps=6)
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3, grad_clip_norm=0.5)
        log = trainer.train_step(samples)
        self.assertGreaterEqual(log.total_loss, 0.0)
        self.assertEqual(trainer.global_step, 1)

    def test_schedule_modifies_learning_rate(self) -> None:
        torch.manual_seed(0)
        samples = _make_stacked(num_steps=6)
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        sched = WarmupCosineSchedule(total_steps=10, warmup_steps=0)
        trainer = Trainer(
            model, learning_rate=1.0, grad_clip_norm=None, schedule=sched
        )
        first = trainer.train_step(samples)
        second = trainer.train_step(samples)
        # Cosine starts at 1.0 (step 0) and decays from there.
        self.assertAlmostEqual(first.learning_rate, 1.0, places=6)
        self.assertLess(second.learning_rate, first.learning_rate)


class TrainerCheckpointTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        torch.manual_seed(0)
        samples = _make_stacked(num_steps=6)
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3)
        trainer.train_step(samples)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt.pt"
            trainer.save_checkpoint(ckpt)

            model2 = PlayerCentricSpottingModel(
                hidden_dim=8,
                num_hgt_layers=1,
                num_heads=2,
                num_mstcn_stages=1,
                num_mstcn_layers=2,
                knn=2,
            )
            trainer2 = Trainer(model2, learning_rate=1e-3)
            trainer2.load_checkpoint(ckpt)
            self.assertEqual(trainer2.global_step, trainer.global_step)
            # Parameters should match.
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1.detach(), p2.detach()))


class TrainerFitTests(unittest.TestCase):
    def test_fit_runs_with_validation_and_best_ckpt(self) -> None:
        torch.manual_seed(0)
        samples = _make_stacked(num_steps=6)
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3)
        scores = iter([0.4, 0.7, 0.5])  # ep 1 is the best
        with tempfile.TemporaryDirectory() as tmp:
            logs = trainer.fit(
                samples,
                epochs=3,
                batch_size=1,
                validation_fn=lambda _t: {"map": next(scores)},
                checkpoint_dir=tmp,
                keep_best_metric="map",
            )
            self.assertEqual(len(logs), 3)
            self.assertTrue((Path(tmp) / "best.pt").exists())
            self.assertTrue((Path(tmp) / "epoch_0000.pt").exists())
            self.assertTrue((Path(tmp) / "epoch_0001.pt").exists())
            self.assertTrue((Path(tmp) / "epoch_0002.pt").exists())
            # Validation values are captured in EpochLog.
            self.assertAlmostEqual(logs[1].validation["map"], 0.7)


if __name__ == "__main__":
    unittest.main()
