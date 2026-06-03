"""Tests for ``pcspot.models.pipeline``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.schema import (
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
    stack_sample,
)
from pcspot.models.pipeline import (
    EXTRA_SCALAR_DIM,
    TIME_FEATURE_DIM,
    PlayerCentricSpottingModel,
    _compute_acceleration,
    _compute_time_features,
    compute_goal_distances,
    stacked_to_batch,
)


def _make_sample(num_steps: int = 8, players: tuple[int, ...] = (101, 102, 201, 202)) -> Sample:
    snaps_per_step = []
    for t in range(num_steps):
        step = []
        for pid in players:
            step.append(
                PlayerSnapshot(
                    player_id=pid,
                    team=0 if pid < 200 else 1,
                    shirt_number=pid % 100,
                    role_id=1 + (pid % 13),
                    x=0.1 * (pid % 5),
                    y=0.1 * (pid % 7),
                    speed_x=0.0,
                    speed_y=0.0,
                    bbox_xywh=(10.0, 20.0, 30.0, 40.0),
                    visible=True,
                )
            )
        snaps_per_step.append(step)
    events = [EventLabel(frame=num_steps // 2, player_id=players[0], class_id=2)]
    return Sample(
        frames=np.arange(num_steps, dtype=np.int64),
        players_per_step=snaps_per_step,
        events=events,
        meta=SampleMeta(match_id="m"),
    )


class PipelineTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        sample_a = stack_sample(_make_sample(num_steps=12))
        sample_b = stack_sample(_make_sample(num_steps=12, players=(101, 201)))
        batch = stacked_to_batch([sample_a, sample_b])
        model = PlayerCentricSpottingModel(
            hidden_dim=16,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=2,
            num_mstcn_layers=3,
            knn=2,
        )
        out = model(batch)
        B = 2
        T = 12
        P = sample_a.num_players  # max across batch
        self.assertEqual(out["logits"].shape, (B, T, P, model.num_classes))
        self.assertEqual(out["confidence"].shape, (B, T, P))
        self.assertEqual(out["stage_logits"].shape[0], 2)

    def test_batch_includes_acceleration_and_time_features(self) -> None:
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])
        self.assertIsNotNone(batch.acceleration)
        self.assertEqual(batch.acceleration.shape, batch.velocity.shape)
        self.assertIsNotNone(batch.time_features)
        self.assertEqual(batch.time_features.shape, (1, 6, TIME_FEATURE_DIM))

    def test_model_runs_without_edge_features(self) -> None:
        torch.manual_seed(0)
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
            use_edge_features=False,
            use_acceleration=False,
            use_time_features=False,
        )
        out = model(batch)
        self.assertEqual(out["logits"].shape[-1], model.num_classes)

    def test_grad_flows_through_full_pipeline(self) -> None:
        torch.manual_seed(0)
        sample = stack_sample(_make_sample(num_steps=8))
        batch = stacked_to_batch([sample])
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        out = model(batch)
        loss = out["logits"].sum() + out["confidence"].sum()
        loss.backward()
        non_zero = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0.0
        )
        self.assertGreater(non_zero, 0)


class ZoneNodeModelTests(unittest.TestCase):
    def test_forward_with_zones_and_extras_grad_flows(self) -> None:
        torch.manual_seed(0)
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])
        model = PlayerCentricSpottingModel(
            hidden_dim=16,
            num_hgt_layers=2,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
            use_zone_nodes=True,
            zone_grid=(6, 4),
            use_jersey=True,
            use_goal_distances=True,
            use_radius_edges=True,
        )
        out = model(batch)
        loss = out["logits"].sum() + out["confidence"].sum()
        loss.backward()
        # Zone-specific weights must receive gradients.
        zone_param_grads = []
        for name, p in model.named_parameters():
            if "zone_value" in name or "player_count_proj" in name:
                if p.grad is not None:
                    zone_param_grads.append(float(p.grad.abs().sum()))
        self.assertTrue(zone_param_grads)
        self.assertGreater(sum(zone_param_grads), 0.0)

    def test_forward_with_all_extensions_off_matches_legacy_shape(self) -> None:
        torch.manual_seed(0)
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
            use_zone_nodes=False,
            use_jersey=False,
            use_goal_distances=False,
            use_radius_edges=False,
        )
        out = model(batch)
        self.assertEqual(out["logits"].shape[-1], model.num_classes)

    def test_batch_carries_left_to_right_and_shirt(self) -> None:
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])
        self.assertIsNotNone(batch.left_to_right)
        self.assertEqual(batch.left_to_right.shape, batch.valid_mask.shape)
        self.assertIsNotNone(batch.shirt_numbers)
        self.assertEqual(batch.shirt_numbers.shape, batch.teams.shape)


class GoalDistanceTests(unittest.TestCase):
    def test_distances_are_attacking_oriented(self) -> None:
        pitch_xy = torch.zeros(1, 1, 2, 2)
        # Player A at x=0.2, attacking +x (ltr=+1): own goal at x=0 -> 0.2;
        # opp goal at x=1 -> 0.8.
        pitch_xy[0, 0, 0] = torch.tensor([0.2, 0.3])
        # Player B at x=0.2, attacking -x (ltr=-1): own goal at x=1 -> 0.8;
        # opp goal at x=0 -> 0.2.
        pitch_xy[0, 0, 1] = torch.tensor([0.2, 0.7])
        ltr = torch.tensor([[[1.0, -1.0]]])
        gd = compute_goal_distances(pitch_xy, ltr)
        self.assertAlmostEqual(float(gd[0, 0, 0, 0]), 0.2, places=5)  # own
        self.assertAlmostEqual(float(gd[0, 0, 0, 1]), 0.8, places=5)  # opp
        self.assertAlmostEqual(float(gd[0, 0, 1, 0]), 0.8, places=5)
        self.assertAlmostEqual(float(gd[0, 0, 1, 1]), 0.2, places=5)
        # Nearest sideline is symmetric: min(y, 1-y).
        self.assertAlmostEqual(float(gd[0, 0, 0, 2]), 0.3, places=5)
        self.assertAlmostEqual(float(gd[0, 0, 1, 2]), 0.3, places=5)

    def test_extra_scalar_dim_constant(self) -> None:
        # The constant exists so docs/configs can refer to it; sanity check.
        self.assertEqual(EXTRA_SCALAR_DIM, 5)


class TimeAndAccelerationHelperTests(unittest.TestCase):
    def test_acceleration_is_finite_differences(self) -> None:
        vel = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)
        # vel: t=0 [0,1], t=1 [2,3], t=2 [4,5]
        acc = _compute_acceleration(vel)
        self.assertEqual(acc.shape, vel.shape)
        # t=0 zero-padded.
        self.assertTrue(torch.allclose(acc[:, 0], torch.zeros_like(acc[:, 0])))
        # t=1 -> [2,2], t=2 -> [2,2].
        self.assertTrue(torch.allclose(acc[:, 1], torch.full_like(acc[:, 1], 2.0)))

    def test_time_features_have_expected_dim(self) -> None:
        frames = torch.arange(8, dtype=torch.int64).unsqueeze(0)
        tf = _compute_time_features(frames, fps=25.0)
        self.assertEqual(tf.shape, (1, 8, TIME_FEATURE_DIM))
        # sin(0) == 0, cos(0) == 1 at frame 0 -> alternating zeros and ones.
        self.assertAlmostEqual(float(tf[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(tf[0, 0, 1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
