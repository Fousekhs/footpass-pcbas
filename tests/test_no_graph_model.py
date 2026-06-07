"""Tests for the graph-ablation :class:`NoGraphSpottingModel`.

Mirrors the shape assertions in ``tests/test_pipeline.py`` /
``tests/test_visual_fusion.py`` for the graph model, plus a check that
the no-graph variant genuinely has no HGT / inter-player message
passing — that's the entire point of the ablation.
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

from pcspot.data.schema import (
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
    attach_visual_features,
    stack_sample,
)
from pcspot.models.no_graph_model import NoGraphSpottingModel
from pcspot.models.pipeline import stacked_to_batch


def _make_sample(
    num_steps: int = 6,
    players: tuple[int, ...] = (101, 102, 201, 202),
) -> Sample:
    snaps_per_step = []
    for t in range(num_steps):
        step = []
        for pid in players:
            step.append(
                PlayerSnapshot(
                    player_id=pid,
                    team=0 if pid < 200 else 1,
                    shirt_number=pid % 100,
                    role_id=1 + (pid % 5),
                    x=0.1 * (pid % 5),
                    y=0.1 * (pid % 7),
                    speed_x=0.0,
                    speed_y=0.0,
                    bbox_xywh=(10.0, 20.0, 30.0, 40.0),
                    visible=True,
                )
            )
        snaps_per_step.append(step)
    return Sample(
        frames=np.arange(num_steps, dtype=np.int64),
        players_per_step=snaps_per_step,
        events=[EventLabel(frame=num_steps // 2, player_id=players[0], class_id=2)],
        meta=SampleMeta(match_id="m", half_id="h"),
    )


class NoGraphModelTests(unittest.TestCase):
    def test_has_no_hgt(self) -> None:
        model = NoGraphSpottingModel(hidden_dim=8, num_mstcn_stages=1, num_mstcn_layers=2)
        self.assertFalse(hasattr(model, "hgt"))
        for name, _ in model.named_modules():
            self.assertNotIn("hgt", name.lower())

    def test_forward_kinematic_only(self) -> None:
        torch.manual_seed(0)
        sample = stack_sample(_make_sample(num_steps=6))
        batch = stacked_to_batch([sample])

        model = NoGraphSpottingModel(hidden_dim=16, num_mstcn_stages=1, num_mstcn_layers=2)
        out = model(batch)

        B, T, P = 1, 6, sample.num_players
        self.assertEqual(out["logits"].shape, (B, T, P, model.num_classes))
        self.assertEqual(out["stage_logits"][0].shape, (B, T, P, model.num_classes))
        self.assertIn("confidence", out)
        self.assertEqual(out["confidence"].shape, (B, T, P))
        self.assertIn("stage_confidence", out)

    def test_forward_with_visual_features(self) -> None:
        torch.manual_seed(0)
        F_visual = 8
        sample = stack_sample(_make_sample(num_steps=6))
        feats = np.random.randn(6, sample.num_players, F_visual).astype(np.float32)
        sample = attach_visual_features(sample, feats)
        batch = stacked_to_batch([sample])

        model = NoGraphSpottingModel(
            hidden_dim=16,
            visual_dim=F_visual,
            visual_proj_dim=8,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
        )
        out = model(batch)
        self.assertEqual(out["logits"].shape, (1, 6, sample.num_players, model.num_classes))

    def test_grad_flows_through_visual_branch(self) -> None:
        torch.manual_seed(0)
        F_visual = 8
        sample = stack_sample(_make_sample(num_steps=4))
        feats = np.random.randn(4, sample.num_players, F_visual).astype(np.float32)
        sample = attach_visual_features(sample, feats)
        batch = stacked_to_batch([sample])

        model = NoGraphSpottingModel(
            hidden_dim=8,
            visual_dim=F_visual,
            visual_proj_dim=4,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
        )
        out = model(batch)
        out["logits"].sum().backward()
        proj_layer = model.embedder.visual_proj
        any_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in proj_layer.parameters()
        )
        self.assertTrue(any_grad)

    def test_no_goal_distances_drops_extra_scalars(self) -> None:
        with_dist = NoGraphSpottingModel(hidden_dim=8, num_mstcn_stages=1, num_mstcn_layers=2,
                                          use_goal_distances=True)
        without_dist = NoGraphSpottingModel(hidden_dim=8, num_mstcn_stages=1, num_mstcn_layers=2,
                                             use_goal_distances=False)
        self.assertEqual(with_dist.extra_scalar_dim, 3)
        self.assertEqual(without_dist.extra_scalar_dim, 0)


if __name__ == "__main__":
    unittest.main()
