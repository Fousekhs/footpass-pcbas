"""Tests for visual feature fusion in the embedder, batch, and model."""

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
from pcspot.models.graph import PlayerNodeEmbedder
from pcspot.models.pipeline import (
    PlayerCentricSpottingModel,
    stacked_to_batch,
)


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


class EmbedderVisualTests(unittest.TestCase):
    def test_visual_features_change_output(self) -> None:
        torch.manual_seed(0)
        B, T, P = 1, 3, 4
        D = 16
        F_visual = 8
        embed = PlayerNodeEmbedder(
            hidden_dim=D,
            visual_dim=F_visual,
            visual_proj_dim=4,
            use_acceleration=False,
        )
        pitch_xy = torch.rand(B, T, P, 2)
        velocity = torch.zeros(B, T, P, 2)
        bbox = torch.zeros(B, T, P, 4)
        roles = torch.randint(1, 14, (B, T, P), dtype=torch.long)
        teams = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        # Use non-constant tensors: a constant tensor (like all-zeros or
        # all-ones) is collapsed to zero by LayerNorm regardless of its
        # value, so two different constants would produce identical
        # outputs and not exercise the visual branch.
        v_a = torch.zeros(B, T, P, F_visual)
        v_b = torch.randn(B, T, P, F_visual)

        h_a = embed(pitch_xy, velocity, bbox, roles, teams, visual_features=v_a)
        h_b = embed(pitch_xy, velocity, bbox, roles, teams, visual_features=v_b)
        self.assertGreater(float((h_a - h_b).abs().sum()), 0.0)

    def test_visual_required_when_dim_set(self) -> None:
        embed = PlayerNodeEmbedder(hidden_dim=8, visual_dim=4)
        with self.assertRaises(ValueError):
            embed(
                pitch_xy=torch.zeros(1, 1, 1, 2),
                velocity=torch.zeros(1, 1, 1, 2),
                bbox=torch.zeros(1, 1, 1, 4),
                roles=torch.zeros(1, 1, 1, dtype=torch.long),
                teams=torch.zeros(1, 1, dtype=torch.long),
                visual_features=None,
            )

    def test_visual_dim_zero_ignores_visual_features(self) -> None:
        embed = PlayerNodeEmbedder(hidden_dim=8, visual_dim=0)
        # Passing visual_features when disabled should not crash.
        pitch_xy = torch.zeros(1, 1, 2, 2)
        out = embed(
            pitch_xy=pitch_xy,
            velocity=torch.zeros_like(pitch_xy),
            bbox=torch.zeros(1, 1, 2, 4),
            roles=torch.zeros(1, 1, 2, dtype=torch.long),
            teams=torch.zeros(1, 2, dtype=torch.long),
            visual_features=torch.ones(1, 1, 2, 7),  # ignored
        )
        self.assertEqual(out.shape, (1, 1, 2, 8))

    def test_invalid_visual_shape_raises(self) -> None:
        embed = PlayerNodeEmbedder(hidden_dim=8, visual_dim=4)
        pitch_xy = torch.zeros(1, 2, 3, 2)
        with self.assertRaises(ValueError):
            embed(
                pitch_xy=pitch_xy,
                velocity=torch.zeros_like(pitch_xy),
                bbox=torch.zeros(1, 2, 3, 4),
                roles=torch.zeros(1, 2, 3, dtype=torch.long),
                teams=torch.zeros(1, 3, dtype=torch.long),
                visual_features=torch.zeros(1, 2, 3, 5),  # F mismatch
            )


class BatchPaddingTests(unittest.TestCase):
    def test_visual_features_are_padded_in_batch(self) -> None:
        F_visual = 6
        sample_a = stack_sample(_make_sample(num_steps=4))
        sample_b = stack_sample(_make_sample(num_steps=4, players=(101, 201)))
        # Attach mock visual features (T, P, F_visual).
        feats_a = np.random.randn(4, sample_a.num_players, F_visual).astype(np.float32)
        feats_b = np.random.randn(4, sample_b.num_players, F_visual).astype(np.float32)
        sample_a = attach_visual_features(sample_a, feats_a)
        sample_b = attach_visual_features(sample_b, feats_b)

        batch = stacked_to_batch([sample_a, sample_b])
        self.assertIsNotNone(batch.visual_features)
        assert batch.visual_features is not None
        self.assertEqual(
            batch.visual_features.shape,
            (2, 4, sample_a.num_players, F_visual),
        )
        # Padded columns (sample_b has fewer players) should be zero.
        pad_cols = sample_a.num_players - sample_b.num_players
        if pad_cols > 0:
            self.assertEqual(
                int(batch.visual_features[1, :, sample_b.num_players :].abs().sum()),
                0,
            )

    def test_visual_dim_mismatch_raises(self) -> None:
        s1 = stack_sample(_make_sample(num_steps=3))
        s2 = stack_sample(_make_sample(num_steps=3))
        s1 = attach_visual_features(
            s1, np.zeros((3, s1.num_players, 4), dtype=np.float32)
        )
        s2 = attach_visual_features(
            s2, np.zeros((3, s2.num_players, 5), dtype=np.float32)
        )
        with self.assertRaises(ValueError):
            stacked_to_batch([s1, s2])


class FullModelTests(unittest.TestCase):
    def test_forward_with_visual_features(self) -> None:
        torch.manual_seed(0)
        F_visual = 8
        sample = stack_sample(_make_sample(num_steps=6))
        feats = np.random.randn(6, sample.num_players, F_visual).astype(np.float32)
        sample = attach_visual_features(sample, feats)
        batch = stacked_to_batch([sample])

        model = PlayerCentricSpottingModel(
            hidden_dim=16,
            visual_dim=F_visual,
            visual_proj_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
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

        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            visual_dim=F_visual,
            visual_proj_dim=4,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        out = model(batch)
        out["logits"].sum().backward()
        # The visual projection layer must receive gradient.
        proj_layer = model.embedder.visual_proj
        any_grad = False
        for p in proj_layer.parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0.0:
                any_grad = True
                break
        self.assertTrue(any_grad)


if __name__ == "__main__":
    unittest.main()
