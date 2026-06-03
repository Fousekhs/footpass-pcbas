"""Tests for ``pcspot.losses.pc_calf``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.losses.pc_calf import PlayerAwareCalfLoss, pc_calf_loss


class PCCalfLossTests(unittest.TestCase):
    def test_zero_weights_yield_zero_loss(self) -> None:
        S, B, T, P, C = 2, 1, 4, 3, 5
        logits = torch.randn(S, B, T, P, C, requires_grad=True)
        targets = torch.zeros(B, T, P, C)
        weights = torch.zeros(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        out = pc_calf_loss(logits, targets, weights, valid, tmse_lambda=0.0)
        self.assertEqual(float(out.bce), 0.0)
        self.assertEqual(float(out.total), 0.0)
        self.assertEqual(float(out.objectness), 0.0)

    def test_loss_shapes_and_grads(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 2, 2, 6, 3, 4
        logits = torch.randn(S, B, T, P, C, requires_grad=True)
        targets = torch.zeros(B, T, P, C)
        targets[:, 2, 0, 1] = 1.0
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        out = pc_calf_loss(logits, targets, weights, valid, tmse_lambda=0.1)
        self.assertEqual(out.total.ndim, 0)
        out.total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_invalid_steps_excluded(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 1, 1, 4, 2, 2
        logits = torch.full((S, B, T, P, C), 5.0, requires_grad=True)
        targets = torch.zeros(B, T, P, C)
        weights = torch.ones(B, T, P, C)
        valid = torch.zeros(B, T, P, dtype=torch.bool)  # all invalid
        out = pc_calf_loss(logits, targets, weights, valid, tmse_lambda=0.0)
        self.assertEqual(float(out.bce), 0.0)


class PCCalfObjectnessTests(unittest.TestCase):
    def test_objectness_term_contributes_gradients(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 2, 1, 4, 3, 5
        logits = torch.zeros(S, B, T, P, C, requires_grad=True)
        targets = torch.zeros(B, T, P, C)
        weights = torch.zeros(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        # Only objectness signal exists.
        obj_logits = torch.zeros(S, B, T, P, requires_grad=True)
        obj_targets = torch.zeros(B, T, P)
        obj_targets[:, 2, 0] = 1.0
        obj_weights = torch.ones(B, T, P)
        out = pc_calf_loss(
            logits,
            targets,
            weights,
            valid,
            tmse_lambda=0.0,
            stage_objectness_logits=obj_logits,
            objectness_targets=obj_targets,
            objectness_weights=obj_weights,
            objectness_lambda=1.0,
        )
        self.assertGreater(float(out.objectness), 0.0)
        self.assertAlmostEqual(float(out.total), float(out.objectness), places=6)
        out.total.backward()
        self.assertIsNotNone(obj_logits.grad)
        self.assertGreater(float(obj_logits.grad.abs().sum()), 0.0)

    def test_objectness_off_by_default(self) -> None:
        S, B, T, P, C = 2, 1, 4, 3, 5
        logits = torch.zeros(S, B, T, P, C)
        targets = torch.zeros(B, T, P, C)
        targets[:, 2, 0, 1] = 1.0
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        out = pc_calf_loss(logits, targets, weights, valid, tmse_lambda=0.0)
        self.assertEqual(float(out.objectness), 0.0)
        self.assertGreater(float(out.bce), 0.0)


class PCCalfStageWeightingTests(unittest.TestCase):
    def test_uniform_vs_geometric_differ(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 3, 1, 4, 2, 3
        # Make later stages predict the target perfectly and earlier stages
        # predict completely wrong, so the weighting choice changes the loss
        # significantly.
        logits = torch.zeros(S, B, T, P, C)
        logits[0] = -10.0
        logits[-1] = 10.0
        targets = torch.ones(B, T, P, C)
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        loss_uniform = pc_calf_loss(
            logits, targets, weights, valid, tmse_lambda=0.0, stage_weights="uniform"
        )
        loss_geom = pc_calf_loss(
            logits, targets, weights, valid, tmse_lambda=0.0, stage_weights="geometric"
        )
        # Geometric weighting puts more mass on the last (correct) stage,
        # so its loss should be lower.
        self.assertLess(float(loss_geom.total), float(loss_uniform.total))

    def test_explicit_stage_weights_validation(self) -> None:
        S, B, T, P, C = 3, 1, 4, 2, 3
        logits = torch.zeros(S, B, T, P, C)
        targets = torch.zeros(B, T, P, C)
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        with self.assertRaises(ValueError):
            pc_calf_loss(
                logits, targets, weights, valid, stage_weights=[1.0, 1.0]
            )
        with self.assertRaises(ValueError):
            pc_calf_loss(
                logits, targets, weights, valid, stage_weights=[1.0, -1.0, 1.0]
            )


class PCCalfTMSEMaskTests(unittest.TestCase):
    def test_tmse_skips_zero_weight_pairs(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 1, 1, 6, 1, 1
        # Logits with a single huge spike between t=2 and t=3.
        logits = torch.zeros(S, B, T, P, C)
        logits[0, 0, 2, 0, 0] = -8.0
        logits[0, 0, 3, 0, 0] = 8.0
        targets = torch.zeros(B, T, P, C)
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        out_full = pc_calf_loss(
            logits, targets, weights, valid, tmse_lambda=1.0
        )
        # Now mask out the spike via the effective weight (e.g. uncertain-before).
        weights_masked = weights.clone()
        weights_masked[0, 2, 0, 0] = 0.0
        out_masked = pc_calf_loss(
            logits, targets, weights_masked, valid, tmse_lambda=1.0
        )
        self.assertLess(float(out_masked.tmse), float(out_full.tmse))


class PlayerAwareCalfLossModuleTests(unittest.TestCase):
    def test_module_forward_matches_function(self) -> None:
        torch.manual_seed(0)
        S, B, T, P, C = 2, 1, 4, 2, 3
        logits = torch.randn(S, B, T, P, C)
        targets = torch.zeros(B, T, P, C)
        targets[:, 2, 0, 1] = 1.0
        weights = torch.ones(B, T, P, C)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        obj_logits = torch.zeros(S, B, T, P)
        obj_targets = torch.zeros(B, T, P)
        obj_weights = torch.ones(B, T, P)

        module = PlayerAwareCalfLoss(
            tmse_lambda=0.1, objectness_lambda=0.5, stage_weights="uniform"
        )
        out_mod = module(
            logits,
            targets,
            weights,
            valid,
            stage_objectness_logits=obj_logits,
            objectness_targets=obj_targets,
            objectness_weights=obj_weights,
        )
        out_fn = pc_calf_loss(
            logits,
            targets,
            weights,
            valid,
            tmse_lambda=0.1,
            objectness_lambda=0.5,
            stage_weights="uniform",
            stage_objectness_logits=obj_logits,
            objectness_targets=obj_targets,
            objectness_weights=obj_weights,
        )
        self.assertAlmostEqual(float(out_mod.total), float(out_fn.total), places=6)


if __name__ == "__main__":
    unittest.main()
