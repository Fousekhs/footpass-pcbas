"""Tests for ``pcspot.models.mstcn``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.models.mstcn import PlayerMSTCN


class PlayerMSTCNTests(unittest.TestCase):
    def test_forward_per_stage_shapes(self) -> None:
        torch.manual_seed(0)
        B, T, P, D = 2, 16, 4, 8
        h = torch.randn(B, T, P, D)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        valid[:, -2:, -1] = False
        model = PlayerMSTCN(hidden_dim=D, num_stages=3, num_layers_per_stage=4)
        out = model(h, valid=valid)
        self.assertEqual(len(out.stages), 3)
        for s in out.stages:
            self.assertEqual(s.shape, h.shape)

    def test_padded_steps_are_zeroed(self) -> None:
        torch.manual_seed(0)
        B, T, P, D = 1, 8, 2, 4
        h = torch.randn(B, T, P, D)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        valid[:, 4:, :] = False
        model = PlayerMSTCN(hidden_dim=D, num_stages=2, num_layers_per_stage=2)
        out = model(h, valid=valid)
        for s in out.stages:
            self.assertTrue(torch.allclose(s[:, 4:, :, :], torch.zeros_like(s[:, 4:, :, :])))

    def test_dim_mismatch_raises(self) -> None:
        model = PlayerMSTCN(hidden_dim=16, num_stages=1, num_layers_per_stage=2)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4, 2, 8))


if __name__ == "__main__":
    unittest.main()
