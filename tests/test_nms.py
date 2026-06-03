"""Tests for ``pcspot.eval.nms``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.eval.nms import (
    Prediction,
    decode_predictions,
    player_centric_nms,
)


class DecodeTests(unittest.TestCase):
    def test_extracts_local_maxima(self) -> None:
        T, P, C = 5, 1, 2
        # Class 0: scores 0.1, 0.4, 0.9, 0.4, 0.1 -> peak at t=2
        # Class 1: flat 0.5
        logits = torch.full((T, P, C), -10.0)
        logits[2, 0, 0] = 5.0
        logits[1, 0, 0] = 0.0
        logits[3, 0, 0] = 0.0
        logits[:, 0, 1] = 0.0
        valid = np.ones((T, P), dtype=bool)
        preds = decode_predictions(
            logits, confidence=None, valid_mask=valid,
            player_ids=np.asarray([42], dtype=np.int64),
            score_threshold=0.5,
        )
        cls0 = [p for p in preds if p.class_id == 1]
        self.assertEqual(len(cls0), 1)
        self.assertEqual(cls0[0].time, 2)
        self.assertEqual(cls0[0].player_id, 42)


class NMSTests(unittest.TestCase):
    def test_keeps_best_within_radius(self) -> None:
        preds = [
            Prediction(time=10, class_id=1, player_id=101, score=0.9),
            Prediction(time=11, class_id=1, player_id=101, score=0.6),
            Prediction(time=20, class_id=1, player_id=101, score=0.7),
        ]
        kept = player_centric_nms(preds, window_radius=3)
        # Two should remain: t=10 (best) and t=20 (separated).
        self.assertEqual(len(kept), 2)
        times = sorted(p.time for p in kept)
        self.assertEqual(times, [10, 20])

    def test_independent_per_player_and_class(self) -> None:
        preds = [
            Prediction(time=5, class_id=1, player_id=101, score=0.9),
            Prediction(time=6, class_id=1, player_id=201, score=0.8),
            Prediction(time=5, class_id=2, player_id=101, score=0.7),
        ]
        kept = player_centric_nms(preds, window_radius=3)
        # All three are different (class, player) pairs -> all kept.
        self.assertEqual(len(kept), 3)

    def test_cross_class_per_player_collapses_classes(self) -> None:
        preds = [
            Prediction(time=5, class_id=1, player_id=101, score=0.9),
            Prediction(time=5, class_id=2, player_id=101, score=0.7),
            Prediction(time=5, class_id=3, player_id=201, score=0.4),
        ]
        kept = player_centric_nms(preds, window_radius=3, mode="per_player")
        # Player 101 should keep only the top scoring (class 1).
        # Player 201 keeps its single prediction.
        self.assertEqual(len(kept), 2)
        kept_pids = sorted((p.player_id, p.class_id) for p in kept)
        self.assertEqual(kept_pids, [(101, 1), (201, 3)])

    def test_per_class_collapses_players(self) -> None:
        preds = [
            Prediction(time=5, class_id=1, player_id=101, score=0.9),
            Prediction(time=6, class_id=1, player_id=201, score=0.8),
            Prediction(time=20, class_id=1, player_id=301, score=0.7),
        ]
        kept = player_centric_nms(preds, window_radius=3, mode="per_class")
        # All three are class 1; the second one (within 1 frame of the
        # first) is suppressed; the third is kept (separated).
        self.assertEqual(len(kept), 2)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            player_centric_nms([], window_radius=1, mode="bogus")


if __name__ == "__main__":
    unittest.main()
