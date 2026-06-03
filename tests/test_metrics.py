"""Tests for ``pcspot.eval.metrics``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.schema import EventLabel
from pcspot.eval.metrics import (
    average_map_at_tolerances,
    match_predictions,
    player_identity_accuracy,
)
from pcspot.eval.nms import Prediction


class MatchTests(unittest.TestCase):
    def test_class_only_matches_within_tolerance(self) -> None:
        gts = [EventLabel(frame=10, player_id=101, class_id=1)]
        preds = [Prediction(time=12, class_id=1, player_id=999, score=0.9)]
        scores, is_tp, n_gt = match_predictions(preds, gts, tolerance=3)
        self.assertEqual(n_gt, 1)
        self.assertTrue(bool(is_tp[0]))

    def test_player_required_matching_is_stricter(self) -> None:
        gts = [EventLabel(frame=10, player_id=101, class_id=1)]
        preds = [
            Prediction(time=10, class_id=1, player_id=999, score=0.9),
            Prediction(time=10, class_id=1, player_id=101, score=0.8),
        ]
        # Class-only: prediction with score 0.9 wins; that's a TP.
        _, is_tp_co, _ = match_predictions(preds, gts, tolerance=2)
        self.assertTrue(bool(is_tp_co[0]))
        # Player-required: only the second prediction matches; first is FP.
        _, is_tp_pl, _ = match_predictions(
            preds, gts, tolerance=2, require_player=True
        )
        self.assertFalse(bool(is_tp_pl[0]))
        self.assertTrue(bool(is_tp_pl[1]))


class IdentityAccuracyTests(unittest.TestCase):
    def test_identity_accuracy_basic(self) -> None:
        gts = [
            EventLabel(frame=10, player_id=101, class_id=1),
            EventLabel(frame=20, player_id=201, class_id=1),
        ]
        preds = [
            Prediction(time=10, class_id=1, player_id=101, score=0.9),  # correct
            Prediction(time=20, class_id=1, player_id=999, score=0.8),  # wrong player
        ]
        acc, matched = player_identity_accuracy(preds, gts, tolerance=2)
        self.assertEqual(matched, 2)
        self.assertAlmostEqual(acc, 0.5, places=5)


class AverageMapTests(unittest.TestCase):
    def test_perfect_predictions_yield_high_map(self) -> None:
        gts = [
            EventLabel(frame=10, player_id=101, class_id=1),
            EventLabel(frame=30, player_id=101, class_id=1),
        ]
        preds = [
            Prediction(time=10, class_id=1, player_id=101, score=0.99),
            Prediction(time=30, class_id=1, player_id=101, score=0.95),
        ]
        summaries = average_map_at_tolerances(
            preds, gts, tolerances=[2], class_ids=[1, 2]
        )
        self.assertEqual(len(summaries), 1)
        s = summaries[0]
        self.assertGreater(s.average_map, 0.9)
        self.assertGreater(s.average_map_joint, 0.9)

    def test_wrong_player_kills_joint_map_only(self) -> None:
        gts = [EventLabel(frame=10, player_id=101, class_id=1)]
        preds = [Prediction(time=10, class_id=1, player_id=999, score=0.99)]
        summaries = average_map_at_tolerances(
            preds, gts, tolerances=[2], class_ids=[1]
        )
        s = summaries[0]
        self.assertGreater(s.average_map, 0.9)
        self.assertLess(s.average_map_joint, 0.1)


class APInterpolationTests(unittest.TestCase):
    def test_interpolation_modes_produce_finite_numbers(self) -> None:
        gts = [
            EventLabel(frame=10, player_id=101, class_id=1),
            EventLabel(frame=30, player_id=101, class_id=1),
            EventLabel(frame=50, player_id=101, class_id=1),
        ]
        preds = [
            Prediction(time=10, class_id=1, player_id=101, score=0.99),
            Prediction(time=12, class_id=1, player_id=101, score=0.8),  # FP
            Prediction(time=30, class_id=1, player_id=101, score=0.95),
            Prediction(time=50, class_id=1, player_id=101, score=0.4),
        ]
        results = {}
        for mode in ("11-point", "101-point", "continuous"):
            s = average_map_at_tolerances(
                preds, gts, tolerances=[2], class_ids=[1], ap_interpolation=mode
            )[0]
            results[mode] = s.average_map
            self.assertGreaterEqual(s.average_map, 0.0)
            self.assertLessEqual(s.average_map, 1.0)
        # 101-point and continuous tend to be similar but differ from
        # 11-point on partial-recall curves; just verify finite differences.
        self.assertNotEqual(results["11-point"], results["continuous"])

    def test_unknown_interpolation_raises(self) -> None:
        gts = [EventLabel(frame=10, player_id=101, class_id=1)]
        preds = [Prediction(time=10, class_id=1, player_id=101, score=0.5)]
        with self.assertRaises(ValueError):
            average_map_at_tolerances(
                preds, gts, tolerances=[2], class_ids=[1], ap_interpolation="bad"
            )

    def test_perfect_predictions_give_ap_one_in_all_modes(self) -> None:
        gts = [
            EventLabel(frame=10, player_id=101, class_id=1),
            EventLabel(frame=30, player_id=101, class_id=1),
        ]
        preds = [
            Prediction(time=10, class_id=1, player_id=101, score=0.99),
            Prediction(time=30, class_id=1, player_id=101, score=0.95),
        ]
        for mode in ("11-point", "101-point", "continuous"):
            s = average_map_at_tolerances(
                preds, gts, tolerances=[2], class_ids=[1], ap_interpolation=mode
            )[0]
            self.assertAlmostEqual(s.average_map, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
