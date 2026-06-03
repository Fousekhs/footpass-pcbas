"""Tests for ``pcspot.data.targets``."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

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
from pcspot.data.targets import (
    CalfConfig,
    build_objectness_targets,
    build_pc_calf_targets,
)


def _snap(
    pid: int,
    x: float = 0.5,
    y: float = 0.5,
    team: int | None = None,
) -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=pid,
        team=(0 if pid < 200 else 1) if team is None else team,
        shirt_number=pid % 100,
        role_id=1,
        x=x,
        y=y,
        speed_x=0.0,
        speed_y=0.0,
        bbox_xywh=(0.0, 0.0, 10.0, 10.0),
        visible=True,
    )


def _make_window(num_steps: int, players: list[int], events: list[EventLabel]) -> Sample:
    snaps_per_step = [[_snap(p) for p in players] for _ in range(num_steps)]
    return Sample(
        frames=np.arange(num_steps, dtype=np.int64),
        players_per_step=snaps_per_step,
        events=events,
        meta=SampleMeta(match_id="m"),
    )


class CalfTargetTests(unittest.TestCase):
    def test_no_events_zero_targets_unit_weights(self) -> None:
        sample = _make_window(num_steps=10, players=[101, 201], events=[])
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, num_classes=12)
        self.assertEqual(t.shape, (10, 2, 12))
        self.assertEqual(w.shape, (10, 2, 12))
        self.assertEqual(float(t.sum()), 0.0)
        self.assertEqual(float(w.min()), 1.0)

    def test_responsible_player_gets_positive_kernel(self) -> None:
        cfg = CalfConfig(k1_default=1, k2_default=2, positive_weight=4.0)
        sample = _make_window(
            num_steps=8,
            players=[101, 201],
            events=[EventLabel(frame=4, player_id=201, class_id=2)],
        )
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        target_pass_p_resp = t[:, 1, 1]
        self.assertEqual(float(target_pass_p_resp[4]), 1.0)
        self.assertEqual(float(target_pass_p_resp[5]), 1.0)
        self.assertAlmostEqual(float(target_pass_p_resp[6]), 0.5, places=5)
        self.assertAlmostEqual(float(target_pass_p_resp[7]), 0.0, places=5)
        self.assertEqual(float(target_pass_p_resp[2]), 0.0)
        self.assertEqual(float(target_pass_p_resp[3]), 0.0)
        self.assertEqual(float(w[2, 1, 1]), 0.0)
        self.assertEqual(float(w[3, 1, 1]), 0.0)
        self.assertAlmostEqual(float(w[4, 1, 1]), 4.0, places=5)
        self.assertAlmostEqual(float(w[5, 1, 1]), 4.0, places=5)

    def test_other_player_target_stays_zero(self) -> None:
        cfg = CalfConfig(k1_default=1, k2_default=2)
        sample = _make_window(
            num_steps=6,
            players=[101, 201],
            events=[EventLabel(frame=3, player_id=101, class_id=4)],
        )
        stacked = stack_sample(sample)
        t, _ = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        self.assertEqual(float(t[:, 1, :].sum()), 0.0)

    def test_invalid_columns_get_zero_weight(self) -> None:
        snaps = [
            [_snap(101)],
            [_snap(101), _snap(201)],
            [_snap(101)],
            [_snap(101)],
        ]
        sample = Sample(
            frames=np.arange(4, dtype=np.int64),
            players_per_step=snaps,
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        _, w = build_pc_calf_targets(stacked, num_classes=12)
        self.assertEqual(float(w[0, 1, :].max()), 0.0)
        self.assertEqual(float(w[2, 1, :].max()), 0.0)


class CalfHardCutoffTests(unittest.TestCase):
    def test_hard_cutoff_preserves_legacy_behavior(self) -> None:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.2,
            ambiguity_radius=0.5,
            distance_falloff="hard",
        )
        snaps = [
            [_snap(101, x=0.5, y=0.5), _snap(201, x=0.51, y=0.5)]
            for _ in range(6)
        ]
        sample = Sample(
            frames=np.arange(6, dtype=np.int64),
            players_per_step=snaps,
            events=[EventLabel(frame=3, player_id=101, class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        _, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        # Player 201 is an opponent for class=2 (Pass, attacking) so team
        # factor is 1.0. Distance 0.01 < radius 0.5 -> hard floor 0.2.
        nearby_window = w[:, 1, 1]
        self.assertLessEqual(float(nearby_window.max()), 0.2 + 1e-6)


class CalfGaussianFalloffTests(unittest.TestCase):
    def _run_with_distance(self, distance: float, *, sigma: float = 0.1) -> float:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.3,
            ambiguity_radius=sigma,
            gaussian_sigma=sigma,
            distance_falloff="gaussian",
        )
        snaps = [
            [
                _snap(101, x=0.5, y=0.5, team=0),
                _snap(201, x=0.5 + distance, y=0.5, team=1),
            ]
            for _ in range(6)
        ]
        sample = Sample(
            frames=np.arange(6, dtype=np.int64),
            players_per_step=snaps,
            events=[EventLabel(frame=3, player_id=101, class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        _, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        # Sample the weight at the event time for the nearby player and class.
        return float(w[3, 1, 1])

    def test_weight_increases_with_distance(self) -> None:
        w_close = self._run_with_distance(0.0)
        w_mid = self._run_with_distance(0.1)
        w_far = self._run_with_distance(1.0)
        self.assertLess(w_close, w_mid)
        self.assertLess(w_mid, w_far)
        self.assertAlmostEqual(w_close, 0.3, places=5)
        # At d >> sigma we should approach 1.0.
        self.assertGreater(w_far, 0.99)

    def test_weight_matches_gaussian_formula(self) -> None:
        d = 0.05
        sigma = 0.1
        floor = 0.3
        expected = floor + (1.0 - floor) * (1.0 - math.exp(-(d * d) / (2.0 * sigma * sigma)))
        got = self._run_with_distance(d, sigma=sigma)
        self.assertAlmostEqual(got, expected, places=5)


class CalfTeamPolicyTests(unittest.TestCase):
    def _stack(self, players: list[tuple[int, int, float]]) -> tuple:
        snaps = [
            [_snap(pid, x=x, y=0.5, team=team) for pid, team, x in players]
            for _ in range(6)
        ]
        sample = Sample(
            frames=np.arange(6, dtype=np.int64),
            players_per_step=snaps,
            events=[EventLabel(frame=3, player_id=players[0][0], class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        return stack_sample(sample)

    def test_attacking_class_softens_teammate_not_opponent(self) -> None:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.0,  # remove distance term to isolate team factor
            ambiguity_radius=1e-6,
            gaussian_sigma=1e-6,
            teammate_floor=0.4,
            opponent_floor=0.4,
        )
        # Two teammates and one opponent, all reasonably far away so
        # distance term is ~1.0.
        stacked = self._stack(
            players=[
                (101, 0, 0.50),  # actor
                (102, 0, 0.30),  # teammate
                (201, 1, 0.30),  # opponent
            ]
        )
        _, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        # Class 2 (Pass) is attacking -> teammate gets teammate_floor, opponent stays 1.0.
        w_teammate = float(w[3, 1, 1])
        w_opponent = float(w[3, 2, 1])
        self.assertAlmostEqual(w_teammate, 0.4, places=2)
        self.assertAlmostEqual(w_opponent, 1.0, places=2)

    def test_duel_class_softens_opponent_not_teammate(self) -> None:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.0,
            ambiguity_radius=1e-6,
            gaussian_sigma=1e-6,
            teammate_floor=0.4,
            opponent_floor=0.4,
        )
        stacked = self._stack(
            players=[
                (101, 0, 0.50),
                (102, 0, 0.30),
                (201, 1, 0.30),
            ]
        )
        # Override the event class to class 7 (Tackle).
        stacked.targets_class[:] = 0
        stacked.targets_class[3, 0] = 7
        _, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        w_teammate = float(w[3, 1, 6])  # class_id 7 -> index 6
        w_opponent = float(w[3, 2, 6])
        self.assertAlmostEqual(w_teammate, 1.0, places=2)
        self.assertAlmostEqual(w_opponent, 0.4, places=2)


class CalfPositiveProtectionTests(unittest.TestCase):
    def test_nearby_overlapping_event_does_not_clobber_plateau(self) -> None:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.1,
            ambiguity_radius=0.5,
            gaussian_sigma=0.5,
            distance_falloff="gaussian",
            teammate_floor=0.1,
            opponent_floor=0.1,
            ambiguity_time_scale=2.0,
        )
        # Two same-team players, both performing the same class within
        # overlapping windows, standing right next to each other.
        snaps = [
            [
                _snap(101, x=0.50, y=0.5, team=0),
                _snap(102, x=0.51, y=0.5, team=0),
            ]
            for _ in range(8)
        ]
        sample = Sample(
            frames=np.arange(8, dtype=np.int64),
            players_per_step=snaps,
            events=[
                EventLabel(frame=3, player_id=101, class_id=2),
                EventLabel(frame=4, player_id=102, class_id=2),
            ],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        # Player 101's plateau at t=3 must remain at positive_weight=4.0
        # even though event 2 (on 102) would, under the old code, have
        # clobbered it down to ambiguity_weight=0.1.
        self.assertEqual(float(t[3, 0, 1]), 1.0)
        self.assertAlmostEqual(float(w[3, 0, 1]), 4.0, places=5)
        # Player 102's plateau at t=4 must also remain at 4.0.
        self.assertEqual(float(t[4, 1, 1]), 1.0)
        self.assertAlmostEqual(float(w[4, 1, 1]), 4.0, places=5)


class CalfPerClassOverridesTests(unittest.TestCase):
    def test_per_class_window_and_positive_weight(self) -> None:
        cfg = CalfConfig(
            k1_default=2,
            k2_default=4,
            positive_weight=4.0,
            per_class_window={6: (5, 10)},  # Throw-in
            per_class_positive_weight={6: 8.0},
            ambiguity_weight=0.0,
        )
        sample = _make_window(
            num_steps=20,
            players=[101],
            events=[EventLabel(frame=10, player_id=101, class_id=6)],
        )
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        c_idx = 5  # class_id 6 -> index 5
        # K1=5 -> plateau covers t in [10, 15].
        for tt in range(10, 16):
            self.assertEqual(float(t[tt, 0, c_idx]), 1.0)
            self.assertAlmostEqual(float(w[tt, 0, c_idx]), 8.0, places=5)
        # K1+K2=15 -> decay tail up to t=25; t=15 still plateau, t=18 mid-decay.
        # T=20 so indices stop at 19; t=18 has delta=8, decay_delta=3, target=1-3/10=0.7.
        self.assertGreater(float(t[18, 0, c_idx]), 0.0)


class ObjectnessTargetTests(unittest.TestCase):
    def test_objectness_targets_and_weights(self) -> None:
        cfg = CalfConfig(k1_default=1, k2_default=2, positive_weight=4.0)
        sample = _make_window(
            num_steps=8,
            players=[101],
            events=[EventLabel(frame=4, player_id=101, class_id=2)],
        )
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        obj_t, obj_w = build_objectness_targets(t, w)
        self.assertEqual(obj_t.shape, (8, 1))
        self.assertEqual(obj_w.shape, (8, 1))
        # Plateau positions become positive objectness.
        self.assertEqual(float(obj_t[4, 0]), 1.0)
        self.assertEqual(float(obj_t[5, 0]), 1.0)
        # Plateau weight is the max-over-classes ramp from the actor's class
        # which is the per-class positive weight (4.0 here).
        self.assertGreater(float(obj_w[4, 0]), 1.0)
        # Uncertain-before zone propagates to objectness via min-over-classes
        # weight: the actor's active class has weight 0 there, so objectness
        # weight is 0 (ignored).
        self.assertEqual(float(obj_w[2, 0]), 0.0)
        self.assertEqual(float(obj_w[3, 0]), 0.0)
        # Far-before frames keep weight 1 (no class is in ignore).
        self.assertEqual(float(obj_w[0, 0]), 1.0)

    def test_objectness_for_non_actor_is_supervised(self) -> None:
        cfg = CalfConfig(
            k1_default=1,
            k2_default=2,
            positive_weight=4.0,
            ambiguity_weight=0.5,
            ambiguity_radius=0.1,
        )
        # Two players; 101 performs the event, 201 is an opponent.
        snaps = [
            [_snap(101, x=0.5, team=0), _snap(201, x=0.9, team=1)]
            for _ in range(6)
        ]
        sample = Sample(
            frames=np.arange(6, dtype=np.int64),
            players_per_step=snaps,
            events=[EventLabel(frame=3, player_id=101, class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        t, w = build_pc_calf_targets(stacked, config=cfg, num_classes=12)
        obj_t, obj_w = build_objectness_targets(t, w)
        # 201 (column 1) is far from the actor, so every class weight there is
        # ~1.0 and objectness weight is also ~1.0 throughout the window.
        self.assertGreaterEqual(float(obj_w[:, 1].min()), 0.99)
        # And 201 never has a positive objectness target.
        self.assertEqual(float(obj_t[:, 1].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
