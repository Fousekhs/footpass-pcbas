"""Tests for ``pcspot.data.schema``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.schema import (
    NUM_PCBAS_CLASSES,
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
    attach_visual_features,
    stack_sample,
)


def _snap(
    pid: int,
    team: int = 0,
    x: float = 0.5,
    y: float = 0.5,
    *,
    left_to_right: float = 1.0,
    shirt_number: int | None = None,
) -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=pid,
        team=team,
        shirt_number=pid % 100 if shirt_number is None else shirt_number,
        role_id=1,
        x=x,
        y=y,
        speed_x=0.0,
        speed_y=0.0,
        bbox_xywh=(0.0, 0.0, 10.0, 20.0),
        visible=True,
        left_to_right=left_to_right,
    )


class StackSampleTests(unittest.TestCase):
    def test_columns_are_stable_across_steps(self) -> None:
        frames = np.arange(3)
        sample = Sample(
            frames=frames,
            players_per_step=[
                [_snap(101), _snap(201, team=1)],
                [_snap(201, team=1), _snap(101)],  # different order
                [_snap(101)],  # 201 disappears
            ],
            events=[],
            meta=SampleMeta(match_id="m"),
        )

        stacked = stack_sample(sample)

        self.assertEqual(stacked.num_steps, 3)
        self.assertEqual(stacked.num_players, 2)
        # Column ordering follows first-appearance, so 101 -> 0, 201 -> 1.
        self.assertEqual(int(stacked.player_ids[0]), 101)
        self.assertEqual(int(stacked.player_ids[1]), 201)
        # Player 201 is missing on the last step and must be masked out.
        self.assertTrue(stacked.valid_mask[0, 1])
        self.assertTrue(stacked.valid_mask[1, 1])
        self.assertFalse(stacked.valid_mask[2, 1])
        self.assertEqual(int(stacked.teams[1]), 1)

    def test_targets_align_with_events(self) -> None:
        frames = np.array([10, 11, 12])
        sample = Sample(
            frames=frames,
            players_per_step=[
                [_snap(101), _snap(201, team=1)],
                [_snap(101), _snap(201, team=1)],
                [_snap(101), _snap(201, team=1)],
            ],
            events=[
                EventLabel(frame=11, player_id=201, class_id=2),  # Pass
                EventLabel(frame=12, player_id=101, class_id=4),  # Shot
            ],
            meta=SampleMeta(match_id="m"),
        )

        stacked = stack_sample(sample)

        self.assertEqual(int(stacked.targets_class[1, 1]), 2)
        self.assertEqual(int(stacked.targets_class[2, 0]), 4)

        onehot = stacked.per_step_event_targets()
        self.assertEqual(onehot.shape, (3, 2, NUM_PCBAS_CLASSES))
        # Pass class id 2 -> column index 1 (background excluded).
        self.assertEqual(onehot[1, 1, 1], 1.0)
        self.assertEqual(onehot[2, 0, 3], 1.0)
        self.assertAlmostEqual(onehot.sum(), 2.0)

    def test_event_for_unseen_player_is_dropped(self) -> None:
        frames = np.array([0, 1])
        sample = Sample(
            frames=frames,
            players_per_step=[[_snap(101)], [_snap(101)]],
            events=[EventLabel(frame=1, player_id=999, class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertEqual(int(stacked.targets_class.sum()), 0)

    def test_event_outside_window_is_dropped(self) -> None:
        frames = np.array([5, 6, 7])
        sample = Sample(
            frames=frames,
            players_per_step=[[_snap(101)]] * 3,
            events=[EventLabel(frame=99, player_id=101, class_id=2)],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertEqual(int(stacked.targets_class.sum()), 0)

    def test_nan_bbox_is_zeroed(self) -> None:
        frames = np.array([0])
        snap = PlayerSnapshot(
            player_id=101,
            team=0,
            shirt_number=10,
            role_id=12,
            x=0.5,
            y=0.5,
            speed_x=0.0,
            speed_y=0.0,
            bbox_xywh=(float("nan"),) * 4,
            visible=False,
        )
        sample = Sample(
            frames=frames,
            players_per_step=[[snap]],
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertTrue(np.all(stacked.bbox_xywh[0, 0] == 0.0))
        self.assertFalse(bool(stacked.visible[0, 0]))
        # Still counts as a valid column position because the row exists.
        self.assertTrue(bool(stacked.valid_mask[0, 0]))

    def test_global_features_validation(self) -> None:
        frames = np.arange(4)
        with self.assertRaises(ValueError):
            Sample(
                frames=frames,
                players_per_step=[[_snap(101)]] * 4,
                events=[],
                global_features=np.zeros((3, 8), dtype=np.float32),
            )


class LeftToRightAndShirtTests(unittest.TestCase):
    def test_left_to_right_per_frame(self) -> None:
        frames = np.arange(2)
        sample = Sample(
            frames=frames,
            players_per_step=[
                [_snap(101, left_to_right=1.0), _snap(201, team=1, left_to_right=-1.0)],
                [_snap(101, left_to_right=1.0), _snap(201, team=1, left_to_right=-1.0)],
            ],
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertEqual(stacked.left_to_right.shape, (2, 2))
        # Column 0 = player 101 (ltr=+1); column 1 = player 201 (ltr=-1).
        self.assertAlmostEqual(float(stacked.left_to_right[0, 0]), 1.0)
        self.assertAlmostEqual(float(stacked.left_to_right[1, 1]), -1.0)

    def test_shirt_numbers_stable_per_column(self) -> None:
        frames = np.arange(2)
        sample = Sample(
            frames=frames,
            players_per_step=[
                [_snap(101, shirt_number=7), _snap(201, team=1, shirt_number=11)],
                [_snap(101, shirt_number=7), _snap(201, team=1, shirt_number=11)],
            ],
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertEqual(stacked.shirt_numbers.shape, (2,))
        self.assertEqual(int(stacked.shirt_numbers[0]), 7)
        self.assertEqual(int(stacked.shirt_numbers[1]), 11)

    def test_attach_visual_features_preserves_new_fields(self) -> None:
        frames = np.arange(3)
        sample = Sample(
            frames=frames,
            players_per_step=[[_snap(101, shirt_number=7)]] * 3,
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        attached = attach_visual_features(
            stacked, np.zeros((3, 1, 4), dtype=np.float32)
        )
        self.assertIsNotNone(attached.left_to_right)
        self.assertIsNotNone(attached.shirt_numbers)
        self.assertEqual(int(attached.shirt_numbers[0]), 7)


class VisualFeaturesTests(unittest.TestCase):
    def test_default_visual_features_is_none(self) -> None:
        frames = np.arange(2)
        sample = Sample(
            frames=frames,
            players_per_step=[[_snap(101)]] * 2,
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        self.assertIsNone(stacked.visual_features)

    def test_attach_visual_features_validates_shape(self) -> None:
        frames = np.arange(3)
        sample = Sample(
            frames=frames,
            players_per_step=[[_snap(101), _snap(201, team=1)]] * 3,
            events=[],
            meta=SampleMeta(match_id="m"),
        )
        stacked = stack_sample(sample)
        good = np.zeros((3, 2, 5), dtype=np.float32)
        attached = attach_visual_features(stacked, good)
        self.assertIsNotNone(attached.visual_features)
        # Wrong T.
        with self.assertRaises(ValueError):
            attach_visual_features(stacked, np.zeros((4, 2, 5), dtype=np.float32))
        # Wrong P.
        with self.assertRaises(ValueError):
            attach_visual_features(stacked, np.zeros((3, 3, 5), dtype=np.float32))
        # Not 3D.
        with self.assertRaises(ValueError):
            attach_visual_features(stacked, np.zeros((3, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
