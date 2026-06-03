"""Tests for ``pcspot.data.loader``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.loader import (
    EXPECTED_NCOLS,
    HalfArray,
    array_to_sample,
    iter_samples_from_half,
)
from pcspot.data.windows import Window


def _make_row(
    frame: int,
    pid: int,
    *,
    role: int = 1,
    x: float = 0.5,
    y: float = 0.5,
    cls: int = 0,
    shirt: int = 10,
    nan_bbox: bool = False,
) -> np.ndarray:
    row = np.zeros((EXPECTED_NCOLS,), dtype=np.float32)
    row[0] = frame
    row[1] = pid
    row[2] = 1.0  # left_to_right
    row[3] = shirt
    row[4] = role
    row[5] = x
    row[6] = y
    row[7] = 0.0
    row[8] = 0.0
    row[9] = 100.0 if not nan_bbox else float("nan")
    row[10] = 100.0 if not nan_bbox else float("nan")
    row[11] = 50.0 if not nan_bbox else float("nan")
    row[12] = 50.0 if not nan_bbox else float("nan")
    row[13] = cls
    return row


class ArrayToSampleTests(unittest.TestCase):
    def test_groups_rows_by_frame(self) -> None:
        rows = np.stack(
            [
                _make_row(0, 101),
                _make_row(0, 201),
                _make_row(1, 101),
                _make_row(1, 201, cls=2),  # Pass by player 201 at frame 1
                _make_row(2, 101),
            ]
        )
        sample = array_to_sample(rows, Window(0, 3), match_id="m", half_id="m_H1")

        self.assertEqual(sample.num_steps, 3)
        self.assertEqual(len(sample.players_per_step[0]), 2)
        self.assertEqual(len(sample.players_per_step[2]), 1)
        self.assertEqual(len(sample.events), 1)
        ev = sample.events[0]
        self.assertEqual(ev.frame, 1)
        self.assertEqual(ev.player_id, 201)
        self.assertEqual(ev.class_id, 2)

    def test_team_assignment_from_player_id(self) -> None:
        rows = np.stack([_make_row(0, 101), _make_row(0, 201)])
        sample = array_to_sample(rows, Window(0, 1), match_id="m")
        snaps = sample.players_per_step[0]
        teams = {s.player_id: s.team for s in snaps}
        self.assertEqual(teams[101], 0)
        self.assertEqual(teams[201], 1)

    def test_visibility_flag_with_nan_bbox(self) -> None:
        rows = np.stack([_make_row(0, 101, nan_bbox=True)])
        sample = array_to_sample(rows, Window(0, 1), match_id="m")
        self.assertFalse(sample.players_per_step[0][0].visible)

    def test_empty_window_yields_padded_sample(self) -> None:
        sample = array_to_sample(
            np.zeros((0, EXPECTED_NCOLS), dtype=np.float32),
            Window(10, 13),
            match_id="m",
        )
        self.assertEqual(sample.num_steps, 3)
        self.assertEqual(sum(len(p) for p in sample.players_per_step), 0)
        self.assertEqual(sample.events, [])


class IterSamplesTests(unittest.TestCase):
    def test_iter_samples_covers_frame_range(self) -> None:
        rows = np.stack([_make_row(f, 101) for f in range(7)])
        half = HalfArray(match_id="m", half_id="m_H1", array=rows)
        samples = list(
            iter_samples_from_half(half, window_size=3, stride=3)
        )
        # Frames 0..6 with size 3 stride 3 -> windows [0,3), [3,6), tail [4,7)
        self.assertGreaterEqual(len(samples), 2)
        seen = set()
        for s in samples:
            for f in s.frames.tolist():
                seen.add(f)
        self.assertEqual(seen, set(range(7)))


if __name__ == "__main__":
    unittest.main()
