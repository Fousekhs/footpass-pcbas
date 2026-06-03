"""Tests for ``pcspot.data.windows``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.windows import Window, iter_windows


class WindowTests(unittest.TestCase):
    def test_non_overlapping(self) -> None:
        windows = list(iter_windows(0, 9, window_size=5))
        self.assertEqual(windows, [Window(0, 5), Window(5, 10)])

    def test_with_stride_and_tail(self) -> None:
        windows = list(iter_windows(0, 11, window_size=5, stride=3))
        # Full windows at 0, 3, 6 and a tail window covering 7..11.
        self.assertEqual(
            windows,
            [Window(0, 5), Window(3, 8), Window(6, 11), Window(7, 12)],
        )

    def test_drop_last_skips_tail(self) -> None:
        windows = list(iter_windows(0, 11, window_size=5, stride=3, drop_last=True))
        self.assertEqual(windows, [Window(0, 5), Window(3, 8), Window(6, 11)])

    def test_empty_range(self) -> None:
        self.assertEqual(list(iter_windows(5, 4, window_size=3)), [])


if __name__ == "__main__":
    unittest.main()
