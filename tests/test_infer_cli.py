"""Tests for ``scripts/infer.py`` CLI helper logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import infer  # noqa: E402


class HalfMatchesTests(unittest.TestCase):
    def test_matches_full_half_id(self) -> None:
        self.assertTrue(infer._half_matches("game_0_H1", "H1"))
        self.assertTrue(infer._half_matches("game_0_H2", "H2"))

    def test_rejects_other_half(self) -> None:
        self.assertFalse(infer._half_matches("game_0_H1", "H2"))

    def test_matches_bare_half_id(self) -> None:
        self.assertTrue(infer._half_matches("H1", "H1"))

    def test_does_not_match_different_match_with_shared_suffix(self) -> None:
        self.assertFalse(infer._half_matches("game_10_H1", "0_H1"))


if __name__ == "__main__":
    unittest.main()
