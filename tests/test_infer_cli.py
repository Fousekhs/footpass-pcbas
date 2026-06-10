"""Tests for ``scripts/infer.py`` CLI helper logic."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import infer  # noqa: E402
from pcspot.data.loader import COL_PLAYER_ID, COL_SHIRT, EXPECTED_NCOLS, HalfArray  # noqa: E402


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


class BuildJerseyLookupTests(unittest.TestCase):
    def _half(self, rows: list[tuple[float, float]]) -> HalfArray:
        arr = np.zeros((len(rows), EXPECTED_NCOLS), dtype=np.float32)
        for i, (pid, shirt) in enumerate(rows):
            arr[i, COL_PLAYER_ID] = pid
            arr[i, COL_SHIRT] = shirt
        return HalfArray(match_id="game_0", half_id="game_0_H1", array=arr)

    def test_maps_player_id_to_shirt_number(self) -> None:
        half = self._half([(101, 10), (201, 4)])
        lookup = infer._build_jersey_lookup([half])
        self.assertEqual(lookup, {101: 10, 201: 4})

    def test_skips_nan_entries(self) -> None:
        half = self._half([(101, float("nan"))])
        lookup = infer._build_jersey_lookup([half])
        self.assertEqual(lookup, {})

    def test_merges_lookups_across_halves(self) -> None:
        h1 = self._half([(101, 10)])
        h2 = self._half([(201, 4)])
        lookup = infer._build_jersey_lookup([h1, h2])
        self.assertEqual(lookup, {101: 10, 201: 4})


class ResolveModelKwargsTests(unittest.TestCase):
    def _write_run_json(self, tmp: Path, args: dict) -> Path:
        checkpoint = tmp / "best.pt"
        (tmp / "run.json").write_text(json.dumps({"args": args}), encoding="utf-8")
        return checkpoint

    def test_zone_grid_string_is_parsed_to_tuple(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint = self._write_run_json(
                Path(tmp), {"hidden_dim": 512, "zone_grid": "6x4"}
            )
            kwargs = infer._resolve_model_kwargs(
                checkpoint, {"hidden_dim": None}, model_init_keys=("hidden_dim", "zone_grid")
            )
        self.assertEqual(kwargs["zone_grid"], (6, 4))
        self.assertEqual(kwargs["hidden_dim"], 512)

    def test_zone_grid_list_is_parsed_to_tuple(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint = self._write_run_json(Path(tmp), {"zone_grid": [6, 4]})
            kwargs = infer._resolve_model_kwargs(
                checkpoint, {}, model_init_keys=("zone_grid",)
            )
        self.assertEqual(kwargs["zone_grid"], (6, 4))


if __name__ == "__main__":
    unittest.main()
