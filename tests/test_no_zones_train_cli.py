"""CLI smoke tests for the no_zones variant's ``scripts/no_zones/train.py``.

Mirrors ``NoGraphArgParserSmokeTests`` in ``test_no_graph_train_cli.py``:
all variants share the same ``cli_common.add_common_args`` base, so this
focuses on what's *different* — the ``architecture`` group keeps the
radius-edge flags but drops the zone flags, and ``build_model``
constructs :class:`NoZonesSpottingModel` (HGT present, zones removed).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load scripts/no_zones/train.py under a unique module name. Every
# variant's train.py is named ``train``, so importing it as ``train``
# would collide in ``sys.modules`` with the other variants' CLI tests
# (whichever is imported first wins). A file-spec load with an explicit
# name keeps this test order-independent.
_spec = importlib.util.spec_from_file_location(
    "no_zones_train_cli", ROOT / "scripts" / "no_zones" / "train.py"
)
no_zones_train_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(no_zones_train_cli)

from pcspot.models.no_zones_model import NoZonesSpottingModel  # noqa: E402


class NoZonesArgParserSmokeTests(unittest.TestCase):
    def test_shared_flags_present(self) -> None:
        parser = no_zones_train_cli._build_argparser()
        args = parser.parse_args(
            [
                "--splits", "s.json",
                "--output-dir", "o",
                "--validation-split", "val",
                "--wandb",
                "--metric-tolerances", "3,12",
            ]
        )
        self.assertTrue(args.wandb)
        self.assertEqual(args.validation_split, "val")
        self.assertEqual(args.metric_tolerances, "3,12")

    def test_architecture_group_has_no_zones_flags(self) -> None:
        parser = no_zones_train_cli._build_argparser()
        help_text = parser.format_help()
        # No zone knobs in this variant.
        for flag in ("--use-zone-nodes", "--zone-grid"):
            self.assertNotIn(flag, help_text)
        # ...but the radius-edge and per-player feature flags remain.
        for flag in ("--use-jersey", "--use-goal-distances", "--use-radius-edges", "--radius"):
            self.assertIn(flag, help_text)

    def test_architecture_defaults(self) -> None:
        parser = no_zones_train_cli._build_argparser()
        args = parser.parse_args(["--splits", "s.json", "--output-dir", "o"])
        self.assertTrue(args.use_jersey)
        self.assertTrue(args.use_goal_distances)
        self.assertTrue(args.use_radius_edges)
        self.assertEqual(args.radius, 0.15)
        self.assertFalse(hasattr(args, "use_zone_nodes"))
        self.assertFalse(hasattr(args, "zone_grid"))


class NoZonesBuildModelTests(unittest.TestCase):
    def test_build_model_constructs_no_zones_spotting_model(self) -> None:
        args = argparse.Namespace(
            hidden_dim=8,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            visual_dim=0,
            use_jersey=True,
            use_goal_distances=True,
            use_radius_edges=True,
            radius=0.15,
        )
        model = no_zones_train_cli.build_model(args)
        self.assertIsInstance(model, NoZonesSpottingModel)
        self.assertTrue(hasattr(model, "hgt"))
        self.assertFalse(model.use_zone_nodes)
        self.assertEqual(model.num_zones, 0)


class NoZonesModelVariantTests(unittest.TestCase):
    def test_model_variant_constant(self) -> None:
        self.assertEqual(no_zones_train_cli.MODEL_VARIANT, "no_zones")


if __name__ == "__main__":
    unittest.main()
