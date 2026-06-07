"""CLI smoke tests for the no-graph variant's ``scripts/no_graph/train.py``.

Mirrors ``ArgParserSmokeTests`` in ``test_train_cli.py`` (which covers
the graph variant's ``scripts/graph/train.py``): both share the same
``cli_common.add_common_args`` base, so this focuses on what's
*different* — the smaller ``architecture`` group (no zone/radius/edge
flags) and ``build_model`` constructing :class:`NoGraphSpottingModel`.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NO_GRAPH_SCRIPTS_DIR = ROOT / "scripts" / "no_graph"
if str(NO_GRAPH_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(NO_GRAPH_SCRIPTS_DIR))

import train as no_graph_train_cli  # noqa: E402  (scripts/no_graph/train.py as a module)

from pcspot.models.no_graph_model import NoGraphSpottingModel  # noqa: E402


class NoGraphArgParserSmokeTests(unittest.TestCase):
    def test_shared_flags_present(self) -> None:
        parser = no_graph_train_cli._build_argparser()
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

    def test_architecture_group_has_no_graph_only_flags(self) -> None:
        parser = no_graph_train_cli._build_argparser()
        help_text = parser.format_help()
        for flag in ("--use-zone-nodes", "--zone-grid", "--use-radius-edges", "--radius",
                     "--use-edge-features", "--knn"):
            self.assertNotIn(flag, help_text)
        for flag in ("--use-jersey", "--use-goal-distances"):
            self.assertIn(flag, help_text)

    def test_architecture_defaults(self) -> None:
        parser = no_graph_train_cli._build_argparser()
        args = parser.parse_args(["--splits", "s.json", "--output-dir", "o"])
        self.assertTrue(args.use_jersey)
        self.assertTrue(args.use_goal_distances)
        self.assertFalse(hasattr(args, "use_zone_nodes"))
        self.assertFalse(hasattr(args, "use_radius_edges"))


class NoGraphBuildModelTests(unittest.TestCase):
    def test_build_model_constructs_no_graph_spotting_model(self) -> None:
        args = argparse.Namespace(
            hidden_dim=8,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            visual_dim=0,
            use_jersey=True,
            use_goal_distances=False,
        )
        model = no_graph_train_cli.build_model(args)
        self.assertIsInstance(model, NoGraphSpottingModel)
        self.assertFalse(hasattr(model, "hgt"))
        self.assertEqual(model.extra_scalar_dim, 0)


class NoGraphModelVariantTests(unittest.TestCase):
    def test_model_variant_constant(self) -> None:
        self.assertEqual(no_graph_train_cli.MODEL_VARIANT, "no_graph")


if __name__ == "__main__":
    unittest.main()
