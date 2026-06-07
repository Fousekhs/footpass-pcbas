"""Train the graph-based player-centric ball-action spotting model.

This is the **main** model variant —
:class:`pcspot.models.graph_model.PlayerCentricSpottingModel`
(``embedder -> HGTEncoder -> MS-TCN++ -> head``). For the graph-ablation
sibling that drops inter-player message passing entirely, see
``scripts/no_graph/train.py``.

Thin wrapper: defines the ``architecture`` CLI group + a
``build_model(args)`` factory, then delegates everything else (data
loading, sampler dispatch, validation, W&B, checkpointing) to
:func:`pcspot.train.runner.run`.

Example::

    python scripts/graph/train.py \\
        --config config.toml \\
        --splits data/splits.json \\
        --output-dir checkpoints/graph/run1 \\
        --epochs 5 --batch-size 4 --hidden-dim 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcspot.models.graph_model import PlayerCentricSpottingModel
from pcspot.train import cli_common, runner
from pcspot.train.cli_common import (  # noqa: F401  (re-exported for tests/sweep)
    _build_calf_config,
    _parse_tolerances,
    _parse_zone_grid,
    _preparse_train_config,
)
from pcspot.train.runner import (  # noqa: F401  (re-exported for tests/sweep)
    _filter_halves,
    _gt_events_for_half,
    make_log_fn,
    make_validation_fn,
)

MODEL_VARIANT = "graph"

# Graph-only [train] TOML keys layered on top of cli_common's shared set.
_EXTRA_TRAIN_KEYS = frozenset({"use_zone_nodes", "zone_grid", "use_radius_edges", "radius"})


def build_model(args: argparse.Namespace) -> PlayerCentricSpottingModel:
    """Construct the graph model from a parsed argparse ``Namespace``."""
    zone_grid = _parse_zone_grid(args.zone_grid)
    return PlayerCentricSpottingModel(
        hidden_dim=args.hidden_dim,
        num_mstcn_stages=args.num_mstcn_stages,
        num_mstcn_layers=args.num_mstcn_layers,
        visual_dim=args.visual_dim,
        use_zone_nodes=bool(args.use_zone_nodes),
        zone_grid=zone_grid,
        use_jersey=bool(args.use_jersey),
        use_goal_distances=bool(args.use_goal_distances),
        use_radius_edges=bool(args.use_radius_edges),
        radius=float(args.radius),
    )


def _build_argparser(defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli_common.add_common_args(p, defaults)

    d = defaults or {}

    def _df(name: str, builtin: Any) -> Any:
        return d.get(name, builtin)

    arch_group = p.add_argument_group("architecture")
    arch_group.add_argument(
        "--use-zone-nodes",
        action=argparse.BooleanOptionalAction,
        default=_df("use_zone_nodes", True),
        help=(
            "Enable heterogeneous zone nodes in the HGT. They provide a "
            "count-aware (SUM-aggregated) view of physical congestion "
            "that softmax/mean aggregators cannot represent. Use "
            "--no-use-zone-nodes to reproduce the pre-zone model."
        ),
    )
    arch_group.add_argument(
        "--zone-grid",
        default=_df("zone_grid", "6x4"),
        help=(
            "Pitch grid for zone nodes, formatted as 'GxxGy' (e.g. "
            "'6x4') or 'Gx,Gy'. Only used when --use-zone-nodes is on."
        ),
    )
    arch_group.add_argument(
        "--use-jersey",
        action=argparse.BooleanOptionalAction,
        default=_df("use_jersey", True),
        help=(
            "Enable the jersey-number embedding branch in the node "
            "embedder. Jersey is more identity-stable than player_id "
            "across tracklet switches."
        ),
    )
    arch_group.add_argument(
        "--use-goal-distances",
        action=argparse.BooleanOptionalAction,
        default=_df("use_goal_distances", True),
        help=(
            "Enable [dist_own_goal, dist_opp_goal, dist_sideline] in "
            "the embedder's extra-scalars branch. Distances are oriented "
            "to attacking direction via the FOOTPASS left_to_right column."
        ),
    )
    arch_group.add_argument(
        "--use-radius-edges",
        action=argparse.BooleanOptionalAction,
        default=_df("use_radius_edges", True),
        help=(
            "Enable the variable-degree 'radius' player-player edge "
            "type plus per-player [n_same_within_r, n_opp_within_r] "
            "degree counts in the extra-scalars branch."
        ),
    )
    arch_group.add_argument(
        "--radius",
        type=float,
        default=_df("radius", 0.15),
        help=(
            "Pitch distance (normalized units) for the radius edge "
            "type and the matching degree counts. Default 0.15 ~ 9 m "
            "on a 60 m pitch width."
        ),
    )
    return p


def _load_train_config(path: Path) -> dict[str, Any]:
    return cli_common._load_train_config(path, extra_train_keys=_EXTRA_TRAIN_KEYS)


def run(args: argparse.Namespace, *, wandb_run: Any = None) -> int:
    """Execute a training run from a pre-built Namespace.

    Delegates to :func:`pcspot.train.runner.run` with this variant's
    :func:`build_model`. Called by :func:`main` for normal CLI use, or
    directly by ``scripts/graph/sweep.py`` with a W&B run already
    initialized by the sweep agent.
    """
    return runner.run(args, build_model, wandb_run=wandb_run)


def main() -> int:
    # Pre-parse --train-config so its values become the parser's
    # defaults; any CLI flag still wins on the real parse pass below.
    train_config_path = _preparse_train_config()
    file_defaults: dict[str, Any] = {}
    if train_config_path is not None:
        file_defaults = _load_train_config(train_config_path.resolve())
    args = _build_argparser(defaults=file_defaults).parse_args()
    # CALF tuning has no CLI flags (the per-class overrides are dicts);
    # carry the parsed [calf] table onto the Namespace so run() can apply
    # it. ``None`` -> CalfConfig defaults.
    args.calf_overrides = file_defaults.get("_calf")
    args.model_variant = MODEL_VARIANT
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
