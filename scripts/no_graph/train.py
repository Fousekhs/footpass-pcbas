"""Train the graph-ablation player-centric ball-action spotting model.

This is the **no-graph** model variant —
:class:`pcspot.models.no_graph_model.NoGraphSpottingModel`
(``embedder -> MS-TCN++ -> head``, with the ``HGTEncoder`` removed
entirely: no edge attention, no zone nodes, no inter-player message
passing of any kind). Tactical (kinematic) and per-player visual
features still flow into the embedder exactly as in the graph variant
(``scripts/graph/train.py``), so a side-by-side comparison of the two
runs' validation metrics measures the added value of the graph.

Thin wrapper: defines the ``architecture`` CLI group + a
``build_model(args)`` factory, then delegates everything else (data
loading, sampler dispatch, validation, W&B, checkpointing) to
:func:`pcspot.train.runner.run`.

Example::

    python scripts/no_graph/train.py \\
        --config config.toml \\
        --splits data/splits.json \\
        --output-dir checkpoints/no_graph/run1 \\
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

from pcspot.models.no_graph_model import NoGraphSpottingModel
from pcspot.train import cli_common, runner
from pcspot.train.cli_common import (  # noqa: F401  (re-exported for tests/sweep)
    _build_calf_config,
    _parse_tolerances,
    _preparse_train_config,
)
from pcspot.train.runner import (  # noqa: F401  (re-exported for tests/sweep)
    _filter_halves,
    _gt_events_for_half,
    make_log_fn,
    make_validation_fn,
)

MODEL_VARIANT = "no_graph"

# The no-graph model has no architecture-specific [train] TOML keys
# beyond cli_common's shared set (use_jersey / use_goal_distances are
# already in there).
_EXTRA_TRAIN_KEYS: frozenset[str] = frozenset()


def build_model(args: argparse.Namespace) -> NoGraphSpottingModel:
    """Construct the no-graph model from a parsed argparse ``Namespace``."""
    return NoGraphSpottingModel(
        hidden_dim=args.hidden_dim,
        num_mstcn_stages=args.num_mstcn_stages,
        num_mstcn_layers=args.num_mstcn_layers,
        visual_dim=args.visual_dim,
        use_jersey=bool(args.use_jersey),
        use_goal_distances=bool(args.use_goal_distances),
    )


def _build_argparser(defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli_common.add_common_args(p, defaults)

    d = defaults or {}

    def _df(name: str, builtin: Any) -> Any:
        return d.get(name, builtin)

    arch_group = p.add_argument_group("architecture")
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
            "the embedder's extra-scalars branch (the only extra-scalar "
            "signal available here — radius-edge degree counts are "
            "graph-derived and therefore dropped). Distances are "
            "oriented to attacking direction via the FOOTPASS "
            "left_to_right column."
        ),
    )
    return p


def _load_train_config(path: Path) -> dict[str, Any]:
    return cli_common._load_train_config(path, extra_train_keys=_EXTRA_TRAIN_KEYS)


def run(args: argparse.Namespace, *, wandb_run: Any = None) -> int:
    """Execute a training run from a pre-built Namespace.

    Delegates to :func:`pcspot.train.runner.run` with this variant's
    :func:`build_model`. Called by :func:`main` for normal CLI use, or
    directly by ``scripts/no_graph/sweep.py`` with a W&B run already
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
