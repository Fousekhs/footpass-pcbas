"""W&B sweep agent entry point for hyperparameter search.

Usage — two modes:

1. **Create + run** (sweep.py creates the sweep and immediately starts an agent):

    python scripts/sweep.py \\
        --sweep-config configs/sweeps/hyperparams.yaml \\
        --splits data/splits.json \\
        --output-dir checkpoints/sweeps \\
        --device cuda:0 --epochs 10 --count 20

2. **Agent-only** (sweep was already created with ``wandb sweep``):

    python scripts/sweep.py \\
        --sweep-id <entity/project/sweep_id> \\
        --splits data/splits.json \\
        --output-dir checkpoints/sweeps \\
        --device cuda:0 --epochs 10 --count 20

Fixed args (paths, device, epochs, visual cache, validation settings) are
passed here on the command line and stay constant across all trials.  Per-trial
hyperparameters (learning_rate, hidden_dim, …) are injected by the W&B agent
via ``wandb.config`` and applied through :func:`_build_argparser`'s
``defaults`` dict, which is keyed by argparse dest names (underscores).

Each trial writes its checkpoints to ``<output_dir>/<run_id>/`` so concurrent
agents never collide.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make train.py importable without installing the package.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import _build_argparser, run as _run_training  # noqa: E402


def _build_sweep_argparser() -> argparse.ArgumentParser:
    """Parse the sweep-invariant (fixed) arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sweep identity — exactly one of these is required.
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Existing W&B sweep ID in the form 'entity/project/sweep_id'. "
            "Use this when you created the sweep with ``wandb sweep`` separately."
        ),
    )
    group.add_argument(
        "--sweep-config",
        type=Path,
        default=None,
        help=(
            "Path to a sweep YAML (e.g. configs/sweeps/hyperparams.yaml). "
            "The sweep is created automatically and the agent starts immediately."
        ),
    )

    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of trials to run. Omit to run indefinitely until the sweep finishes.",
    )

    # --- Fixed per-run args (paths, device, …) ---
    p.add_argument("--config", type=Path, default=Path("config.toml"),
                   help="Data-mirror config.toml (same as scripts/train.py --config).")
    p.add_argument("--splits", type=Path, required=True,
                   help="Path to a SplitManifest JSON.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for sweep checkpoints. Each trial writes to <output_dir>/<run_id>/.",
    )
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs per trial.")
    p.add_argument("--device", default="cpu")

    # Visual cache (optional, fixed across all trials)
    p.add_argument("--visual-cache", type=Path, default=None)
    p.add_argument("--visual-backbone", default="dinov2_vits14")
    p.add_argument("--visual-dim", type=int, default=0,
                   help="Must match the cache dim (384 for DINOv2 ViT-S/14).")
    p.add_argument("--target-cache", type=Path, default=None)

    # Validation (fixed across all trials — we always want comparable metrics)
    p.add_argument("--validation-split", default=None)
    p.add_argument("--validation-stride", type=int, default=None)
    p.add_argument("--decode-threshold", type=float, default=0.5)
    p.add_argument("--nms-radius", type=int, default=12)
    p.add_argument("--nms-mode", default="per_player_class",
                   choices=["per_player_class", "per_player", "per_class"])
    p.add_argument("--metric-tolerances", default="3,12,25")
    p.add_argument("--keep-best-metric", default=None,
                   help="Metric to track for best.pt inside each trial's output dir.")

    # W&B project/entity (the sweep was already created in this project)
    p.add_argument("--wandb-project", default="pcspot")
    p.add_argument("--wandb-entity", default=None)

    return p


def _make_trial_fn(fixed: argparse.Namespace) -> Any:
    """Return a zero-argument callable suitable for ``wandb.agent``."""
    import wandb  # type: ignore[import-not-found]

    def _trial() -> None:
        with wandb.init() as wandb_run:
            swept: dict[str, Any] = dict(wandb_run.config)

            # Build a full Namespace: swept values become argparse defaults;
            # the required positional paths are passed as explicit argv so
            # argparse never complains about missing required arguments.
            cli: list[str] = [
                "--config",    str(fixed.config),
                "--splits",    str(fixed.splits),
                "--output-dir", str(fixed.output_dir / wandb_run.id),
                "--epochs",    str(fixed.epochs),
                "--device",    fixed.device,
                "--metric-tolerances", fixed.metric_tolerances,
                "--nms-mode",  fixed.nms_mode,
                "--decode-threshold", str(fixed.decode_threshold),
                "--nms-radius", str(fixed.nms_radius),
            ]
            if fixed.visual_cache is not None:
                cli += ["--visual-cache", str(fixed.visual_cache),
                        "--visual-backbone", fixed.visual_backbone,
                        "--visual-dim", str(fixed.visual_dim)]
            if fixed.target_cache is not None:
                cli += ["--target-cache", str(fixed.target_cache)]
            if fixed.validation_split is not None:
                cli += ["--validation-split", fixed.validation_split]
            if fixed.validation_stride is not None:
                cli += ["--validation-stride", str(fixed.validation_stride)]
            if fixed.keep_best_metric is not None:
                cli += ["--keep-best-metric", fixed.keep_best_metric]

            args = _build_argparser(defaults=swept).parse_args(cli)
            # The wandb run is already open — tell run() not to call wandb.init() again.
            args.wandb = False

            _run_training(args, wandb_run=wandb_run)

    return _trial


def main() -> int:
    fixed = _build_sweep_argparser().parse_args()

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"error: wandb is not installed: {exc}", file=sys.stderr)
        return 1

    sweep_id: Optional[str] = fixed.sweep_id

    if sweep_id is None:
        # Create the sweep from the YAML config file.
        config_path = fixed.sweep_config
        if not config_path.exists():
            print(f"error: sweep config not found: {config_path}", file=sys.stderr)
            return 1
        try:
            with config_path.open("rb") as fh:
                sweep_cfg = tomllib.load(fh)
        except Exception as exc:
            # Try YAML fallback (wandb accepts dicts directly).
            try:
                import yaml  # type: ignore[import-not-found]
                with config_path.open() as fh:
                    sweep_cfg = yaml.safe_load(fh)
            except Exception:
                print(f"error: failed to parse sweep config: {exc}", file=sys.stderr)
                return 1

        create_kwargs: dict[str, Any] = {"sweep": sweep_cfg}
        if fixed.wandb_project:
            create_kwargs["project"] = fixed.wandb_project
        if fixed.wandb_entity:
            create_kwargs["entity"] = fixed.wandb_entity
        sweep_id = wandb.sweep(**create_kwargs)
        print(f"Created sweep: {sweep_id}")

    trial_fn = _make_trial_fn(fixed)
    wandb.agent(
        sweep_id,
        function=trial_fn,
        count=fixed.count,
        project=fixed.wandb_project or None,
        entity=fixed.wandb_entity or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
