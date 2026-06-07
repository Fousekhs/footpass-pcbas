"""Shared W&B sweep-agent machinery for every model variant.

Generalises ``scripts/sweep.py`` so each variant's
``scripts/<variant>/sweep.py`` only supplies its own argparser/run pair
(``build_argparser`` from ``scripts/<variant>/train.py``, ``run`` =
``functools.partial(runner.run, build_model=...)`` or equivalent) and a
description string; the sweep-config loading, fixed-arg parsing, and
``wandb.agent`` wiring are shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


BuildArgparser = Callable[..., argparse.ArgumentParser]
RunFn = Callable[..., int]


def build_sweep_argparser(description: str) -> argparse.ArgumentParser:
    """Parse the sweep-invariant (fixed) arguments."""
    p = argparse.ArgumentParser(
        description=description,
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
            "Path to a sweep YAML (e.g. configs/sweeps/graph.yaml). "
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
                   help="Data-mirror config.toml (same as scripts/<variant>/train.py --config).")
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

    # Training throughput knobs (fixed across all trials). These are forwarded
    # to train.py so the sweep does not silently fall back to its defaults
    # (--sampler sequential, --num-workers 0), which eagerly materialises every
    # window on one core before the first step.
    p.add_argument(
        "--sampler",
        default=None,
        choices=["sequential", "mixed", "uniform"],
        help=(
            "Window sampler. Use 'mixed' to honour the swept positive_ratio; "
            "'sequential' (train.py default) ignores positive_ratio and "
            "materialises all windows up front."
        ),
    )
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker processes. >0 streams batches instead "
                        "of materialising the full window list on one core.")
    p.add_argument("--samples-per-epoch", type=int, default=None,
                   help="Windows drawn per epoch for mixed/uniform samplers.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Training batch size (passed through to train.py).")
    p.add_argument("--visual-cache-capacity", type=int, default=None,
                   help="Resident shard-index count per worker. Raise for random "
                        "(mixed/uniform) sampling so shards are not re-opened per window.")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable mixed-precision training (forwarded to train.py).")
    p.add_argument("--amp-dtype", default=None, choices=["bf16", "fp16"],
                   help="AMP dtype when --amp is set (bf16 recommended on Ada/Blackwell).")

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


def make_trial_fn(
    fixed: argparse.Namespace,
    *,
    build_argparser: BuildArgparser,
    run_training: RunFn,
) -> Any:
    """Return a zero-argument callable suitable for ``wandb.agent``.

    ``build_argparser`` / ``run_training`` are the variant's own
    ``_build_argparser`` (accepts ``defaults=...``) and ``run``
    (accepts ``args, *, wandb_run=...``) — typically re-exported from
    ``scripts/<variant>/train.py``.
    """
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
            if fixed.sampler is not None:
                cli += ["--sampler", fixed.sampler]
            if fixed.num_workers is not None:
                cli += ["--num-workers", str(fixed.num_workers)]
            if fixed.samples_per_epoch is not None:
                cli += ["--samples-per-epoch", str(fixed.samples_per_epoch)]
            if fixed.batch_size is not None:
                cli += ["--batch-size", str(fixed.batch_size)]
            if fixed.visual_cache_capacity is not None:
                cli += ["--visual-cache-capacity", str(fixed.visual_cache_capacity)]
            if fixed.amp:
                cli += ["--amp"]
            if fixed.amp_dtype is not None:
                cli += ["--amp-dtype", fixed.amp_dtype]

            args = build_argparser(defaults=swept).parse_args(cli)
            # The wandb run is already open — tell run() not to call wandb.init() again.
            args.wandb = False

            run_training(args, wandb_run=wandb_run)

    return _trial


def main(
    *,
    description: str,
    build_argparser: BuildArgparser,
    run_training: RunFn,
    argv: Optional[list[str]] = None,
) -> int:
    """Shared sweep-agent entry point.

    ``description`` becomes the parser's docstring summary;
    ``build_argparser`` / ``run_training`` are the variant's
    ``_build_argparser`` and ``run`` (or thin wrappers around them —
    e.g. ``run`` pre-bound to the variant's ``build_model``).
    """
    fixed = build_sweep_argparser(description).parse_args(argv)

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

    trial_fn = make_trial_fn(fixed, build_argparser=build_argparser, run_training=run_training)
    wandb.agent(
        sweep_id,
        function=trial_fn,
        count=fixed.count,
        project=fixed.wandb_project or None,
        entity=fixed.wandb_entity or None,
    )
    return 0
