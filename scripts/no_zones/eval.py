"""Standalone checkpoint evaluation for the no_zones model variant.

Reconstructs :class:`pcspot.models.no_zones_model.NoZonesSpottingModel`
from a checkpoint's ``run.json`` (written by ``scripts/no_zones/train.py``)
and reproduces the validation metrics for a chosen split via
:func:`pcspot.train.eval_runner.evaluate_checkpoint` — the exact same
metric pipeline (decode -> NMS -> average_map_at_tolerances) used during
training, so standalone numbers match the training-time validation log.

Example::

    python scripts/no_zones/eval.py \\
        --checkpoint checkpoints/no_zones/run1/best.pt \\
        --config config.toml --splits data/splits.json --split val \\
        --visual-cache .cache/visual --visual-dim 384 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcspot.models.no_zones_model import NoZonesSpottingModel
from pcspot.train.eval_runner import evaluate_checkpoint

# Restricted to the keys NoZonesSpottingModel.__init__ accepts; mined
# from run.json's "args" payload (CLI overrides win). Same as the graph
# variant's MODEL_INIT_KEYS minus the two zone-only keys (use_zone_nodes
# / zone_grid), which this variant's __init__ does not accept. Keep this
# in sync with scripts/infer.py's _MODEL_REGISTRY["no_zones"].
MODEL_INIT_KEYS = (
    "hidden_dim",
    "num_classes",
    "num_hgt_layers",
    "num_heads",
    "num_mstcn_stages",
    "num_mstcn_layers",
    "knn",
    "global_dim",
    "visual_dim",
    "visual_proj_dim",
    "with_confidence",
    "use_acceleration",
    "use_time_features",
    "use_edge_features",
    "use_jersey",
    "use_goal_distances",
    "use_radius_edges",
    "radius",
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--split", default="val", help="Split name to evaluate against.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=None,
                   help="Defaults to --window-size (non-overlapping).")
    p.add_argument("--visual-cache", type=Path, default=None)
    p.add_argument("--visual-backbone", default="dinov2_vits14")
    p.add_argument("--decode-threshold", type=float, default=0.5)
    p.add_argument("--nms-radius", type=int, default=12)
    p.add_argument("--nms-mode", default="per_player_class",
                   choices=["per_player_class", "per_player", "per_class"])
    p.add_argument("--metric-tolerances", default="3,12,25")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the metrics dict as JSON.")
    # Model construction overrides. Defaults are None so values are
    # pulled from run.json next to the checkpoint when available.
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-mstcn-stages", type=int, default=None)
    p.add_argument("--num-mstcn-layers", type=int, default=None)
    p.add_argument("--visual-dim", type=int, default=None)
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    overrides: dict[str, Any] = {
        "hidden_dim": args.hidden_dim,
        "num_mstcn_stages": args.num_mstcn_stages,
        "num_mstcn_layers": args.num_mstcn_layers,
        "visual_dim": args.visual_dim,
    }
    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        config=args.config,
        splits=args.splits,
        split=args.split,
        model_cls=NoZonesSpottingModel,
        model_init_keys=MODEL_INIT_KEYS,
        model_kwarg_overrides=overrides,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        visual_cache=args.visual_cache,
        visual_backbone=args.visual_backbone,
        decode_threshold=args.decode_threshold,
        nms_radius=args.nms_radius,
        nms_mode=args.nms_mode,
        metric_tolerances=args.metric_tolerances,
    )
    print(json.dumps(metrics, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
