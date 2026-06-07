"""Model-agnostic CLI building blocks shared by every training variant.

Holds the argparse groups, train-config TOML loading/validation, and
CALF-config parsing that do not depend on which model
(:class:`~pcspot.models.graph_model.PlayerCentricSpottingModel`,
:class:`~pcspot.models.no_graph_model.NoGraphSpottingModel`, ...) a
variant's ``scripts/<variant>/train.py`` builds. Each variant calls
:func:`add_common_args` to populate the bulk of its parser, then adds
its own ``architecture`` argument group and extends
``_KNOWN_TRAIN_KEYS`` (via ``extra_train_keys``) for whatever
model-specific TOML keys it accepts.
"""

from __future__ import annotations

import argparse
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Any, FrozenSet, Optional, Sequence

from pcspot.data.schema import NUM_PCBAS_CLASSES
from pcspot.data.targets import CalfConfig


_KNOWN_SECTIONS = {"train", "validation", "wandb", "calf"}

# Base ``[train]`` keys shared by every model variant. Graph-only
# architecture keys (``use_zone_nodes``, ``zone_grid``,
# ``use_radius_edges``, ``radius``) are NOT included here — the graph
# variant passes them via ``_load_train_config``'s ``extra_train_keys``.
_KNOWN_TRAIN_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
    "window_size",
    "stride",
    "no_objectness",
    "hidden_dim",
    "num_mstcn_stages",
    "num_mstcn_layers",
    "visual_dim",
    "visual_backbone",
    "visual_cache_capacity",
    "device",
    "seed",
    "sampler",
    "positive_ratio",
    "event_weighting",
    "samples_per_epoch",
    "sampler_seed",
    "grad_accum_steps",
    "amp",
    "amp_dtype",
    "num_workers",
    "prefetch_factor",
    "persistent_workers",
    "pin_memory",
    "compile",
    "use_jersey",
    "use_goal_distances",
}

_KNOWN_VALIDATION_KEYS = {
    "validation_split",
    "validation_stride",
    "decode_threshold",
    "nms_radius",
    "nms_mode",
    "metric_tolerances",
}

# CALF loss tuning lives in its own ``[calf]`` table rather than as CLI
# flags: the per-class overrides are dicts keyed by class id, which do
# not map cleanly onto scalar argparse flags. These keys mirror the
# fields of :class:`pcspot.data.targets.CalfConfig` 1:1.
_CALF_INT_KEYS = {"k1_default", "k2_default"}
_CALF_FLOAT_KEYS = {
    "positive_weight",
    "ambiguity_weight",
    "ambiguity_radius",
    "gaussian_sigma",
    "teammate_floor",
    "opponent_floor",
    "ambiguity_time_scale",
}
_KNOWN_CALF_KEYS = (
    _CALF_INT_KEYS
    | _CALF_FLOAT_KEYS
    | {
        "distance_falloff",
        "per_class_window",
        "per_class_positive_weight",
        "attacking_classes",
        "duel_classes",
    }
)

# Wandb section keys are remapped to argparse dest names because the
# CLI flags carry the ``wandb_`` prefix while the TOML section already
# names the integration.
_WANDB_KEY_MAP = {
    "enabled": "wandb",
    "project": "wandb_project",
    "entity": "wandb_entity",
    "tags": "wandb_tags",
    "mode": "wandb_mode",
    "log_artifacts": "wandb_log_artifacts",
}


def _abort(message: str) -> None:
    raise SystemExit(message)


def _calf_class_id(path: Path, ctx: str, raw: Any) -> int:
    """Validate and return a 1-based PCBAS class id from a ``[calf]`` table."""
    try:
        cid = int(raw)
    except (TypeError, ValueError):
        _abort(
            f"--train-config {path}: [calf].{ctx} class id {raw!r} "
            f"must be an integer (1..{NUM_PCBAS_CLASSES})."
        )
    if not 1 <= cid <= NUM_PCBAS_CLASSES:
        _abort(
            f"--train-config {path}: [calf].{ctx} class id {cid} is out of "
            f"range; PCBAS classes are 1..{NUM_PCBAS_CLASSES}."
        )
    return cid


def _parse_calf_section(path: Path, section: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``[calf]`` table and return a JSON-safe overrides dict.

    Values are normalised to plain ``int`` / ``float`` / ``str`` / ``list``
    so the dict can ride on the argparse ``Namespace`` and be dumped into
    ``run.json`` unchanged. The list -> ``frozenset`` / ``tuple`` promotion
    that :class:`CalfConfig` expects happens later in
    :func:`_build_calf_config`.
    """
    out: dict[str, Any] = {}
    for k, v in section.items():
        if k not in _KNOWN_CALF_KEYS:
            _abort(
                f"--train-config {path}: unknown key [calf].{k}; "
                f"expected one of {sorted(_KNOWN_CALF_KEYS)}."
            )
        if k in _CALF_INT_KEYS:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                _abort(f"--train-config {path}: [calf].{k} must be an integer.")
            if out[k] < 0:
                _abort(f"--train-config {path}: [calf].{k} must be >= 0.")
        elif k in _CALF_FLOAT_KEYS:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                _abort(f"--train-config {path}: [calf].{k} must be a number.")
        elif k == "distance_falloff":
            s = str(v)
            if s not in ("hard", "gaussian"):
                _abort(
                    f"--train-config {path}: [calf].distance_falloff must be "
                    f"'hard' or 'gaussian', got {v!r}."
                )
            out[k] = s
        elif k in ("attacking_classes", "duel_classes"):
            if not isinstance(v, (list, tuple)):
                _abort(
                    f"--train-config {path}: [calf].{k} must be a list of "
                    f"class ids, e.g. [1, 2, 4]."
                )
            out[k] = [_calf_class_id(path, k, x) for x in v]
        elif k == "per_class_window":
            if not isinstance(v, dict):
                _abort(
                    f"--train-config {path}: [calf.per_class_window] must be a "
                    f"table of class_id = [K1, K2]."
                )
            window: dict[int, list[int]] = {}
            for raw_k, raw_v in v.items():
                cid = _calf_class_id(path, "per_class_window", raw_k)
                if not (isinstance(raw_v, (list, tuple)) and len(raw_v) == 2):
                    _abort(
                        f"--train-config {path}: [calf.per_class_window].{raw_k} "
                        f"must be a 2-item list [K1, K2]."
                    )
                try:
                    k1, k2 = int(raw_v[0]), int(raw_v[1])
                except (TypeError, ValueError):
                    _abort(
                        f"--train-config {path}: [calf.per_class_window].{raw_k} "
                        f"entries must be integers."
                    )
                if k1 < 0 or k2 <= 0:
                    _abort(
                        f"--train-config {path}: [calf.per_class_window].{raw_k} "
                        f"requires K1 >= 0 and K2 > 0."
                    )
                window[cid] = [k1, k2]
            out[k] = window
        elif k == "per_class_positive_weight":
            if not isinstance(v, dict):
                _abort(
                    f"--train-config {path}: [calf.per_class_positive_weight] "
                    f"must be a table of class_id = weight."
                )
            weights: dict[int, float] = {}
            for raw_k, raw_v in v.items():
                cid = _calf_class_id(path, "per_class_positive_weight", raw_k)
                try:
                    w = float(raw_v)
                except (TypeError, ValueError):
                    _abort(
                        f"--train-config {path}: "
                        f"[calf.per_class_positive_weight].{raw_k} must be a number."
                    )
                if w < 0:
                    _abort(
                        f"--train-config {path}: "
                        f"[calf.per_class_positive_weight].{raw_k} must be >= 0."
                    )
                weights[cid] = w
            out[k] = weights
    return out


def _build_calf_config(overrides: Optional[dict[str, Any]]) -> CalfConfig:
    """Construct a :class:`CalfConfig` from a parsed ``[calf]`` overrides dict.

    ``overrides`` is the JSON-safe dict produced by
    :func:`_parse_calf_section` (or ``None`` for built-in defaults). The
    per-class lists are promoted back to the ``tuple`` / ``frozenset``
    shapes that :class:`CalfConfig` stores.
    """
    if not overrides:
        return CalfConfig()
    kwargs = dict(overrides)
    if "per_class_window" in kwargs:
        kwargs["per_class_window"] = {
            int(cid): (int(win[0]), int(win[1]))
            for cid, win in kwargs["per_class_window"].items()
        }
    if "per_class_positive_weight" in kwargs:
        kwargs["per_class_positive_weight"] = {
            int(cid): float(w)
            for cid, w in kwargs["per_class_positive_weight"].items()
        }
    if "attacking_classes" in kwargs:
        kwargs["attacking_classes"] = frozenset(int(x) for x in kwargs["attacking_classes"])
    if "duel_classes" in kwargs:
        kwargs["duel_classes"] = frozenset(int(x) for x in kwargs["duel_classes"])
    return CalfConfig(**kwargs)


def _load_train_config(
    path: Path,
    *,
    extra_train_keys: FrozenSet[str] = frozenset(),
) -> dict[str, Any]:
    """Load a check-in-able training config TOML into argparse defaults.

    Returns a flat ``{dest: value}`` dict keyed by argparse ``dest``
    names so it can be passed straight to a variant's
    ``_build_argparser``. Unknown sections or keys abort with a clear
    ``SystemExit`` so typos never silently change a run. The
    data-mirror ``config.toml`` is unrelated and intentionally not
    touched by this loader.

    ``extra_train_keys`` lets a variant accept additional
    architecture-specific ``[train]`` keys (e.g. the graph variant's
    ``use_zone_nodes`` / ``zone_grid`` / ``use_radius_edges`` /
    ``radius``) without widening the keys every variant validates against.
    """
    known_train_keys = _KNOWN_TRAIN_KEYS | set(extra_train_keys)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        _abort(f"--train-config {path}: file not found.")
    except OSError as exc:
        _abort(f"--train-config {path}: failed to read TOML: {exc}")
    except tomllib.TOMLDecodeError as exc:
        _abort(f"--train-config {path}: invalid TOML: {exc}")

    if not isinstance(data, dict):
        _abort(f"--train-config {path}: top-level must be a TOML table.")

    unknown_sections = set(data.keys()) - _KNOWN_SECTIONS
    if unknown_sections:
        _abort(
            f"--train-config {path}: unknown section(s) {sorted(unknown_sections)}; "
            f"expected only {sorted(_KNOWN_SECTIONS)}."
        )

    out: dict[str, Any] = {}

    train_section = data.get("train")
    if train_section is not None:
        if not isinstance(train_section, dict):
            _abort(f"--train-config {path}: [train] must be a table.")
        for k, v in train_section.items():
            if k not in known_train_keys:
                _abort(
                    f"--train-config {path}: unknown key [train].{k}; "
                    f"expected one of {sorted(known_train_keys)}."
                )
            if k == "no_objectness":
                # The argparse flag is the positive form (--objectness)
                # so we invert here. ``True`` in TOML -> objectness off.
                out["objectness"] = not bool(v)
            else:
                out[k] = v

    val_section = data.get("validation")
    if val_section is not None:
        if not isinstance(val_section, dict):
            _abort(f"--train-config {path}: [validation] must be a table.")
        for k, v in val_section.items():
            if k not in _KNOWN_VALIDATION_KEYS:
                _abort(
                    f"--train-config {path}: unknown key [validation].{k}; "
                    f"expected one of {sorted(_KNOWN_VALIDATION_KEYS)}."
                )
            if k == "metric_tolerances" and isinstance(v, list):
                out[k] = ",".join(str(int(x)) for x in v)
            else:
                out[k] = v

    wb_section = data.get("wandb")
    if wb_section is not None:
        if not isinstance(wb_section, dict):
            _abort(f"--train-config {path}: [wandb] must be a table.")
        for k, v in wb_section.items():
            if k not in _WANDB_KEY_MAP:
                _abort(
                    f"--train-config {path}: unknown key [wandb].{k}; "
                    f"expected one of {sorted(_WANDB_KEY_MAP.keys())}."
                )
            dest = _WANDB_KEY_MAP[k]
            if k == "tags" and isinstance(v, list):
                out[dest] = ",".join(
                    str(t).strip() for t in v if str(t).strip()
                )
            else:
                out[dest] = v

    calf_section = data.get("calf")
    if calf_section is not None:
        if not isinstance(calf_section, dict):
            _abort(f"--train-config {path}: [calf] must be a table.")
        # Stored under a reserved key (not an argparse dest) because CALF
        # tuning is applied via _build_calf_config rather than argparse.
        out["_calf"] = _parse_calf_section(path, calf_section)

    return out


def _preparse_train_config(argv: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Peek at ``argv`` for ``--train-config`` before the real parser runs.

    Returns the resolved :class:`Path` or ``None`` if the flag was not
    supplied. Unknown flags are ignored so this never errors on the
    final parser's required-but-missing arguments.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train-config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)
    return pre_args.train_config


def _parse_zone_grid(spec: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Parse a ``zone_grid`` config value into a ``(Gx, Gy)`` tuple.

    Accepts ``"6x4"`` / ``"6,4"`` from the CLI, or a 2-element list /
    tuple straight from a TOML config. Raises ``SystemExit`` with a
    clear message on malformed input so it surfaces during arg parsing
    rather than mid-training. Used by the graph variant only — the
    no-graph model has no zone nodes.
    """
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
    else:
        text = str(spec).strip()
        sep = "x" if "x" in text else "," if "," in text else None
        if sep is None:
            raise SystemExit(
                f"--zone-grid {text!r} must look like '6x4' or '6,4'"
            )
        vals = [v.strip() for v in text.split(sep)]
    if len(vals) != 2:
        raise SystemExit(f"--zone-grid must have two components, got {spec!r}")
    try:
        gx, gy = int(vals[0]), int(vals[1])
    except (TypeError, ValueError):
        raise SystemExit(f"--zone-grid components must be integers, got {spec!r}")
    if gx < 1 or gy < 1:
        raise SystemExit(f"--zone-grid components must be positive, got {spec!r}")
    return gx, gy


def _parse_tolerances(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise SystemExit("--metric-tolerances must contain at least one positive integer")
    if any(t <= 0 for t in out):
        raise SystemExit("--metric-tolerances entries must be > 0")
    return out


def add_common_args(
    p: argparse.ArgumentParser,
    defaults: Optional[dict[str, Any]] = None,
) -> argparse.ArgumentParser:
    """Populate ``p`` with every model-agnostic training argument.

    Covers paths/sizes/optimisation, the sampler / acceleration /
    objectness / validation / wandb groups. Each variant adds its own
    ``architecture`` group on top (and may add further keys to
    ``_load_train_config`` via ``extra_train_keys``).
    """
    d = defaults or {}

    def _df(name: str, builtin: Any) -> Any:
        return d.get(name, builtin)

    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help=(
            "Optional reusable training config TOML (see "
            "configs/train/graph/baseline.toml or "
            "configs/train/no_graph/baseline.toml). CLI flags still "
            "override anything set here; this file should NOT contain "
            "secrets."
        ),
    )
    p.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Path to a SplitManifest JSON (see pcspot.data.splits.SplitManifest).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory under which checkpoints (including best.pt) are written.",
    )
    p.add_argument("--window-size", type=int, default=_df("window_size", 128))
    p.add_argument("--stride", type=int, default=_df("stride", 96))
    p.add_argument("--epochs", type=int, default=_df("epochs", 5))
    p.add_argument("--batch-size", type=int, default=_df("batch_size", 16))
    p.add_argument("--learning-rate", type=float, default=_df("learning_rate", 1e-3))
    p.add_argument("--weight-decay", type=float, default=_df("weight_decay", 1e-4))
    p.add_argument("--warmup-steps", type=int, default=_df("warmup_steps", 50))
    p.add_argument("--grad-clip", type=float, default=_df("grad_clip", 1.0))
    p.add_argument("--hidden-dim", type=int, default=_df("hidden_dim", 64))
    p.add_argument("--num-mstcn-stages", type=int, default=_df("num_mstcn_stages", 3))
    p.add_argument("--num-mstcn-layers", type=int, default=_df("num_mstcn_layers", 10))
    p.add_argument(
        "--visual-cache",
        type=Path,
        default=None,
        help=(
            "Root of the visual feature cache built by "
            "scripts/precompute_visual_features.py. Omit to train without "
            "visual features (visual_dim=0)."
        ),
    )
    p.add_argument("--visual-backbone", default=_df("visual_backbone", "dinov2_vits14"))
    p.add_argument("--visual-dim", type=int, default=_df("visual_dim", 0),
                   help="Visual feature dim; must match the cache (DINOv2 ViT-S/14 = 384).")
    p.add_argument(
        "--visual-cache-capacity",
        type=int,
        default=_df("visual_cache_capacity", 8),
        help=(
            "How many match-half shard indices to keep resident per worker. "
            "Raise it (e.g. to the number of halves) when using random "
            "(--sampler mixed/uniform) sampling so shards are not re-opened "
            "every window. With the memmap cache layout each resident shard "
            "costs only its small per-row index, not the features."
        ),
    )
    p.add_argument(
        "--target-cache",
        type=Path,
        default=None,
        help="Optional disk-backed TargetCache directory (CALF + objectness).",
    )
    p.add_argument("--device", default=_df("device", "cpu"))
    p.add_argument("--seed", type=int, default=_df("seed", 42))
    p.add_argument("--keep-best-metric", default=None,
                   help="Metric name from validation_fn to track; saved as best.pt.")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint to resume from (epoch, optimizer state, "
            "global step). Pair with --output-dir to continue writing "
            "epoch_NNNN.pt / latest.pt into the same directory."
        ),
    )

    sampler_group = p.add_argument_group("sampler")
    sampler_group.add_argument(
        "--sampler",
        default=_df("sampler", "sequential"),
        choices=["sequential", "mixed", "uniform"],
        help=(
            "How to draw training windows. 'sequential' (default) "
            "materialises every window once per epoch in dataset order "
            "(legacy behavior). 'mixed' uses MixedEventSampler with "
            "--positive-ratio oversampling; 'uniform' uses "
            "UniformWindowSampler over the dataset."
        ),
    )
    sampler_group.add_argument(
        "--positive-ratio",
        type=float,
        default=_df("positive_ratio", 0.7),
        help="Fraction of each batch drawn from event-bearing windows (mixed sampler).",
    )
    sampler_group.add_argument(
        "--event-weighting",
        default=_df("event_weighting", "uniform"),
        choices=["uniform", "linear"],
        help="Per-positive weighting inside the mixed sampler.",
    )
    sampler_group.add_argument(
        "--samples-per-epoch",
        type=int,
        default=_df("samples_per_epoch", None),
        help=(
            "Number of windows drawn per epoch when --sampler is mixed or "
            "uniform. Defaults to len(dataset) when omitted."
        ),
    )
    sampler_group.add_argument(
        "--sampler-seed",
        type=int,
        default=_df("sampler_seed", None),
        help=(
            "Base seed for the sampler. Epoch e uses seed (base + e) so "
            "different epochs see different draws while staying reproducible."
        ),
    )

    accel_group = p.add_argument_group("acceleration")
    accel_group.add_argument(
        "--grad-accum-steps",
        type=int,
        default=_df("grad_accum_steps", 1),
        help=(
            "Number of forward/backward passes to accumulate before each "
            "optimizer step. Multiplies the effective batch size."
        ),
    )
    accel_group.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=_df("amp", False),
        help="Enable torch.autocast mixed precision (requires CUDA).",
    )
    accel_group.add_argument(
        "--amp-dtype",
        default=_df("amp_dtype", "bf16"),
        choices=["bf16", "fp16"],
        help="Autocast dtype when --amp is set. bf16 is preferred on Ampere+.",
    )
    accel_group.add_argument(
        "--num-workers",
        type=int,
        default=_df("num_workers", 0),
        help=(
            "DataLoader worker processes for sample/target preparation. "
            "0 (default) keeps the legacy single-process loop; >0 moves "
            "padding, visual-feature alignment, and CALF/objectness target "
            "building off the main thread so they overlap with GPU compute. "
            "4-8 typically saturates this pipeline."
        ),
    )
    accel_group.add_argument(
        "--prefetch-factor",
        type=int,
        default=_df("prefetch_factor", 4),
        help="Batches prefetched per worker (only used when --num-workers > 0).",
    )
    accel_group.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=_df("persistent_workers", True),
        help=(
            "Keep DataLoader workers alive across epochs (avoids re-spawning; "
            "matters most on Windows spawn). Only used with --num-workers > 0."
        ),
    )
    accel_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=_df("pin_memory", None),
        help=(
            "Pin host memory for faster async host->device copies. Defaults to "
            "on when training on CUDA, off otherwise."
        ),
    )
    accel_group.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=_df("compile", False),
        help=(
            "Wrap the model in torch.compile for higher GPU utilisation. "
            "Best-effort: falls back to eager mode if compilation fails."
        ),
    )
    p.add_argument(
        "--objectness",
        action=argparse.BooleanOptionalAction,
        default=_df("objectness", True),
        help=(
            "Enable objectness supervision. Use --no-objectness to "
            "disable (the previous CLI form). In the train-config TOML, "
            "set [train].no_objectness = true to disable."
        ),
    )

    val = p.add_argument_group("validation")
    val.add_argument(
        "--validation-split",
        default=_df("validation_split", None),
        help=(
            "Optional split name from --splits to use for per-epoch "
            "validation. When set, predictions are decoded and scored "
            "via average_map_at_tolerances and the metric keys are "
            "available to --keep-best-metric (e.g. val/map_joint)."
        ),
    )
    val.add_argument(
        "--validation-stride",
        type=int,
        default=_df("validation_stride", None),
        help="Validation window stride (defaults to --window-size; non-overlapping).",
    )
    val.add_argument("--decode-threshold", type=float, default=_df("decode_threshold", 0.5),
                     help="Score threshold for decode_predictions during validation.")
    val.add_argument("--nms-radius", type=int, default=_df("nms_radius", 12),
                     help="player_centric_nms window radius (frames).")
    val.add_argument(
        "--nms-mode",
        default=_df("nms_mode", "per_player_class"),
        choices=["per_player_class", "per_player", "per_class"],
        help="player_centric_nms grouping mode.",
    )
    val.add_argument(
        "--metric-tolerances",
        default=_df("metric_tolerances", "3,12,25"),
        help=(
            "Comma-separated frame tolerances (e.g. '3,12,25' for ~120ms, "
            "~480ms, 1s at 25 fps). Used by average_map_at_tolerances."
        ),
    )

    wb = p.add_argument_group("weights & biases")
    wb.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=_df("wandb", False),
        help="Stream training metrics to Weights & Biases (use --no-wandb to disable).",
    )
    wb.add_argument("--wandb-project", default=_df("wandb_project", "pcspot"),
                    help="W&B project name (defaults to 'pcspot').")
    wb.add_argument("--wandb-entity", default=_df("wandb_entity", None),
                    help="W&B entity (team or username). Defaults to the user's default.")
    wb.add_argument("--wandb-run-name", default=None,
                    help="Optional human-friendly W&B run name (CLI-only).")
    wb.add_argument(
        "--wandb-mode",
        default=_df("wandb_mode", None),
        choices=["online", "offline", "disabled"],
        help=(
            "W&B run mode. Defaults to the wandb client default "
            "(respects WANDB_MODE). Use 'offline' to run without network "
            "and sync later with `wandb sync`."
        ),
    )
    wb.add_argument("--wandb-tags", default=_df("wandb_tags", None),
                    help="Comma-separated tags for the W&B run.")
    wb.add_argument("--wandb-notes", default=None,
                    help="Free-form notes for the W&B run (CLI-only).")
    wb.add_argument(
        "--wandb-run-dir",
        type=Path,
        default=None,
        help="Directory under which W&B stores per-run state (defaults to ./wandb).",
    )
    wb.add_argument(
        "--wandb-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=_df("wandb_log_artifacts", False),
        help="After training, upload run.json and *.pt checkpoints as a W&B artifact.",
    )
    return p
