"""Train the player-centric ball-action spotting model.

Minimal CLI wrapper around :class:`pcspot.train.trainer.Trainer`. It
loads the FOOTPASS / PCBAS halves discovered under the directory
referenced by ``[pcbas].output_dir`` in ``config.toml``, builds a
:class:`pcspot.data.dataset.PCBASDataset` for the ``train`` split of a
:class:`pcspot.data.splits.SplitManifest`, optionally attaches a
precomputed DINOv2 visual cache, and runs
:func:`Trainer.fit`.

Optional integrations:

- ``--validation-split``: opt into per-epoch validation. A second
  :class:`PCBASDataset` is built for the named split, model
  predictions are decoded with
  :func:`pcspot.eval.nms.decode_predictions`, suppressed with
  :func:`pcspot.eval.nms.player_centric_nms`, and scored against the
  ground-truth events of the val halves using
  :func:`pcspot.eval.metrics.average_map_at_tolerances`. The returned
  ``dict[str, float]`` is what ``--keep-best-metric`` matches against.
- ``--wandb``: stream step / epoch losses and any validation metrics to
  a Weights & Biases run. Authenticate with ``wandb login`` or by
  exporting ``WANDB_API_KEY`` in the shell; no credentials are read
  from ``config.toml``.

The Python-API equivalent is documented inline in
``docs/player_centric_hgt_mstcn_calf_design.md`` (Section 7).

Example::

    python scripts/train.py \\
        --config config.toml \\
        --splits data/splits.json \\
        --output-dir checkpoints/run1 \\
        --epochs 5 --batch-size 4 --hidden-dim 64

For a quick smoke test on the existing CPU venv without a visual cache,
omit ``--visual-cache``; the model will be built with ``visual_dim=0``
and the dataset will pass through kinematic / role / team features only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import (
    COL_CLASS,
    COL_FRAME,
    COL_PLAYER_ID,
    HalfArray,
    load_halves_from_pcbas,
)
from pcspot.data.sampling import (
    build_dataset_batch_provider,
    reseeded_mixed_sampler_factory,
)
# Canonical JSON-safe rendering of a CalfConfig (frozensets -> sorted
# lists). Shared with the target cache so run.json and the cache key
# agree on the config's serialised form.
from pcspot.data.cache import _serializable_config as _calf_serializable
from pcspot.data.schema import NUM_PCBAS_CLASSES, EventLabel
from pcspot.data.splits import SplitManifest
from pcspot.data.targets import CalfConfig
from pcspot.eval.metrics import average_map_at_tolerances
from pcspot.eval.nms import Prediction, decode_predictions, player_centric_nms
from pcspot.models.pipeline import PlayerCentricSpottingModel, stacked_to_batch
from pcspot.train.trainer import (
    EpochLog,
    Trainer,
    TrainStepLog,
    WarmupCosineSchedule,
    build_collated_dataloader_provider,
)


# A large offset added per-half during validation to disambiguate
# absolute frame indices across different match-halves. Matching in
# ``average_map_at_tolerances`` is by ``frame`` distance, so two halves
# that both number frames from 0 would otherwise cross-match.
_HALF_FRAME_OFFSET = 10_000_000


def _load_output_dir(config_path: Path) -> Path:
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("pcbas")
    if not isinstance(section, dict) or "output_dir" not in section:
        raise SystemExit(
            f"{config_path}: missing [pcbas].output_dir; see config.example.toml"
        )
    out = Path(section["output_dir"])
    if not out.is_absolute():
        out = (config_path.parent / out).resolve()
    return out


def _make_visual_cache(root: Path, backbone: str, lru_capacity: int = 8):
    from pcspot.features.cache import VisualFeatureCache

    return VisualFeatureCache(
        root=root, backbone_name=backbone, lru_capacity=max(1, int(lru_capacity))
    )


def _materialize_samples(dataset: PCBASDataset):
    samples = []
    for i in range(len(dataset)):
        stacked, _ = dataset[i]
        samples.append(stacked)
    return samples


def _filter_halves(
    all_halves: Sequence[HalfArray],
    manifest: SplitManifest,
    split: str,
) -> tuple[list[HalfArray], set[tuple[str, str]]]:
    wanted = set(manifest.halves_for(split))
    halves = [
        h for h in all_halves if (str(h.match_id), str(h.half_id)) in wanted
    ]
    missing = wanted - {(str(h.match_id), str(h.half_id)) for h in halves}
    return halves, missing


def _gt_events_for_half(half: HalfArray) -> list[EventLabel]:
    """Extract player-centric ground-truth events from a half array.

    Mirrors :func:`pcspot.data.loader._events_from_array` but is kept
    local so we do not depend on that module's private symbol.
    """
    arr = half.array
    if arr.size == 0:
        return []
    cls = arr[:, COL_CLASS]
    mask = cls != 0
    rows = arr[mask]
    out: list[EventLabel] = []
    for r in rows:
        out.append(
            EventLabel(
                frame=int(r[COL_FRAME]),
                player_id=int(r[COL_PLAYER_ID]),
                class_id=int(r[COL_CLASS]),
            )
        )
    return out


def _format_step_log(record: TrainStepLog) -> str:
    return (
        f"  step={record.step:6d} "
        f"loss={record.total_loss:.4f} "
        f"bce={record.bce_loss:.4f} "
        f"tmse={record.tmse_loss:.4f} "
        f"obj={record.objectness_loss:.4f} "
        f"lr={record.learning_rate:.2e}"
    )


def _format_epoch_log(record: EpochLog) -> str:
    return (
        f"epoch {record.epoch:3d} "
        f"steps={record.num_steps} "
        f"avg_loss={record.avg_total_loss:.4f} "
        f"avg_bce={record.avg_bce_loss:.4f} "
        f"lr={record.last_learning_rate:.2e} "
        f"val={record.validation}"
    )


def _best_is_better(new: float, old: float, mode: str) -> bool:
    return new > old if mode == "max" else new < old


def make_log_fn(
    *,
    wandb_run: Any = None,
    print_fn: Callable[[str], None] = print,
    extra_sinks: Sequence[Callable[[Any], None]] = (),
    best_metric: Optional[str] = None,
) -> Callable[[TrainStepLog | EpochLog], None]:
    """Build a ``log_fn`` callback for :func:`Trainer.fit`.

    Always prints to stdout (preserving the original CLI behaviour). If
    ``wandb_run`` is not ``None``, also forwards step/epoch metrics to
    Weights & Biases on the same step axis as the trainer. Each entry in
    ``extra_sinks`` is invoked with the raw record (e.g. a
    metrics.jsonl appender).

    When ``best_metric`` is given (e.g. ``"val/map_joint"``) the running best
    of that metric is tracked and mirrored into ``wandb_run.summary`` as
    ``best/<metric>`` plus the epoch it occurred on, so a sweep's summary table
    and parallel-coordinates plot reflect the best epoch rather than the last.
    The direction is inferred from the name: metrics containing ``loss`` /
    ``error`` are minimised, everything else maximised.
    """
    best_mode = (
        "min"
        if best_metric and any(t in best_metric.lower() for t in ("loss", "error"))
        else "max"
    )
    state: dict[str, Any] = {"last_step": 0, "best": None, "epoch_t0": None}

    def _log(record: TrainStepLog | EpochLog) -> None:
        for sink in extra_sinks:
            try:
                sink(record)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"warning: log sink failed: {exc}", file=sys.stderr)
        if isinstance(record, TrainStepLog):
            state["last_step"] = int(record.step)
            if state["epoch_t0"] is None:
                state["epoch_t0"] = time.perf_counter()
            print_fn(_format_step_log(record))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_total": float(record.total_loss),
                        "train/loss_bce": float(record.bce_loss),
                        "train/loss_tmse": float(record.tmse_loss),
                        "train/loss_objectness": float(record.objectness_loss),
                        "train/learning_rate": float(record.learning_rate),
                        "train/step": int(record.step),
                    },
                    step=int(record.step),
                )
        elif isinstance(record, EpochLog):
            print_fn(_format_epoch_log(record))
            if wandb_run is not None:
                payload: dict[str, float] = {
                    "epoch": int(record.epoch),
                    "train/avg_loss_total": float(record.avg_total_loss),
                    "train/avg_loss_bce": float(record.avg_bce_loss),
                    "train/avg_loss_tmse": float(record.avg_tmse_loss),
                    "train/avg_loss_objectness": float(record.avg_objectness_loss),
                    "train/last_learning_rate": float(record.last_learning_rate),
                    "train/num_steps_this_epoch": int(record.num_steps),
                }
                # Wall-clock timing + throughput for this epoch.
                if state["epoch_t0"] is not None:
                    epoch_secs = time.perf_counter() - state["epoch_t0"]
                    payload["time/epoch_seconds"] = float(epoch_secs)
                    if record.num_steps and epoch_secs > 0:
                        payload["time/steps_per_second"] = float(
                            record.num_steps / epoch_secs
                        )
                    state["epoch_t0"] = time.perf_counter()
                for k, v in record.validation.items():
                    try:
                        payload[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                # Track the running best of the chosen metric in run summary.
                if best_metric is not None and best_metric in payload:
                    cur = payload[best_metric]
                    if state["best"] is None or _best_is_better(
                        cur, state["best"], best_mode
                    ):
                        state["best"] = cur
                        try:
                            wandb_run.summary[f"best/{best_metric}"] = cur
                            wandb_run.summary["best/epoch"] = int(record.epoch)
                        except Exception:  # pragma: no cover - defensive
                            pass
                wandb_run.log(payload, step=state["last_step"])

    return _log


def _move_batch_to_device(batch, device: str):
    for name in batch.__dataclass_fields__:
        v = getattr(batch, name)
        if isinstance(v, torch.Tensor):
            setattr(batch, name, v.to(device))
    return batch


def make_validation_fn(
    *,
    val_dataset: PCBASDataset,
    val_gt_per_half: dict[tuple[str, str], list[EventLabel]],
    num_classes: int,
    decode_threshold: float,
    nms_radius: int,
    nms_mode: str,
    tolerances: Sequence[int],
    device: str,
) -> Callable[[Trainer], dict[str, float]]:
    """Build a ``validation_fn`` callback for :func:`Trainer.fit`.

    The closure runs the model over every window in ``val_dataset``,
    decodes predictions, applies per-half NMS, then computes
    Average-mAP / joint Average-mAP / player identity accuracy via
    :func:`pcspot.eval.metrics.average_map_at_tolerances`. Halves are
    disambiguated with a large per-half frame offset so cross-half
    matches cannot inflate scores.
    """
    class_ids = list(range(1, int(num_classes) + 1))
    sorted_keys = sorted(val_gt_per_half.keys())
    tols = [int(t) for t in tolerances if int(t) > 0]
    if not tols:
        raise ValueError("validation tolerances must be a non-empty list of positive integers")

    def _validate(trainer: Trainer) -> dict[str, float]:
        trainer.model.eval()
        preds_by_half: dict[tuple[str, str], list[Prediction]] = {}
        with torch.inference_mode():
            for i in range(len(val_dataset)):
                stacked, _ = val_dataset[i]
                batch = stacked_to_batch([stacked])
                batch = _move_batch_to_device(batch, device)
                outputs = trainer.model(batch)
                logits = outputs["logits"][0].detach().cpu()
                confidence = (
                    outputs["confidence"][0].detach().cpu()
                    if "confidence" in outputs
                    else None
                )
                preds = decode_predictions(
                    logits=logits,
                    confidence=confidence,
                    valid_mask=stacked.valid_mask,
                    player_ids=stacked.player_ids,
                    score_threshold=float(decode_threshold),
                )
                if not preds:
                    continue
                window_frames = stacked.frames.tolist()
                key = (
                    str(stacked.meta.match_id),
                    str(stacked.meta.half_id) if stacked.meta.half_id is not None else "",
                )
                bucket = preds_by_half.setdefault(key, [])
                for p in preds:
                    bucket.append(
                        Prediction(
                            time=int(window_frames[p.time]),
                            class_id=int(p.class_id),
                            player_id=int(p.player_id),
                            score=float(p.score),
                        )
                    )

        all_preds: list[Prediction] = []
        all_gts: list[EventLabel] = []
        # Iterate the union of GT halves and prediction halves so a half
        # with predictions but no GT entries still adds false positives
        # (and vice versa).
        keys = sorted(set(sorted_keys) | set(preds_by_half.keys()))
        for idx, key in enumerate(keys):
            offset = idx * _HALF_FRAME_OFFSET
            half_preds = preds_by_half.get(key, [])
            nms_preds = player_centric_nms(
                half_preds, window_radius=int(nms_radius), mode=str(nms_mode)
            )
            for p in nms_preds:
                all_preds.append(
                    Prediction(
                        time=int(p.time) + offset,
                        class_id=int(p.class_id),
                        player_id=int(p.player_id),
                        score=float(p.score),
                    )
                )
            for g in val_gt_per_half.get(key, []):
                all_gts.append(
                    EventLabel(
                        frame=int(g.frame) + offset,
                        player_id=int(g.player_id),
                        class_id=int(g.class_id),
                    )
                )

        summaries = average_map_at_tolerances(
            all_preds,
            all_gts,
            tolerances=list(tols),
            class_ids=class_ids,
        )

        metrics: dict[str, float] = {
            "val/num_predictions": float(len(all_preds)),
            "val/num_ground_truth": float(len(all_gts)),
        }
        for s in summaries:
            metrics[f"val/map_at_t{s.tolerance}"] = float(s.average_map)
            metrics[f"val/map_joint_at_t{s.tolerance}"] = float(s.average_map_joint)
            metrics[f"val/player_id_acc_at_t{s.tolerance}"] = float(
                s.player_identity_accuracy
            )
        if summaries:
            metrics["val/map"] = float(np.mean([s.average_map for s in summaries]))
            metrics["val/map_joint"] = float(
                np.mean([s.average_map_joint for s in summaries])
            )
            metrics["val/player_identity_accuracy"] = float(
                np.mean([s.player_identity_accuracy for s in summaries])
            )
        return metrics

    return _validate


def _hash_file(path: Path) -> Optional[str]:
    """Return a short sha256 hex digest of a file, or None on error."""
    import hashlib

    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _collect_env_info() -> dict[str, Any]:
    """Best-effort snapshot of the runtime for reproducibility.

    Captures Python version, torch version, CUDA device name, and the
    current git commit when available. All fields are optional and the
    helper never raises; missing values are recorded as ``None``.
    """
    import platform
    import subprocess

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
    }
    try:
        if info["torch_cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception:
        info["cuda_device"] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
    except Exception:
        info["git_commit"] = None
    return info


def _visual_cache_metadata(cache: Any) -> dict[str, Any]:
    """Read the visual cache sidecar metadata, if available."""
    if cache is None:
        return {}
    meta: dict[str, Any] = {}
    for attr in ("root", "backbone_name"):
        if hasattr(cache, attr):
            meta[attr] = str(getattr(cache, attr))
    if hasattr(cache, "feature_dim"):
        try:
            meta["feature_dim"] = int(cache.feature_dim())
        except Exception:
            pass
    return meta


def _build_run_info(
    *,
    args: argparse.Namespace,
    calf_config: CalfConfig,
    n_per_epoch: int,
    steps_per_epoch: int,
    total_steps: int,
    num_classes: int,
    train_halves: list[HalfArray],
    val_halves: list[HalfArray],
    train_config_path: Optional[Path],
    sampler_mode: str,
    ds_train_len: int,
    visual_cache: Any = None,
) -> dict[str, Any]:
    """Assemble the metadata dict written to ``<output_dir>/run.json``.

    Includes argparse args, CALF config, dataset size / event totals,
    sampler description, visual cache metadata, and a best-effort
    environment snapshot. Token / password fields are never copied in.
    """
    payload: dict[str, Any] = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        # ``asdict`` leaves ``attacking_classes`` / ``duel_classes`` as
        # frozensets, which ``json.dumps`` cannot serialise; ``_calf_serializable``
        # renders them as sorted lists.
        "calf_config": _calf_serializable(calf_config),
        "num_train_windows": int(ds_train_len),
        "samples_per_epoch": int(n_per_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "num_classes": int(num_classes),
        "sampler_mode": sampler_mode,
        "train_halves": [
            {"match_id": str(h.match_id), "half_id": str(h.half_id)}
            for h in train_halves
        ],
        "val_halves": [
            {"match_id": str(h.match_id), "half_id": str(h.half_id)}
            for h in val_halves
        ],
        "train_events": sum(
            int((h.array[:, COL_CLASS] != 0).sum()) if h.array.size else 0
            for h in train_halves
        ),
        "val_events": sum(
            int((h.array[:, COL_CLASS] != 0).sum()) if h.array.size else 0
            for h in val_halves
        ),
        "env": _collect_env_info(),
    }
    if train_config_path is not None:
        payload["train_config"] = {
            "path": str(train_config_path),
            "sha256": _hash_file(Path(train_config_path)),
        }
    splits_path = getattr(args, "splits", None)
    if splits_path is not None:
        payload["splits_manifest"] = {
            "path": str(splits_path),
            "sha256": _hash_file(Path(splits_path)),
        }
    if visual_cache is not None:
        payload["visual_cache"] = _visual_cache_metadata(visual_cache)
    return payload


def _make_metrics_logger(output_dir: Path) -> Callable[[Any], None]:
    """Return a log_fn-friendly callback that appends to ``metrics.jsonl``.

    Each record is a single-line JSON object containing the dataclass
    fields of either a ``TrainStepLog`` or an ``EpochLog``, tagged with
    ``kind``. The file is opened in append mode so resumed runs keep
    growing the same log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.jsonl"

    def _log(record: Any) -> None:
        try:
            if isinstance(record, TrainStepLog):
                payload = {
                    "kind": "step",
                    "step": int(record.step),
                    "total_loss": float(record.total_loss),
                    "bce_loss": float(record.bce_loss),
                    "tmse_loss": float(record.tmse_loss),
                    "objectness_loss": float(record.objectness_loss),
                    "learning_rate": float(record.learning_rate),
                }
            elif isinstance(record, EpochLog):
                payload = {
                    "kind": "epoch",
                    "epoch": int(record.epoch),
                    "num_steps": int(record.num_steps),
                    "avg_total_loss": float(record.avg_total_loss),
                    "avg_bce_loss": float(record.avg_bce_loss),
                    "avg_tmse_loss": float(record.avg_tmse_loss),
                    "avg_objectness_loss": float(record.avg_objectness_loss),
                    "last_learning_rate": float(record.last_learning_rate),
                    "validation": {
                        str(k): float(v)
                        for k, v in record.validation.items()
                        if _is_floatable(v)
                    },
                }
            else:
                return
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except OSError as exc:
            print(f"warning: metrics.jsonl write failed: {exc}", file=sys.stderr)

    return _log


def _is_floatable(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _init_wandb(args: argparse.Namespace, run_info: dict[str, Any]) -> Any:
    """Initialise a Weights & Biases run, or return ``None`` on failure.

    Authentication is read from the environment (``WANDB_API_KEY`` /
    ``wandb login``); nothing is read from ``config.toml``. Failures are
    converted to warnings so training is never blocked by a logging
    integration problem.
    """
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"warning: wandb requested but import failed: {exc}", file=sys.stderr)
        return None

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "config": run_info,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    if args.wandb_tags:
        init_kwargs["tags"] = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if args.wandb_notes:
        init_kwargs["notes"] = args.wandb_notes
    if args.wandb_run_dir:
        init_kwargs["dir"] = str(args.wandb_run_dir)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"warning: wandb.init failed: {exc}", file=sys.stderr)
        return None
    return run


def _log_wandb_artifacts(
    wandb_run: Any,
    *,
    output_dir: Path,
    run_info_path: Optional[Path],
) -> None:
    """Best-effort upload of the run.json plus checkpoint files."""
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"warning: wandb artifact upload skipped (import failed): {exc}",
              file=sys.stderr)
        return
    try:
        artifact = wandb.Artifact(
            name=f"{wandb_run.id}-checkpoints",
            type="model",
        )
        if run_info_path is not None and run_info_path.exists():
            artifact.add_file(str(run_info_path), name="run.json")
        for path in sorted(output_dir.glob("*.pt")):
            artifact.add_file(str(path), name=path.name)
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"warning: wandb artifact upload failed: {exc}", file=sys.stderr)


_KNOWN_SECTIONS = {"train", "validation", "wandb", "calf"}

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
    "use_zone_nodes",
    "zone_grid",
    "use_jersey",
    "use_goal_distances",
    "use_radius_edges",
    "radius",
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


def _load_train_config(path: Path) -> dict[str, Any]:
    """Load a check-in-able training config TOML into argparse defaults.

    Returns a flat ``{dest: value}`` dict keyed by argparse ``dest``
    names so it can be passed straight to :func:`_build_argparser`.
    Unknown sections or keys abort with a clear ``SystemExit`` so typos
    never silently change a run. The data-mirror ``config.toml`` is
    unrelated and intentionally not touched by this loader.
    """
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
            if k not in _KNOWN_TRAIN_KEYS:
                _abort(
                    f"--train-config {path}: unknown key [train].{k}; "
                    f"expected one of {sorted(_KNOWN_TRAIN_KEYS)}."
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


def _build_argparser(defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    d = defaults or {}

    def _df(name: str, builtin: Any) -> Any:
        return d.get(name, builtin)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help=(
            "Optional reusable training config TOML (see "
            "configs/train/baseline.toml). CLI flags still override "
            "anything set here; this file should NOT contain secrets."
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


def _parse_zone_grid(spec: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Parse a ``zone_grid`` config value into a ``(Gx, Gy)`` tuple.

    Accepts ``"6x4"`` / ``"6,4"`` from the CLI, or a 2-element list /
    tuple straight from a TOML config. Raises ``SystemExit`` with a
    clear message on malformed input so it surfaces during arg parsing
    rather than mid-training.
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


def run(args: argparse.Namespace, *, wandb_run: Any = None) -> int:
    """Execute a training run from a pre-built Namespace.

    Called by :func:`main` for normal CLI use, or directly by
    ``scripts/sweep.py`` with a W&B run already initialized by the
    sweep agent (``wandb_run`` passed in, ``_init_wandb`` skipped).
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolved from the Namespace so run() is self-contained whether it
    # is driven by main() or called directly (e.g. scripts/sweep.py).
    train_config_path = getattr(args, "train_config", None)

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2
    data_dir = _load_output_dir(config_path)

    if not args.splits.exists():
        print(f"Splits manifest not found: {args.splits}", file=sys.stderr)
        return 2

    manifest = SplitManifest.from_json(args.splits)
    if "train" not in {v for v in manifest.assignments.values()}:
        print("Manifest does not contain a 'train' split.", file=sys.stderr)
        return 2

    if args.validation_split is not None and args.validation_split not in {
        v for v in manifest.assignments.values()
    }:
        print(
            f"Validation split {args.validation_split!r} is not in the manifest.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading halves from {data_dir} ...")
    all_halves = load_halves_from_pcbas(data_dir)
    print(f"  Discovered {len(all_halves)} half(s) on disk.")

    train_halves, missing_train = _filter_halves(all_halves, manifest, "train")
    if missing_train:
        print(
            f"warning: manifest references {len(missing_train)} train half(s) "
            f"not found on disk: {sorted(missing_train)}",
            file=sys.stderr,
        )

    visual_cache = None
    if args.visual_cache is not None:
        visual_cache = _make_visual_cache(
            args.visual_cache,
            args.visual_backbone,
            lru_capacity=int(args.visual_cache_capacity),
        )
        print(f"  Visual cache: {args.visual_cache} (backbone={args.visual_backbone})")
        try:
            cache_dim = int(visual_cache.feature_dim())
        except Exception as exc:
            print(
                f"warning: could not read visual cache metadata "
                f"({type(exc).__name__}); skipping --visual-dim check.",
                file=sys.stderr,
            )
        else:
            if int(args.visual_dim) > 0 and int(args.visual_dim) != cache_dim:
                print(
                    f"--visual-dim ({args.visual_dim}) does not match the "
                    f"visual cache feature dim ({cache_dim}). Update the "
                    f"flag (or the train-config) before retrying.",
                    file=sys.stderr,
                )
                return 2

    calf_config = _build_calf_config(getattr(args, "calf_overrides", None))
    target_cache = None
    if args.target_cache is not None:
        from pcspot.data.cache import TargetCache
        target_cache = TargetCache(args.target_cache, config=calf_config)

    ds_train = PCBASDataset(
        manifest=manifest,
        split="train",
        window_size=args.window_size,
        stride=args.stride,
        halves=train_halves,
        calf_config=calf_config,
        target_cache=target_cache,
        visual_feature_cache=visual_cache,
    )
    print(f"  Train windows: {len(ds_train)}")
    if len(ds_train) == 0:
        print("No training windows in the manifest. Aborting.", file=sys.stderr)
        return 1

    # Sampler dispatch. Each mode produces a per-epoch index ``sampler_factory``.
    # With --num-workers 0 (default) the legacy paths run: 'sequential'
    # materialises every window once; 'mixed'/'uniform' stream batches through
    # build_dataset_batch_provider so the full window list never lives in memory
    # at once. With --num-workers > 0 every mode is routed through the
    # multi-worker collated DataLoader provider, which performs the padding,
    # visual-feature alignment, and CALF/objectness target building in worker
    # processes so they overlap with GPU compute instead of stalling one core.
    sampler_mode = str(args.sampler).lower()
    samples: list = []
    batch_provider = None
    sampler_factory: Optional[Callable[[int], object]] = None
    samples_per_epoch = int(args.samples_per_epoch) if args.samples_per_epoch else None
    num_workers = max(0, int(args.num_workers))
    use_dataloader = num_workers > 0
    pin_memory = (
        str(args.device).startswith("cuda")
        if args.pin_memory is None
        else bool(args.pin_memory)
    )

    if sampler_mode == "sequential":
        n_per_epoch = len(ds_train)
        if use_dataloader:
            _n = len(ds_train)
            def _sequential_factory(epoch: int, _n: int = _n) -> range:
                return range(_n)
            sampler_factory = _sequential_factory
        else:
            print("Materialising training samples (sampler=sequential) ...")
            samples = _materialize_samples(ds_train)
            n_per_epoch = len(samples)
    elif sampler_mode == "mixed":
        if samples_per_epoch is None:
            samples_per_epoch = len(ds_train)
        sampler_factory = reseeded_mixed_sampler_factory(
            ds_train.event_counts,
            positive_ratio=float(args.positive_ratio),
            event_weighting=str(args.event_weighting),
            num_samples=int(samples_per_epoch),
            base_seed=args.sampler_seed,
        )
        n_per_epoch = int(samples_per_epoch)
        print(
            f"  Sampler: mixed positive_ratio={args.positive_ratio} "
            f"weighting={args.event_weighting} samples_per_epoch={n_per_epoch}"
        )
        if not use_dataloader:
            batch_provider = build_dataset_batch_provider(
                ds_train,
                sampler_factory=sampler_factory,
                batch_size=int(args.batch_size),
            )
    elif sampler_mode == "uniform":
        from pcspot.data.sampling import UniformWindowSampler
        if samples_per_epoch is None:
            samples_per_epoch = len(ds_train)
        def _uniform_factory(epoch: int):
            return UniformWindowSampler(num_items=int(samples_per_epoch))  # type: ignore[arg-type]
        sampler_factory = _uniform_factory
        n_per_epoch = int(samples_per_epoch)
        print(f"  Sampler: uniform samples_per_epoch={n_per_epoch}")
        if not use_dataloader:
            batch_provider = build_dataset_batch_provider(
                ds_train,
                sampler_factory=sampler_factory,
                batch_size=int(args.batch_size),
            )
    else:
        # argparse choices keep this unreachable; defensive guard.
        print(f"Unknown --sampler {sampler_mode!r}", file=sys.stderr)
        return 2

    if use_dataloader:
        batch_provider = build_collated_dataloader_provider(
            ds_train,
            sampler_factory=sampler_factory,  # type: ignore[arg-type]
            batch_size=int(args.batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
        )
        print(
            f"  DataLoader: num_workers={num_workers} "
            f"prefetch_factor={args.prefetch_factor} "
            f"persistent_workers={bool(args.persistent_workers)} "
            f"pin_memory={pin_memory}"
        )

    val_dataset: Optional[PCBASDataset] = None
    val_gt_per_half: dict[tuple[str, str], list[EventLabel]] = {}
    val_tolerances: list[int] = []
    val_halves: list[HalfArray] = []
    if args.validation_split is not None:
        val_halves, missing_val = _filter_halves(
            all_halves, manifest, args.validation_split
        )
        if missing_val:
            print(
                f"warning: manifest references {len(missing_val)} "
                f"{args.validation_split!r} half(s) not found on disk: "
                f"{sorted(missing_val)}",
                file=sys.stderr,
            )
        if not val_halves:
            print(
                f"warning: no validation halves available for split "
                f"{args.validation_split!r}; skipping validation.",
                file=sys.stderr,
            )
        else:
            val_stride = (
                int(args.validation_stride)
                if args.validation_stride is not None
                else int(args.window_size)
            )
            val_dataset = PCBASDataset(
                manifest=manifest,
                split=args.validation_split,
                window_size=args.window_size,
                stride=val_stride,
                halves=val_halves,
                calf_config=calf_config,
                visual_feature_cache=visual_cache,
                compute_targets=False,
            )
            for h in val_halves:
                key = (str(h.match_id), str(h.half_id))
                val_gt_per_half[key] = _gt_events_for_half(h)
            val_tolerances = _parse_tolerances(args.metric_tolerances)
            total_gt = sum(len(v) for v in val_gt_per_half.values())
            print(
                f"  Validation: split={args.validation_split!r} "
                f"halves={len(val_halves)} windows={len(val_dataset)} "
                f"gt_events={total_gt} tolerances={val_tolerances}"
            )

    zone_grid = _parse_zone_grid(args.zone_grid)
    model = PlayerCentricSpottingModel(
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
    steps_per_epoch = max(
        1, int(n_per_epoch) // max(1, int(args.batch_size) * max(1, int(args.grad_accum_steps)))
    )
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    schedule = WarmupCosineSchedule(
        total_steps=total_steps,
        warmup_steps=min(args.warmup_steps, max(0, total_steps - 1)),
    )

    trainer = Trainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        objectness_lambda=1.0 if args.objectness else 0.0,
        calf_config=calf_config,
        device=args.device,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
        schedule=schedule,
        grad_accum_steps=int(args.grad_accum_steps),
        amp_enabled=bool(args.amp),
        amp_dtype=str(args.amp_dtype),
        compile=bool(args.compile),
    )

    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            print(f"--resume checkpoint not found: {resume_path}", file=sys.stderr)
            return 2
        print(f"Resuming from {resume_path}")
        trainer.load_checkpoint(resume_path)
        # The trainer's epoch counter is now set to the last completed
        # epoch; Trainer.fit will use it as the starting epoch.
        print(
            f"  resumed at global_step={trainer.global_step} "
            f"epoch={trainer.epoch}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_info = _build_run_info(
        args=args,
        calf_config=calf_config,
        n_per_epoch=n_per_epoch,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        num_classes=int(model.num_classes),
        train_halves=train_halves,
        val_halves=(val_halves if args.validation_split is not None else []),
        train_config_path=train_config_path,
        sampler_mode=sampler_mode,
        ds_train_len=len(ds_train),
        visual_cache=visual_cache,
    )
    run_info_path = args.output_dir / "run.json"
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    if wandb_run is None and args.wandb:
        wandb_run = _init_wandb(args, run_info)

    if wandb_run is not None:
        # One-time descriptive summary so each (sweep) run is self-describing in
        # the W&B UI without digging into run.json. Written to summary (not
        # config) to avoid clobbering swept hyperparameters.
        try:
            num_params = int(sum(p.numel() for p in model.parameters()))
            total_gt = sum(len(v) for v in val_gt_per_half.values())
            wandb_run.summary.update(
                {
                    "data/train_halves": len(train_halves),
                    "data/train_windows": int(len(ds_train)),
                    "data/val_halves": len(val_halves),
                    "data/val_windows": int(len(val_dataset)) if val_dataset is not None else 0,
                    "data/val_gt_events": int(total_gt),
                    "run/sampler": sampler_mode,
                    "run/num_workers": int(num_workers),
                    "run/use_dataloader": bool(use_dataloader),
                    "run/samples_per_epoch": int(n_per_epoch),
                    "run/steps_per_epoch": int(steps_per_epoch),
                    "run/total_steps": int(total_steps),
                    "model/num_params": num_params,
                    "model/num_classes": int(model.num_classes),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"warning: wandb summary.update failed: {exc}", file=sys.stderr)

    validation_fn: Optional[Callable[[Trainer], dict[str, float]]] = None
    if val_dataset is not None:
        validation_fn = make_validation_fn(
            val_dataset=val_dataset,
            val_gt_per_half=val_gt_per_half,
            num_classes=int(model.num_classes),
            decode_threshold=args.decode_threshold,
            nms_radius=args.nms_radius,
            nms_mode=args.nms_mode,
            tolerances=val_tolerances,
            device=args.device,
        )

    metrics_sink = _make_metrics_logger(args.output_dir)
    log_fn = make_log_fn(
        wandb_run=wandb_run,
        extra_sinks=(metrics_sink,),
        best_metric=args.keep_best_metric,
    )

    print(
        f"Training: epochs={args.epochs} batch_size={args.batch_size} "
        f"grad_accum={args.grad_accum_steps} amp={args.amp} "
        f"sampler={sampler_mode} steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps} "
        f"wandb={'on' if wandb_run is not None else 'off'} "
        f"validation={'on' if validation_fn is not None else 'off'}"
    )
    # Provider path skips materialising the sample list; sequential path
    # still works because trainer.fit detects which input it received.
    train_input = batch_provider if batch_provider is not None else samples
    # When resuming, ``trainer.epoch`` already points at the last
    # completed epoch; advance the bookkeeping by one so we don't
    # overwrite the existing checkpoint and so the W&B step axis
    # continues forward.
    start_epoch: Optional[int] = None
    if args.resume is not None:
        start_epoch = int(trainer.epoch) + 1
    try:
        trainer.fit(
            train_input,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fn=validation_fn,
            checkpoint_dir=args.output_dir,
            keep_best_metric=args.keep_best_metric,
            log_fn=log_fn,
            start_epoch=start_epoch,
        )
    finally:
        if wandb_run is not None and args.wandb_log_artifacts:
            _log_wandb_artifacts(
                wandb_run,
                output_dir=args.output_dir,
                run_info_path=run_info_path,
            )
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                print(f"warning: wandb_run.finish failed: {exc}", file=sys.stderr)

    print(f"Done. Checkpoints at {args.output_dir}")
    return 0


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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
