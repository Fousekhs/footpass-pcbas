"""Shared standalone-checkpoint evaluation for every model variant.

Generalises the model-reconstruction half of ``scripts/infer.py``
(``_resolve_model_kwargs`` / ``_make_model``) so each variant's
``scripts/<variant>/eval.py`` only needs to name its model class and
the kwargs ``run.json`` should be mined for. Reuses
:func:`pcspot.train.runner.make_validation_fn` to score a checkpoint
against a chosen split with the exact same metric pipeline used during
training, so standalone numbers match the training-time validation log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import load_halves_from_pcbas
from pcspot.data.splits import SplitManifest
from pcspot.train.cli_common import _parse_tolerances
from pcspot.train.runner import (
    _filter_halves,
    _gt_events_for_half,
    _load_output_dir,
    _make_visual_cache,
    make_validation_fn,
)


def resolve_model_kwargs(
    checkpoint_path: Path,
    overrides: dict[str, Any],
    *,
    model_init_keys: Sequence[str],
) -> dict[str, Any]:
    """Recover the kwargs used to construct the checkpoint's model.

    Looks for ``run.json`` next to the checkpoint (written by
    ``scripts/<variant>/train.py`` via
    :func:`pcspot.train.runner._build_run_info`) and pulls the model
    construction args — restricted to ``model_init_keys`` — from its
    ``args`` payload. ``overrides`` (typically CLI flags) win.
    """
    kwargs: dict[str, Any] = {}
    run_json = checkpoint_path.parent / "run.json"
    if run_json.exists():
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
            train_args = data.get("args", {}) if isinstance(data, dict) else {}
            for k in model_init_keys:
                if k in train_args and train_args[k] is not None:
                    kwargs[k] = train_args[k]
        except (json.JSONDecodeError, OSError):
            pass
    for k, v in overrides.items():
        if v is not None:
            kwargs[k] = v
    return kwargs


def make_model(
    checkpoint_path: Path,
    device: str,
    model_kwargs: dict[str, Any],
    *,
    model_cls: type[nn.Module],
) -> nn.Module:
    """Reconstruct ``model_cls(**model_kwargs)`` and load its checkpoint weights."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise RuntimeError(
            f"Unexpected checkpoint payload at {checkpoint_path}: "
            f"{type(payload).__name__}"
        )
    model = model_cls(**model_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    config: Path,
    splits: Path,
    split: str,
    model_cls: type[nn.Module],
    model_init_keys: Sequence[str],
    model_kwarg_overrides: dict[str, Any],
    device: str = "cpu",
    window_size: int = 128,
    stride: int | None = None,
    visual_cache: Path | None = None,
    visual_backbone: str = "dinov2_vits14",
    decode_threshold: float = 0.5,
    nms_radius: int = 12,
    nms_mode: str = "per_player_class",
    metric_tolerances: str = "3,12,25",
) -> dict[str, float]:
    """Reconstruct a checkpoint's model and score it against ``split``.

    Builds a :class:`PCBASDataset` for ``split``, reuses
    :func:`pcspot.train.runner.make_validation_fn` so the metric
    pipeline matches per-epoch training validation exactly, and returns
    the resulting ``dict[str, float]``.
    """
    config_path = config.resolve()
    data_dir = _load_output_dir(config_path)
    manifest = SplitManifest.from_json(splits)
    all_halves = load_halves_from_pcbas(data_dir)
    halves, missing = _filter_halves(all_halves, manifest, split)
    if missing:
        print(
            f"warning: manifest references {len(missing)} {split!r} half(s) "
            f"not found on disk: {sorted(missing)}",
            file=sys.stderr,
        )
    if not halves:
        raise RuntimeError(f"No halves available for split {split!r}.")

    cache = None
    if visual_cache is not None:
        cache = _make_visual_cache(visual_cache, visual_backbone)

    eval_stride = int(stride) if stride is not None else int(window_size)
    dataset = PCBASDataset(
        manifest=manifest,
        split=split,
        window_size=window_size,
        stride=eval_stride,
        halves=halves,
        visual_feature_cache=cache,
        compute_targets=False,
    )

    gt_per_half = {
        (str(h.match_id), str(h.half_id)): _gt_events_for_half(h) for h in halves
    }

    model_kwargs = resolve_model_kwargs(
        checkpoint, model_kwarg_overrides, model_init_keys=model_init_keys
    )
    model = make_model(checkpoint, device, model_kwargs, model_cls=model_cls)

    class _TrainerLike:
        def __init__(self, model: nn.Module) -> None:
            self.model = model

    validation_fn = make_validation_fn(
        val_dataset=dataset,
        val_gt_per_half=gt_per_half,
        num_classes=int(model.num_classes),
        decode_threshold=decode_threshold,
        nms_radius=nms_radius,
        nms_mode=nms_mode,
        tolerances=_parse_tolerances(metric_tolerances),
        device=device,
    )
    return validation_fn(_TrainerLike(model))
