"""Run player-centric ball-action spotting inference.

Two subcommands, both writing the same JSON prediction layout:

- ``offline``: iterates :class:`pcspot.data.dataset.PCBASDataset` windows
  for a single match (tactical HDF5 already on disk; visual features
  pulled from a precomputed cache when provided). Fast; preferred when
  evaluating on held-out matches.
- ``online``: streams a broadcast video frame-by-frame through
  :class:`pcspot.inference.online.OnlineSpotter`. Builds DINOv2
  embeddings on the fly with bounded memory. Use this to mimic the
  single-pass Codabench / live-broadcast environment.

Pass ``--model-variant {graph,no_graph}`` to match the architecture the
checkpoint was trained with (default: ``graph``); this selects both the
model class and which ``run.json`` keys are used to reconstruct it.

Example::

    python scripts/infer.py offline \\
        --checkpoint checkpoints/run1/epoch_0004.pt \\
        --config config.toml --match-id game_18 \\
        --visual-cache .cache/visual --visual-backbone dinov2_vits14 \\
        --out predictions/game_18.json

    python scripts/infer.py online \\
        --checkpoint checkpoints/run1/epoch_0004.pt \\
        --config config.toml --match-id game_18 \\
        --video path/to/game_18.mp4 \\
        --out predictions/game_18_online.json
"""

from __future__ import annotations

import argparse
import json
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import (
    COL_FRAME,
    COL_PLAYER_ID,
    COL_SHIRT,
    HalfArray,
    _team_of_player,
    load_halves_from_pcbas,
)
from pcspot.data.schema import PCBAS_CLASS_NAMES
from pcspot.data.splits import SplitManifest
from pcspot.eval.nms import Prediction, decode_predictions, player_centric_nms
from pcspot.models.graph_model import PlayerCentricSpottingModel
from pcspot.models.no_graph_model import NoGraphSpottingModel
from pcspot.models.no_zones_model import NoZonesSpottingModel
from pcspot.models.pipeline import stacked_to_batch
from pcspot.train.cli_common import _parse_zone_grid


def _load_output_dir(config_path: Path) -> Path:
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("pcbas", {})
    out = Path(section.get("output_dir", "data/pcbas_one_match"))
    if not out.is_absolute():
        out = (config_path.parent / out).resolve()
    return out


def _half_matches(half_id: str, target: str) -> bool:
    """Compare ``--half H1``/``H2`` against ``HalfArray.half_id``.

    ``half_id`` is the raw HDF5 key (``"<match_id>_H<n>"``), not a bare
    ``"H1"``/``"H2"`` token, so an exact-equality check never matches.
    """
    half_id = str(half_id)
    target = str(target)
    return half_id == target or half_id.endswith(f"_{target}")


# Per-variant (model class, run.json "args" keys to mine for __init__
# kwargs). Mirrors scripts/{graph,no_zones,no_graph}/eval.py's
# MODEL_INIT_KEYS — kept here too so this general-purpose offline/online
# inference tool does not depend on the per-variant eval scripts.
_MODEL_REGISTRY: dict[str, tuple[type, tuple[str, ...]]] = {
    "graph": (
        PlayerCentricSpottingModel,
        (
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
            "use_zone_nodes",
            "zone_grid",
            "use_jersey",
            "use_goal_distances",
            "use_radius_edges",
            "radius",
        ),
    ),
    "no_zones": (
        NoZonesSpottingModel,
        (
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
        ),
    ),
    "no_graph": (
        NoGraphSpottingModel,
        (
            "hidden_dim",
            "num_classes",
            "num_mstcn_stages",
            "num_mstcn_layers",
            "global_dim",
            "visual_dim",
            "visual_proj_dim",
            "with_confidence",
            "use_acceleration",
            "use_time_features",
            "use_jersey",
            "use_goal_distances",
        ),
    ),
}


def _resolve_model_kwargs(
    checkpoint_path: Path,
    overrides: dict,
    *,
    model_init_keys: tuple[str, ...],
) -> dict:
    """Recover the kwargs used to construct the checkpoint's model.

    Looks for ``run.json`` next to the checkpoint (written by
    ``scripts/<variant>/train.py``) and pulls the model construction
    args — restricted to ``model_init_keys`` — from its ``args``
    payload. ``overrides`` (typically CLI flags) win.
    """
    kwargs: dict = {}
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
    if "zone_grid" in kwargs:
        # run.json stores the raw "--zone-grid" string (e.g. "6x4"); the
        # model __init__ expects an (int, int) tuple.
        kwargs["zone_grid"] = _parse_zone_grid(kwargs["zone_grid"])
    return kwargs


def _make_model(
    checkpoint_path: Path,
    device: str,
    model_kwargs: dict,
    *,
    model_cls: type,
):
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


def _build_jersey_lookup(halves: Iterable[HalfArray]) -> dict[int, int]:
    """Map ``player_id -> shirt number`` from tactical-array rows.

    Shirt numbers are constant for a player across a match, so a single
    lookup built from all available halves covers every prediction.
    """
    lookup: dict[int, int] = {}
    for half in halves:
        arr = half.array
        if arr.size == 0:
            continue
        for pid, shirt in zip(arr[:, COL_PLAYER_ID], arr[:, COL_SHIRT]):
            if not np.isfinite(pid) or not np.isfinite(shirt):
                continue
            lookup.setdefault(int(pid), int(shirt))
    return lookup


def _prediction_to_dict(p: Prediction, jersey_lookup: dict[int, int]) -> dict:
    return {
        "frame": int(p.time),
        "team": _team_of_player(float(p.player_id)),
        "jersey_number": int(jersey_lookup.get(int(p.player_id), -1)),
        "action_class": PCBAS_CLASS_NAMES.get(int(p.class_id), str(p.class_id)),
        "score": float(p.score),
    }


def _write_predictions(out_path: Path, preds: Iterable[Prediction], jersey_lookup: dict[int, int]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_prediction_to_dict(p, jersey_lookup) for p in preds]
    payload.sort(key=lambda x: (x["frame"], x["team"], x["jersey_number"]))
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(payload)


# --------------------------------------------------------------------- offline
def _run_offline(args: argparse.Namespace) -> int:
    data_dir = _load_output_dir(args.config.resolve())
    all_halves = load_halves_from_pcbas(data_dir, match_id=args.match_id, include_challenge=True)
    if not all_halves:
        print(f"No halves found for match-id={args.match_id} under {data_dir}",
              file=sys.stderr)
        return 1
    if args.half is not None and args.half != "both":
        all_halves = [h for h in all_halves if _half_matches(h.half_id, args.half)]
        if not all_halves:
            print(f"No half {args.half!r} for match {args.match_id!r}", file=sys.stderr)
            return 1

    jersey_lookup = _build_jersey_lookup(all_halves)

    manifest = SplitManifest.single(
        "infer",
        [(str(h.match_id), str(h.half_id)) for h in all_halves],
    )

    visual_cache = None
    if args.visual_cache is not None:
        from pcspot.features.cache import VisualFeatureCache
        visual_cache = VisualFeatureCache(
            root=args.visual_cache, backbone_name=args.visual_backbone
        )

    dataset = PCBASDataset(
        manifest=manifest,
        split="infer",
        window_size=args.window_size,
        stride=args.stride,
        halves=all_halves,
        compute_targets=False,
        visual_feature_cache=visual_cache,
    )
    print(f"Offline inference: {len(dataset)} window(s)")

    model_cls, model_init_keys = _MODEL_REGISTRY[args.model_variant]
    overrides = {
        "hidden_dim": args.hidden_dim,
        "num_mstcn_stages": args.num_mstcn_stages,
        "num_mstcn_layers": args.num_mstcn_layers,
        "visual_dim": args.visual_dim,
    }
    model_kwargs = _resolve_model_kwargs(args.checkpoint, overrides, model_init_keys=model_init_keys)
    model = _make_model(args.checkpoint, args.device, model_kwargs, model_cls=model_cls)

    all_preds: list[Prediction] = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            stacked, _ = dataset[i]
            batch = stacked_to_batch([stacked])
            for name in batch.__dataclass_fields__:
                v = getattr(batch, name)
                if isinstance(v, torch.Tensor):
                    setattr(batch, name, v.to(args.device))
            outputs = model(batch)
            logits = outputs["logits"][0].detach().cpu()
            confidence = (
                outputs["confidence"][0].detach().cpu()
                if "confidence" in outputs else None
            )
            preds = decode_predictions(
                logits=logits,
                confidence=confidence,
                valid_mask=stacked.valid_mask,
                player_ids=stacked.player_ids,
                score_threshold=args.decode_threshold,
            )
            window_frames = stacked.frames.tolist()
            absolute = [
                Prediction(
                    time=int(window_frames[p.time]),
                    class_id=int(p.class_id),
                    player_id=int(p.player_id),
                    score=float(p.score),
                )
                for p in preds
            ]
            all_preds.extend(absolute)

    nms_preds = player_centric_nms(
        all_preds, window_radius=args.nms_radius, mode=args.nms_mode
    )
    n = _write_predictions(args.out, nms_preds, jersey_lookup)
    print(f"Wrote {n} predictions to {args.out}")
    return 0


# ---------------------------------------------------------------------- online
def _run_online(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        print(f"opencv-python is required for online inference: {exc}", file=sys.stderr)
        return 2

    data_dir = _load_output_dir(args.config.resolve())
    all_halves = load_halves_from_pcbas(data_dir, match_id=args.match_id)
    if not all_halves:
        print(f"No halves found for match-id={args.match_id} under {data_dir}",
              file=sys.stderr)
        return 1
    if args.half is not None:
        all_halves = [h for h in all_halves if _half_matches(h.half_id, args.half)]
    if not all_halves:
        print("No matching halves after filtering.", file=sys.stderr)
        return 1
    half = all_halves[0]
    jersey_lookup = _build_jersey_lookup([half])

    from pcspot.features.cropper import CropperConfig, PaddedPlayerCropper
    from pcspot.features.dinov2 import DinoV2Config, DinoV2Extractor
    from pcspot.inference.online import OnlineInferenceConfig, OnlineSpotter

    model_cls, model_init_keys = _MODEL_REGISTRY[args.model_variant]
    overrides = {
        "hidden_dim": args.hidden_dim,
        "num_mstcn_stages": args.num_mstcn_stages,
        "num_mstcn_layers": args.num_mstcn_layers,
        "visual_dim": args.visual_dim,
    }
    model_kwargs = _resolve_model_kwargs(args.checkpoint, overrides, model_init_keys=model_init_keys)
    model = _make_model(args.checkpoint, args.device, model_kwargs, model_cls=model_cls)
    cropper = PaddedPlayerCropper(CropperConfig(crop_size=args.crop_size,
                                                pad_factor=args.pad_factor))
    dino_cfg = DinoV2Config(
        backbone_name=args.visual_backbone,
        crop_size=args.crop_size,
        device=args.device,
        use_stub=args.use_stub,
    )
    extractor = DinoV2Extractor(dino_cfg)

    online_cfg = OnlineInferenceConfig(
        window_size=args.window_size,
        stride=args.stride,
        nms_radius=args.nms_radius,
        score_threshold=args.decode_threshold,
        device=args.device,
        match_id=str(half.match_id),
        half_id=str(half.half_id),
        fps=args.fps,
    )
    spotter = OnlineSpotter(model, extractor, cropper, online_cfg)

    arr = half.array
    frames_col = arr[:, COL_FRAME].astype(np.int64)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1

    all_preds: list[Prediction] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lo = int(np.searchsorted(frames_col, frame_index, side="left"))
            hi = int(np.searchsorted(frames_col, frame_index + 1, side="left"))
            tactical_rows = arr[lo:hi]
            preds = spotter.step(frame_index, frame_rgb, tactical_rows)
            all_preds.extend(preds)
            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        cap.release()

    n = _write_predictions(args.out, all_preds, jersey_lookup)
    print(f"Wrote {n} predictions ({frame_index} frames processed) to {args.out}")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-variant", choices=sorted(_MODEL_REGISTRY), default="graph",
                        help="Which model architecture the checkpoint was trained with.")
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--config", type=Path, default=Path("config.toml"))
    common.add_argument("--match-id", required=True)
    common.add_argument("--half", default=None,
                        help="Limit to a single half (e.g. H1). Default: all halves.")
    common.add_argument("--device", default="cpu")
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--window-size", type=int, default=128)
    common.add_argument("--decode-threshold", type=float, default=0.5)
    common.add_argument("--nms-mode", default="per_player_class",
                        choices=["per_player_class", "per_player", "per_class"])
    common.add_argument("--nms-radius", type=int, default=12)
    common.add_argument("--fps", type=float, default=25.0)
    # Model construction overrides. Defaults are None so values are pulled
    # from `run.json` next to the checkpoint when available.
    common.add_argument("--hidden-dim", type=int, default=None)
    common.add_argument("--num-mstcn-stages", type=int, default=None)
    common.add_argument("--num-mstcn-layers", type=int, default=None)
    common.add_argument("--visual-dim", type=int, default=None)

    pa = sub.add_parser("offline", parents=[common],
                        help="Batched inference using precomputed visual cache.")
    pa.add_argument("--stride", type=int, default=96)
    pa.add_argument("--visual-cache", type=Path, default=None)
    pa.add_argument("--visual-backbone", default="dinov2_vits14")
    pa.set_defaults(func=_run_offline)

    pb = sub.add_parser("online", parents=[common],
                        help="Streaming inference, builds visual features on the fly.")
    pb.add_argument("--video", type=Path, required=True)
    pb.add_argument("--stride", type=int, default=32)
    pb.add_argument("--visual-backbone", default="dinov2_vits14")
    pb.add_argument("--crop-size", type=int, default=224)
    pb.add_argument("--pad-factor", type=float, default=1.6)
    pb.add_argument("--use-stub", action="store_true",
                    help="Use the deterministic stub backbone (smoke tests only).")
    pb.add_argument("--max-frames", type=int, default=None,
                    help="Stop after this many frames (debugging).")
    pb.set_defaults(func=_run_online)

    return p


def main() -> int:
    args = _build_argparser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
