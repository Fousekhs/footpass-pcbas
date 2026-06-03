"""Precompute per-player visual feature embeddings for PCBAS halves.

Reads the extracted PCBAS/FOOTPASS split produced by
``mirror_pcbas_one_match.py`` and, for every (match, half), iterates
the broadcast video frame-by-frame. For each frame the script:

1. Builds the list of visible tracked players using the tactical
   array's ROI columns (``roi_x``, ``roi_y``, ``roi_width``,
   ``roi_height``) and skips players with non-finite ROI rows.
2. Crops a padded square around each visible player using
   :class:`pcspot.features.cropper.PaddedPlayerCropper`.
3. Runs the frozen DINOv2 ViT-S/14 backbone in batches with
   ``torch.inference_mode()``.
4. Writes one ``.npz`` shard per match-half through
   :class:`pcspot.features.cache.VisualFeatureStore`.

The shards are then read by :class:`pcspot.data.PCBASDataset` via the
``visual_feature_cache`` argument.

Usage::

    python scripts/precompute_visual_features.py --config config.toml \
        --match-id game_18 --crop-size 224 --pad-factor 1.6 \
        --backbone dinov2_vits14

Use ``--use-stub`` for tests / smoke runs where the real DINOv2
weights are not available; the resulting cache is **not** suitable
for evaluation but exercises the full pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcbas_data import (  # noqa: E402
    COL,
    MatchAssets,
    PCBASConfig,
    list_matches,
    load_tactical_arrays,
)

from pcspot.features.cache import (  # noqa: E402
    VisualFeatureMetadata,
    VisualFeatureStore,
)
from pcspot.features.cropper import (  # noqa: E402
    CropperConfig,
    PaddedPlayerCropper,
)


VALID_SPLITS = ("TRAIN", "VAL", "CHALLENGE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Optional match filter (single match-id). Defaults to all "
        "discovered matches across the chosen --splits.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help=(
            "Comma-separated list of splits to process (TRAIN, VAL, "
            "CHALLENGE). When omitted the script falls back to the legacy "
            "behaviour of using whichever split directories the helpers "
            "happen to discover first."
        ),
    )
    p.add_argument(
        "--match-list",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file containing one match-id per line "
            "to restrict processing. Comments (lines starting with '#') and "
            "blank lines are ignored."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Cache root. Default: <output_dir>/visual_features.",
    )
    p.add_argument("--backbone", type=str, default="dinov2_vits14")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--pad-factor", type=float, default=1.6)
    p.add_argument("--min-box-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional first frame to process (defaults to the half's first frame).",
    )
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional last frame to process (inclusive).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames per half (handy for dry runs).",
    )
    p.add_argument(
        "--use-stub",
        action="store_true",
        help="Use the deterministic stub backbone (no torch.hub network call). "
        "Suitable for tests; not suitable for evaluation.",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip match-halves whose cache shard already exists on disk. "
            "Default ON; use --no-skip-existing or --overwrite to rebuild."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild shards even if they already exist (alias for --no-skip-existing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover matches and report what would be processed without "
            "loading the DINOv2 backbone or reading video frames."
        ),
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if wandb is installed.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="pcspot-features",
        help="W&B project name (default: pcspot-features).",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    return p.parse_args(argv)


def parse_splits(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    splits: list[str] = []
    for chunk in str(spec).split(","):
        s = chunk.strip().upper()
        if not s:
            continue
        if s not in VALID_SPLITS:
            raise SystemExit(
                f"Unknown split {s!r}; expected one of {sorted(VALID_SPLITS)}."
            )
        if s not in splits:
            splits.append(s)
    if not splits:
        raise SystemExit("--splits must contain at least one of TRAIN/VAL/CHALLENGE.")
    return splits


def load_match_list(path: Path) -> set[str]:
    """Read a one-id-per-line text file, ignoring comments and blanks."""
    keep: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        keep.add(s)
    if not keep:
        raise SystemExit(f"--match-list {path}: no match-ids parsed")
    return keep


def discover_matches_in_split(output_dir: Path, split: str) -> list[MatchAssets]:
    """Locate the (tactical_dir, video_dir) for one split and list its matches.

    The full mirror extracts each split into its own
    ``extracted/tactical_data_<SPLIT>/`` and ``extracted/videos_*_<SPLIT>/``
    sub-directories. This helper restricts ``list_matches`` to one split by
    pointing it at an isolated working directory whose ``extracted/`` only
    references the matching split's sub-directories via symlinks.

    For environments where symlinks are not supported (Windows without
    developer mode), the helper falls back to inspecting the HDF5 files
    directly and skipping the symlink dance.
    """
    extracted = output_dir / "extracted"
    if not extracted.exists():
        return []
    tactical_dir = _find_dir_for_split(extracted, "tactical_data_", split)
    video_dir = _find_dir_for_split(extracted, "videos_", split)
    if tactical_dir is None or video_dir is None:
        return []
    return _list_matches_in(tactical_dir, video_dir)


def _find_dir_for_split(extracted: Path, prefix: str, split: str) -> Path | None:
    """Find a sibling directory of `extracted/` matching ``<prefix>*<split>``."""
    target = f"_{split.lower()}"
    for child in sorted(extracted.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.startswith(prefix.lower()) and target in name:
            return child
    return None


def _list_matches_in(tactical_dir: Path, video_dir: Path) -> list[MatchAssets]:
    """Mirror of ``pcbas_data.list_matches`` but with explicit directories."""
    import h5py

    h5_files = sorted(tactical_dir.glob("*.h5"))
    if not h5_files:
        return []
    h5_path = h5_files[0]
    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(f.keys())

    halves_by_match: dict[str, list[str]] = {}
    for k in keys:
        parts = k.rsplit("_", 1)
        if (
            len(parts) == 2
            and parts[1].startswith("H")
            and parts[1][1:].isdigit()
        ):
            halves_by_match.setdefault(parts[0], []).append(k)
        else:
            halves_by_match.setdefault(k, []).append(k)

    matches: list[MatchAssets] = []
    for match_id, halves in halves_by_match.items():
        matches.append(
            MatchAssets(
                match_id=match_id,
                video_path=video_dir / f"{match_id}.mp4",
                halves=sorted(halves),
                tactical_h5=h5_path,
            )
        )
    return matches


def _select_matches(
    output_dir: Path,
    match_id: str | None,
    *,
    splits: list[str] | None = None,
    match_list: set[str] | None = None,
) -> list[tuple[Optional[str], MatchAssets]]:
    """Resolve the list of (split_label, MatchAssets) to process.

    - When ``splits`` is None, fall back to the legacy single-extraction
      layout via ``list_matches`` (no split label attached).
    - When ``splits`` is set, scan each split's extraction directory and
      tag every match with its split label so the manifest can report it.
    - ``match_id`` and ``match_list`` further filter the resolved set.
    """
    pairs: list[tuple[Optional[str], MatchAssets]] = []
    if splits is None:
        for m in list_matches(output_dir):
            pairs.append((None, m))
    else:
        seen: set[str] = set()
        for split in splits:
            for m in discover_matches_in_split(output_dir, split):
                if m.match_id in seen:
                    # The dataset never reuses match-ids across splits;
                    # if a duplicate appears, keep the first occurrence
                    # so split discipline reporting stays meaningful.
                    continue
                seen.add(m.match_id)
                pairs.append((split, m))

    if not pairs:
        raise SystemExit(
            "No matches discovered. Make sure scripts/mirror_pcbas_full.py "
            "(or mirror_pcbas_one_match.py) extracted at least one split."
        )

    if match_id is not None:
        pairs = [(s, m) for (s, m) in pairs if m.match_id == match_id]
        if not pairs:
            raise SystemExit(f"Match {match_id!r} not found in the chosen splits.")

    if match_list is not None:
        pairs = [(s, m) for (s, m) in pairs if m.match_id in match_list]
        if not pairs:
            raise SystemExit(
                f"--match-list filtered out every match (none of "
                f"{sorted(match_list)} are present in the chosen splits)."
            )

    return pairs


def _frame_iter(arr: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rows_for_frame)`` from a tactical array."""
    if arr.size == 0:
        return
    frames = arr[:, COL["frame"]].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    arr = arr[order]
    frames = frames[order]
    unique, idx = np.unique(frames, return_index=True)
    ends = np.append(idx[1:], frames.size)
    for f, lo, hi in zip(unique.tolist(), idx.tolist(), ends.tolist()):
        yield int(f), arr[lo:hi]


def _open_video(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    return cap


def _seek(cap, frame_index: int) -> None:
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))


def _read_frame_bgr_to_rgb(cap) -> np.ndarray | None:
    import cv2

    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resolve_frame_range(
    arr: np.ndarray, args: argparse.Namespace
) -> tuple[int, int]:
    if arr.size == 0:
        return 0, -1
    frames = arr[:, COL["frame"]].astype(np.int64)
    start = int(frames.min()) if args.start_frame is None else int(args.start_frame)
    end = int(frames.max()) if args.end_frame is None else int(args.end_frame)
    if args.max_frames is not None:
        end = min(end, start + int(args.max_frames) - 1)
    return start, end


def precompute_for_half(
    *,
    match_id: str,
    half_id: str,
    arr: np.ndarray,
    video_path: Path,
    cropper: PaddedPlayerCropper,
    extractor,
    store: VisualFeatureStore,
    args: argparse.Namespace,
    wandb_run=None,
    global_step: list[int] | None = None,
) -> int:
    """Process one match-half. Returns number of frames processed."""
    if arr.size == 0:
        print(f"[{match_id}/{half_id}] empty tactical array; skipping")
        return 0
    start_frame, end_frame = _resolve_frame_range(arr, args)
    if end_frame < start_frame:
        return 0

    if global_step is None:
        global_step = [0]

    cap = _open_video(video_path)
    try:
        _seek(cap, start_frame)
        processed = 0
        cur_frame = start_frame
        by_frame = {f: rows for f, rows in _frame_iter(arr)}

        half_t_read = half_t_crop = half_t_infer = half_t_store = 0.0
        half_start = time.perf_counter()

        while cur_frame <= end_frame:
            t0 = time.perf_counter()
            frame = _read_frame_bgr_to_rgb(cap)
            t_read = time.perf_counter() - t0
            if frame is None:
                break

            t_crop = t_infer = t_store = 0.0
            n_players = n_valid = 0

            rows = by_frame.get(int(cur_frame))
            if rows is not None and rows.shape[0] > 0:
                n_players = rows.shape[0]
                rois = rows[:, [COL["roi_x"], COL["roi_y"], COL["roi_width"], COL["roi_height"]]]
                pids = rows[:, COL["player_id"]].astype(np.int64)

                t0 = time.perf_counter()
                crops, mask = cropper.extract(frame, rois)
                t_crop = time.perf_counter() - t0

                if mask.any():
                    n_valid = int(mask.sum())
                    t0 = time.perf_counter()
                    feats = extractor.extract_features(crops[mask])
                    t_infer = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    visible_pids = pids[mask].tolist()
                    store.add_frame(
                        match_id=match_id,
                        half_id=half_id,
                        frame=int(cur_frame),
                        player_ids=visible_pids,
                        features=feats,
                        visible=[True] * feats.shape[0],
                    )
                    t_store = time.perf_counter() - t0

            half_t_read += t_read
            half_t_crop += t_crop
            half_t_infer += t_infer
            half_t_store += t_store

            processed += 1
            global_step[0] += 1

            if wandb_run is not None:
                t_frame_total = t_read + t_crop + t_infer + t_store
                wandb_run.log(
                    {
                        "frame/read_ms": t_read * 1e3,
                        "frame/crop_ms": t_crop * 1e3,
                        "frame/infer_ms": t_infer * 1e3,
                        "frame/store_ms": t_store * 1e3,
                        "frame/total_ms": t_frame_total * 1e3,
                        "frame/players_tracked": n_players,
                        "frame/players_valid": n_valid,
                        "frame/valid_crop_frac": (
                            n_valid / n_players if n_players > 0 else 0.0
                        ),
                        "frame/index": int(cur_frame),
                        "match_id": match_id,
                        "half_id": half_id,
                    },
                    step=global_step[0],
                )

            cur_frame += 1

        half_elapsed = time.perf_counter() - half_start
        if wandb_run is not None and processed > 0:
            wandb_run.log(
                {
                    "half/frames_processed": processed,
                    "half/elapsed_s": half_elapsed,
                    "half/fps": processed / half_elapsed if half_elapsed > 0 else 0.0,
                    "half/read_frac": half_t_read / half_elapsed,
                    "half/crop_frac": half_t_crop / half_elapsed,
                    "half/infer_frac": half_t_infer / half_elapsed,
                    "half/store_frac": half_t_store / half_elapsed,
                    "half/infer_ms_per_frame": half_t_infer / processed * 1e3,
                    "match_id": match_id,
                    "half_id": half_id,
                },
                step=global_step[0],
            )

        return processed
    finally:
        cap.release()


def _shard_exists(store_root: Path, match_id: str, half_id: str) -> bool:
    """Return True if the cache shard for one match-half is already on disk."""
    from pcspot.features.cache import _shard_basename

    return (store_root / f"{_shard_basename(match_id, half_id)}.npz").exists()


def _write_run_manifest(
    out_root: Path,
    *,
    args: argparse.Namespace,
    metadata: VisualFeatureMetadata,
    records: list[dict],
) -> Path:
    """Append-update a run manifest documenting per-shard outcomes.

    The manifest sits at ``<out_root>/<backbone_name>/manifest.json`` and
    is overwritten in full on each invocation. It is intended for human
    audit (and for downstream tooling like the Codabench writer) rather
    than machine-rebuildable indexing.
    """
    target = out_root / metadata.backbone_name / "manifest.json"
    payload = {
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "metadata": metadata.to_dict(),
        "shards": records,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = PCBASConfig.load(args.config)
    out_root = (
        args.out_dir
        if args.out_dir is not None
        else cfg.output_dir / "visual_features"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    splits = parse_splits(args.splits)
    match_list = load_match_list(args.match_list) if args.match_list else None
    skip_existing = bool(args.skip_existing) and not bool(args.overwrite)

    pairs = _select_matches(
        cfg.output_dir,
        args.match_id,
        splits=splits,
        match_list=match_list,
    )

    print(f"Output dir : {out_root}")
    print(f"Splits     : {splits if splits is not None else '(legacy / unsplit)'}")
    print(f"Match list : {sorted(match_list) if match_list else '(none)'}")
    print(f"Matches    : {len(pairs)}")
    print(f"Skip exist : {skip_existing}")
    print(f"Dry run    : {bool(args.dry_run)}")

    cropper_cfg = CropperConfig(
        crop_size=int(args.crop_size),
        pad_factor=float(args.pad_factor),
        min_box_size=int(args.min_box_size),
    )

    # Defer the heavy imports until we know we have work to do; this
    # keeps --dry-run free of torch / DINOv2 side-effects.
    cropper: Optional[PaddedPlayerCropper] = None
    extractor = None
    store: Optional[VisualFeatureStore] = None
    metadata: Optional[VisualFeatureMetadata] = None

    if not args.dry_run:
        cropper = PaddedPlayerCropper(cropper_cfg)
        from pcspot.features.dinov2 import DinoV2Config, DinoV2Extractor

        dino_cfg = DinoV2Config(
            backbone_name=str(args.backbone),
            crop_size=int(args.crop_size),
            device=str(args.device),
            use_stub=bool(args.use_stub),
            batch_size=int(args.batch_size),
        )
        extractor = DinoV2Extractor(dino_cfg)
        metadata = VisualFeatureMetadata(
            backbone_name=dino_cfg.backbone_name + ("_stub" if args.use_stub else ""),
            feature_dim=extractor.feature_dim,
            crop_size=cropper_cfg.crop_size,
            pad_factor=cropper_cfg.pad_factor,
            fullhd_width=cropper_cfg.fullhd_width,
            fullhd_height=cropper_cfg.fullhd_height,
        )
        store = VisualFeatureStore(out_root, metadata=metadata)
    else:
        # Build a metadata stub so the dry-run manifest still records
        # the requested backbone / crop config.
        metadata = VisualFeatureMetadata(
            backbone_name=str(args.backbone) + ("_stub" if args.use_stub else ""),
            feature_dim=-1,
            crop_size=cropper_cfg.crop_size,
            pad_factor=cropper_cfg.pad_factor,
            fullhd_width=cropper_cfg.fullhd_width,
            fullhd_height=cropper_cfg.fullhd_height,
        )

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb and not args.dry_run
    wandb_run = None
    if use_wandb:
        wandb_run = _wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "backbone": args.backbone,
                "crop_size": args.crop_size,
                "pad_factor": args.pad_factor,
                "min_box_size": args.min_box_size,
                "batch_size": args.batch_size,
                "device": args.device,
                "use_stub": args.use_stub,
                "splits": splits,
                "n_matches": len(pairs),
                "out_dir": str(out_root),
            },
            tags=[args.backbone] + ([f"split:{s}" for s in splits] if splits else []),
        )
        print(f"W&B run: {wandb_run.url}")
    elif not _WANDB_AVAILABLE and not args.no_wandb:
        print("wandb not installed — skipping W&B logging. Install with: pip install wandb")

    backbone_root = out_root / metadata.backbone_name
    records: list[dict] = []
    total_frames = 0
    global_step: list[int] = [0]
    for split_label, match in pairs:
        if not match.video_path.exists():
            print(f"[{match.match_id}] video missing ({match.video_path}); skipping")
            records.append(
                {
                    "match_id": match.match_id,
                    "split": split_label,
                    "status": "missing-video",
                    "video_path": str(match.video_path),
                }
            )
            continue
        if args.dry_run:
            records.append(
                {
                    "match_id": match.match_id,
                    "split": split_label,
                    "status": "would-process",
                    "halves": list(match.halves),
                    "video_path": str(match.video_path),
                }
            )
            print(f"[dry-run] {match.match_id} halves={match.halves}")
            continue

        arrays = load_tactical_arrays(match)
        assert store is not None and cropper is not None and extractor is not None
        for half_key, arr in arrays.items():
            if skip_existing and _shard_exists(backbone_root, match.match_id, half_key):
                print(f"[{match.match_id}/{half_key}] cache shard exists; skipping")
                records.append(
                    {
                        "match_id": match.match_id,
                        "half_id": half_key,
                        "split": split_label,
                        "status": "skipped-existing",
                        "shard": str(backbone_root / f"{match.match_id}__{half_key}.npz"),
                    }
                )
                continue
            n = precompute_for_half(
                match_id=match.match_id,
                half_id=half_key,
                arr=arr,
                video_path=match.video_path,
                cropper=cropper,
                extractor=extractor,
                store=store,
                args=args,
                wandb_run=wandb_run,
                global_step=global_step,
            )
            print(f"[{match.match_id}/{half_key}] processed {n} frames")
            total_frames += n
            written = store.flush()
            records.append(
                {
                    "match_id": match.match_id,
                    "half_id": half_key,
                    "split": split_label,
                    "status": "written",
                    "frames": int(n),
                    "shards": [str(p) for p in written],
                }
            )

    manifest_path = _write_run_manifest(
        out_root, args=args, metadata=metadata, records=records
    )
    print(f"Manifest -> {manifest_path}")
    if args.dry_run:
        print("Dry run complete: no shards written.")
    else:
        print(
            f"Done. Wrote shards under {backbone_root} "
            f"({total_frames} new frames; {len(records)} records)"
        )

    if wandb_run is not None:
        wandb_run.summary["total_frames"] = total_frames
        wandb_run.summary["total_shards"] = len(
            [r for r in records if r.get("status") == "written"]
        )
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
