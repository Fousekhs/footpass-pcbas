"""Convert compressed visual-feature shards into a memmap-friendly layout.

The original cache writes one ``<match>__<half>.npz`` per match-half with
``np.savez_compressed``. Reading a single training window forces decompression
of the *entire* half (these shards are ~1 GB each), which makes random-access
(``--sampler mixed``/``uniform``) training crawl and starves the GPU.

This script rewrites each shard into two files the reader can use lazily:

    <match>__<half>.features.npy   # uncompressed float32 (N, F) -- memmap-able
    <match>__<half>.meta.npz       # frames / player_ids / visible (small)

``VisualFeatureCache`` prefers this layout automatically (it memmaps the
features and pages in only the rows a window touches), and falls back to the
legacy ``.npz`` when the converted files are absent.

Usage::

    python scripts/convert_visual_cache_mmap.py \\
        --cache-root data/pcbas_one_match/visual_features \\
        --backbone dinov2_vits14

Add ``--delete-original`` to remove the big compressed ``.npz`` once each shard
is converted (reclaims disk; only happens after the new files are written and
verified). Add ``--overwrite`` to redo shards that were already converted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _iter_legacy_shards(backbone_dir: Path):
    for path in sorted(backbone_dir.glob("*.npz")):
        # Skip the converted metadata sidecars (also end in .npz).
        if path.name.endswith(".meta.npz"):
            continue
        yield path


def convert_shard(npz_path: Path, *, overwrite: bool) -> tuple[bool, str]:
    base = npz_path.name[: -len(".npz")]
    feat_path = npz_path.with_name(f"{base}.features.npy")
    meta_path = npz_path.with_name(f"{base}.meta.npz")

    if feat_path.exists() and meta_path.exists() and not overwrite:
        return False, "already converted"

    with np.load(npz_path) as data:
        features = np.ascontiguousarray(data["features"][:], dtype=np.float32)
        frames = np.asarray(data["frames"][:], dtype=np.int64)
        player_ids = np.asarray(data["player_ids"][:], dtype=np.int64)
        visible = np.asarray(data["visible"][:], dtype=bool)

    if features.ndim != 2:
        return False, f"unexpected features shape {features.shape}; skipped"

    # Write to temp files then atomically rename, so an interrupted run never
    # leaves a half-written .features.npy that the reader would trust.
    feat_tmp = feat_path.with_suffix(feat_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    # Write through file handles: np.save/np.savez would otherwise append a
    # ".npy"/".npz" suffix to a path that does not already end in one, leaving
    # the temp file under a different name and breaking the atomic rename.
    with open(feat_tmp, "wb") as fh:
        np.save(fh, features)
    with open(meta_tmp, "wb") as fh:
        np.savez(fh, frames=frames, player_ids=player_ids, visible=visible)
    feat_tmp.replace(feat_path)
    meta_tmp.replace(meta_path)

    # Verify the memmap round-trips before reporting success.
    chk = np.load(feat_path, mmap_mode="r")
    if chk.shape != features.shape or chk.dtype != np.float32:
        return False, "verification failed; left original in place"

    msg = (
        f"{features.shape[0]}x{features.shape[1]} "
        f"({_human(npz_path.stat().st_size)} -> {_human(feat_path.stat().st_size)})"
    )
    return True, msg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Visual cache root (the parent of the backbone folder).")
    p.add_argument("--backbone", default="dinov2_vits14",
                   help="Backbone subfolder name under --cache-root.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reconvert shards even if converted files already exist.")
    p.add_argument("--delete-original", action="store_true",
                   help="Delete the legacy compressed .npz after a verified conversion.")
    args = p.parse_args(argv)

    backbone_dir = args.cache_root / args.backbone
    if not backbone_dir.is_dir():
        print(f"error: not a directory: {backbone_dir}", file=sys.stderr)
        return 2

    shards = list(_iter_legacy_shards(backbone_dir))
    if not shards:
        print(f"No legacy .npz shards found in {backbone_dir}.")
        return 0

    print(f"Converting {len(shards)} shard(s) in {backbone_dir} ...")
    converted = skipped = failed = 0
    for i, npz_path in enumerate(shards, 1):
        try:
            ok, msg = convert_shard(npz_path, overwrite=args.overwrite)
        except Exception as exc:  # pragma: no cover - defensive
            failed += 1
            print(f"  [{i}/{len(shards)}] {npz_path.name}: ERROR {exc}", file=sys.stderr)
            continue
        if ok:
            converted += 1
            print(f"  [{i}/{len(shards)}] {npz_path.name}: {msg}")
            if args.delete_original:
                npz_path.unlink()
        else:
            skipped += 1
            print(f"  [{i}/{len(shards)}] {npz_path.name}: {msg}")

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
