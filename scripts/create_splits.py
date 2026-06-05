"""Generate a splits.json manifest from the extracted PCBAS dataset directories.

The script scans the ``extracted/`` tree produced by ``mirror_pcbas_one_match.py``
or ``mirror_pcbas_full.py`` and writes a JSON file consumable by
:class:`pcspot.data.splits.SplitManifest`.

Modes
-----
``--mode official`` (default)
    Reads each ``tactical_data_<SPLIT>/`` directory and assigns every
    match-half found there to the corresponding split.  This reproduces
    the dataset's own TRAIN / VAL / CHALLENGE partition.

``--mode custom``
    You specify which match-ids belong to which split via ``--train``,
    ``--val``, and ``--challenge``.  The script discovers the halves for
    each provided match-id by searching all extracted HDF5 files.

Usage::

    # Write splits.json using the official dataset split
    python scripts/create_splits.py --config config.toml

    # Dry-run (print manifest, don't write)
    python scripts/create_splits.py --config config.toml --dry-run

    # Custom split: override which matches go where
    python scripts/create_splits.py --config config.toml --mode custom \\
        --train game_1,game_2,game_3 \\
        --val   game_4 \\
        --challenge game_5,game_6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcbas_data import PCBASConfig  # noqa: E402

VALID_SPLITS = ("TRAIN", "VAL", "CHALLENGE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for splits.json. Default: <output_dir>/splits.json",
    )
    p.add_argument(
        "--mode",
        choices=("official", "custom"),
        default="official",
        help=(
            "official (default): use the dataset's own TRAIN/VAL/CHALLENGE "
            "directory partition. custom: specify match-ids via --train/--val/--challenge."
        ),
    )
    p.add_argument(
        "--train",
        type=str,
        default=None,
        help="Comma-separated match-ids for the train split (--mode custom only).",
    )
    p.add_argument(
        "--val",
        type=str,
        default=None,
        help="Comma-separated match-ids for the val split (--mode custom only).",
    )
    p.add_argument(
        "--challenge",
        type=str,
        default=None,
        help="Comma-separated match-ids for the challenge split (--mode custom only).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest to stdout without writing the file.",
    )
    return p.parse_args(argv)


def _find_tactical_dirs_for_split(extracted: Path, split_upper: str) -> list[Path]:
    """Return all tactical_data_<SPLIT>* directories for the given split."""
    target = f"_{split_upper.lower()}"
    dirs: list[Path] = []
    for child in sorted(extracted.iterdir()):
        if child.is_dir() and child.name.lower().startswith("tactical_data_") and target in child.name.lower():
            dirs.append(child)
    return dirs


def _parse_h5_keys(h5_path: Path) -> dict[str, list[str]]:
    """Return {match_id: [half_key, ...]} from an HDF5 tactical file."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(f.keys())

    halves_by_match: dict[str, list[str]] = {}
    for k in keys:
        parts = k.rsplit("_", 1)
        if len(parts) == 2 and parts[1].startswith("H") and parts[1][1:].isdigit():
            halves_by_match.setdefault(parts[0], []).append(k)
        else:
            halves_by_match.setdefault(k, []).append(k)
    return halves_by_match


def discover_halves_for_split(
    extracted: Path, split_upper: str
) -> list[tuple[str, str]]:
    """Return sorted [(match_id, half_key)] for one split directory."""
    tac_dirs = _find_tactical_dirs_for_split(extracted, split_upper)
    if not tac_dirs:
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for tac_dir in tac_dirs:
        h5_files = sorted(tac_dir.glob("*.h5"))
        if not h5_files:
            continue
        halves_by_match = _parse_h5_keys(h5_files[0])
        for match_id, half_keys in halves_by_match.items():
            for hk in sorted(half_keys):
                pair = (match_id, hk)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return sorted(pairs)


def discover_all_halves(extracted: Path) -> dict[str, list[str]]:
    """Return {match_id: [half_key, ...]} across every extracted HDF5."""
    all_halves: dict[str, list[str]] = {}
    for child in sorted(extracted.iterdir()):
        if not child.is_dir() or not child.name.lower().startswith("tactical_data_"):
            continue
        for h5_file in sorted(child.glob("*.h5")):
            for match_id, half_keys in _parse_h5_keys(h5_file).items():
                existing = all_halves.setdefault(match_id, [])
                for hk in half_keys:
                    if hk not in existing:
                        existing.append(hk)
    return all_halves


def _parse_id_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def build_official_manifest(
    extracted: Path,
) -> dict[str, list[dict[str, str]]]:
    """Build manifest from official TRAIN/VAL/CHALLENGE directory partition."""
    manifest: dict[str, list[dict[str, str]]] = {}
    for split_upper in VALID_SPLITS:
        pairs = discover_halves_for_split(extracted, split_upper)
        if not pairs:
            print(f"[{split_upper}] no tactical directories found; skipping")
            continue
        split_lower = split_upper.lower()
        manifest[split_lower] = [
            {"match_id": mid, "half_id": hk} for mid, hk in pairs
        ]
        print(f"[{split_upper}] {len(pairs)} match-halves discovered")
    return manifest


def build_custom_manifest(
    extracted: Path,
    train_ids: list[str],
    val_ids: list[str],
    challenge_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Build manifest from user-specified match-id lists."""
    all_halves = discover_all_halves(extracted)

    def _resolve(ids: list[str], split_name: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for mid in ids:
            halves = all_halves.get(mid)
            if halves is None:
                raise SystemExit(
                    f"Match {mid!r} not found in any extracted HDF5 file "
                    f"(requested for --{split_name})."
                )
            for hk in sorted(halves):
                entries.append({"match_id": mid, "half_id": hk})
        return entries

    manifest: dict[str, list[dict[str, str]]] = {}
    if train_ids:
        manifest["train"] = _resolve(train_ids, "train")
    if val_ids:
        manifest["val"] = _resolve(val_ids, "val")
    if challenge_ids:
        manifest["challenge"] = _resolve(challenge_ids, "challenge")
    return manifest


def validate_no_duplicates(manifest: dict[str, list[dict[str, str]]]) -> None:
    """Abort if any (match_id, half_id) appears in more than one split."""
    seen: dict[tuple[str, str], str] = {}
    for split_name, entries in manifest.items():
        for entry in entries:
            key = (entry["match_id"], entry["half_id"])
            if key in seen:
                raise SystemExit(
                    f"Duplicate half detected: {key} appears in both "
                    f"'{seen[key]}' and '{split_name}'. "
                    "Fix your --train/--val/--challenge lists."
                )
            seen[key] = split_name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = PCBASConfig.load(args.config)
    extracted = cfg.output_dir / "extracted"

    if not extracted.exists():
        raise SystemExit(
            f"Extracted directory not found: {extracted}\n"
            "Run mirror_pcbas_one_match.py or mirror_pcbas_full.py first."
        )

    out_path = args.out if args.out is not None else cfg.output_dir / "splits.json"

    if args.mode == "official":
        manifest = build_official_manifest(extracted)
    else:
        train_ids = _parse_id_list(args.train)
        val_ids = _parse_id_list(args.val)
        challenge_ids = _parse_id_list(args.challenge)
        if not (train_ids or val_ids or challenge_ids):
            raise SystemExit(
                "Custom mode requires at least one of --train, --val, --challenge."
            )
        manifest = build_custom_manifest(extracted, train_ids, val_ids, challenge_ids)

    if not manifest:
        raise SystemExit(
            "No match-halves discovered. Check that the dataset was extracted "
            "correctly under: " + str(extracted)
        )

    validate_no_duplicates(manifest)

    total = sum(len(v) for v in manifest.values())
    payload = json.dumps(manifest, indent=2, sort_keys=True)

    if args.dry_run:
        print(payload)
        print(f"\n[dry-run] {total} match-halves across {len(manifest)} splits — not written.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(f"Written: {out_path}  ({total} match-halves, {len(manifest)} splits)")
    for split_name, entries in sorted(manifest.items()):
        print(f"  {split_name}: {len(entries)} halves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
