"""Mirror the full PCBAS / FOOTPASS dataset at the best available quality.

This is the production / training counterpart to
``mirror_pcbas_one_match.py`` (which is a smoke-test helper for the
smallest split). The full mirror script:

- Defaults to **all** splits (TRAIN, VAL, CHALLENGE).
- Defaults to **fullHD** broadcast video archives.
- Supports per-split download manifests and a top-level manifest, so a
  long download can be resumed and audited.
- Skips already-downloaded archives by default (use ``--overwrite`` to
  force re-download), and skips already-extracted directories unless
  ``--re-extract`` is passed.
- Optionally mirrors the 352x640 low-res videos as well (debug only)
  via ``--include-lowres``.
- Reads gated credentials (Hugging Face token, SoccerNet NDA password)
  from the same ``config.toml`` schema used by the other scripts.
- Never prints or persists token/password values.

Typical usage on a Vast.ai GPU box::

    python scripts/mirror_pcbas_full.py --config config.toml --dry-run
    python scripts/mirror_pcbas_full.py --config config.toml

For a partial mirror (e.g. only TRAIN + VAL, fullHD) pass
``--splits TRAIN,VAL``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Reuse the well-tested helpers from the one-match mirror so file
# selection, repo listing, and extraction stay consistent across the
# two scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mirror_pcbas_one_match import (  # noqa: E402
    ARCHIVE_EXTS,
    Config,
    FileEntry,
    VALID_SPLITS,
    list_repo_files,
    select_files_for_split,
    select_global_files,
    try_extract,
)


VALID_RESOLUTIONS = ("fullHD", "352x640")


@dataclass
class SplitMirrorResult:
    split: str
    resolution: str
    files: list[dict] = field(default_factory=list)
    extractions: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "resolution": self.resolution,
            "files": self.files,
            "extractions": self.extractions,
            "skipped": self.skipped,
            "errors": self.errors,
        }


@dataclass
class FullMirrorResult:
    repo: str
    splits: list[str]
    resolutions: list[str]
    output_dir: str
    started_at: str
    finished_at: str | None = None
    per_split: list[SplitMirrorResult] = field(default_factory=list)
    global_files: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "repo": self.repo,
            "splits": self.splits,
            "resolutions": self.resolutions,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "global_files": self.global_files,
            "per_split": [s.as_dict() for s in self.per_split],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--splits",
        type=str,
        default="TRAIN,VAL,CHALLENGE",
        help=(
            "Comma-separated list of splits to mirror. Default: all three "
            "(TRAIN,VAL,CHALLENGE). Subset to TRAIN, VAL, or CHALLENGE if "
            "you only want one."
        ),
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="fullHD",
        choices=VALID_RESOLUTIONS,
        help="Primary video resolution to mirror. Default: fullHD (best quality).",
    )
    p.add_argument(
        "--include-lowres",
        action="store_true",
        help=(
            "Additionally mirror the 352x640 archives alongside fullHD. "
            "Useful for quick local rendering / debugging only; doubles "
            "the download size."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List archives that would be downloaded without writing files.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-download archives even if they already exist locally. "
            "Default: skip files whose local size matches the remote size."
        ),
    )
    p.add_argument(
        "--re-extract",
        action="store_true",
        help=(
            "Re-run extraction even if the target directory already exists. "
            "Default: skip extraction when the target dir is non-empty."
        ),
    )
    p.add_argument(
        "--include-global",
        dest="include_global",
        action="store_true",
        default=True,
        help="Also mirror small global metadata files (default).",
    )
    p.add_argument(
        "--no-global",
        dest="include_global",
        action="store_false",
        help="Skip global metadata files.",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help=(
            "Soft cap on total bytes to download in this invocation. The "
            "script will stop after the next archive would exceed the cap. "
            "Useful to spread a full download across short Vast.ai sessions."
        ),
    )
    p.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        default=True,
        help="Only download archives, do not attempt extraction.",
    )
    return p.parse_args(argv)


def parse_splits(spec: str) -> list[str]:
    """Validate and normalize the --splits CSV argument."""
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


def resolutions_for(args: argparse.Namespace) -> list[str]:
    out = [args.resolution]
    if args.include_lowres and "352x640" not in out:
        out.append("352x640")
    return out


def _should_skip_existing(local_path: Path, remote_size: int | None) -> bool:
    """Skip if the local file already exists and (when known) matches size."""
    if not local_path.exists():
        return False
    if remote_size is None:
        # No remote size advertised; be conservative and re-download.
        return False
    try:
        return local_path.stat().st_size == int(remote_size)
    except OSError:
        return False


def _download_one(
    repo: str,
    token: str,
    entry: FileEntry,
    dest: Path,
) -> Path:
    """Wrapper around hf_hub_download for a single entry."""
    from huggingface_hub import hf_hub_download

    kwargs: dict = {
        "repo_id": repo,
        "repo_type": "dataset",
        "filename": entry.repo_path,
        "local_dir": str(dest),
    }
    if token:
        kwargs["token"] = token
    local_path = hf_hub_download(**kwargs)
    return Path(local_path)


def _extraction_target_nonempty(target: Path) -> bool:
    if not target.exists():
        return False
    try:
        return any(target.iterdir())
    except OSError:
        return False


def select_split_files(
    entries: Iterable[FileEntry], split: str, resolutions: list[str]
) -> list[FileEntry]:
    """Union of file selections across the requested resolutions."""
    seen: dict[str, FileEntry] = {}
    for res in resolutions:
        for entry in select_files_for_split(entries, split, res):
            seen.setdefault(entry.repo_path, entry)
    # Deterministic ordering to make manifests reproducible.
    return sorted(seen.values(), key=lambda e: e.repo_path)


def mirror_split(
    *,
    cfg: Config,
    args: argparse.Namespace,
    entries: list[FileEntry],
    split: str,
    raw_root: Path,
    extract_root: Path,
    bytes_remaining: int | None,
) -> tuple[SplitMirrorResult, int | None]:
    """Mirror one split. Returns (result, updated bytes_remaining)."""
    resolutions = resolutions_for(args)
    split_files = select_split_files(entries, split, resolutions)
    result = SplitMirrorResult(split=split, resolution=",".join(resolutions))

    print(f"\n=== Split {split} ({len(split_files)} archive(s)) ===")
    if not split_files:
        print(f"  No archives matched for resolutions={resolutions}.")
        return result, bytes_remaining

    for entry in split_files:
        local_path = raw_root / entry.repo_path
        size = entry.size
        size_str = f"{(size or 0) / 1e6:.1f} MB" if size else "?"
        if not args.overwrite and _should_skip_existing(local_path, size):
            print(f"  [skip-existing] {entry.repo_path} ({size_str})")
            result.skipped.append(
                {
                    "repo_path": entry.repo_path,
                    "local_path": str(local_path),
                    "reason": "exists",
                }
            )
        elif args.dry_run:
            print(f"  [dry-run]        {entry.repo_path} ({size_str})")
            result.files.append(
                {
                    "repo_path": entry.repo_path,
                    "local_path": str(local_path),
                    "size": size,
                    "downloaded": False,
                }
            )
            continue
        else:
            if (
                bytes_remaining is not None
                and size is not None
                and bytes_remaining < size
            ):
                print(
                    f"  [budget-stop]   {entry.repo_path} ({size_str}) "
                    f"would exceed --max-bytes; stopping split."
                )
                result.errors.append(
                    {
                        "repo_path": entry.repo_path,
                        "reason": "max-bytes-budget",
                    }
                )
                return result, bytes_remaining
            print(f"  [download]      {entry.repo_path} ({size_str})")
            try:
                downloaded = _download_one(
                    cfg.huggingface_repo, cfg.huggingface_token, entry, raw_root
                )
            except Exception as exc:
                msg = f"download failed: {exc!r}"
                print(f"    ! {msg}")
                result.errors.append(
                    {"repo_path": entry.repo_path, "reason": msg}
                )
                continue
            local_path = downloaded
            actual_size = local_path.stat().st_size if local_path.exists() else None
            if bytes_remaining is not None and actual_size is not None:
                bytes_remaining = max(0, bytes_remaining - int(actual_size))
            result.files.append(
                {
                    "repo_path": entry.repo_path,
                    "local_path": str(local_path),
                    "size": actual_size,
                    "downloaded": True,
                }
            )

        if not args.extract:
            continue
        if local_path.suffix.lower() not in ARCHIVE_EXTS:
            continue
        target = extract_root / local_path.stem
        if not args.re_extract and _extraction_target_nonempty(target):
            print(f"    [skip-extract] {target} already non-empty")
            result.extractions.append(
                {
                    "archive": str(local_path),
                    "target": str(target),
                    "ok": True,
                    "skipped": True,
                    "message": "already extracted",
                }
            )
            continue
        ok, msg = try_extract(local_path, cfg.soccernet_password, extract_root)
        entry_record = {
            "archive": str(local_path),
            "target": str(target),
            "ok": ok,
            "skipped": False,
            "message": msg,
        }
        if ok:
            print(f"    [extracted]    {msg}")
            result.extractions.append(entry_record)
        else:
            print(f"    [extract-fail] {msg}")
            result.skipped.append(entry_record)

    return result, bytes_remaining


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _warn_credentials(cfg: Config) -> None:
    if not cfg.huggingface_token:
        print(
            "warning: huggingface_token is empty in config.toml. Listing "
            "a gated dataset will likely fail. (Token is never logged.)",
            file=sys.stderr,
        )
    if not cfg.soccernet_password:
        print(
            "warning: soccernet_password is empty in config.toml. "
            "Password-protected archives will not be extracted.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    splits = parse_splits(args.splits)
    resolutions = resolutions_for(args)

    print(f"Repository  : {cfg.huggingface_repo}")
    print(f"Output dir  : {cfg.output_dir}")
    print(f"Splits      : {splits}")
    print(f"Resolutions : {resolutions}")
    print(f"Dry run     : {args.dry_run}")
    print(f"Overwrite   : {args.overwrite}")
    print(f"Re-extract  : {args.re_extract}")
    if args.max_bytes is not None:
        print(f"Max bytes   : {args.max_bytes} ({args.max_bytes / 1e9:.2f} GB)")
    _warn_credentials(cfg)

    entries = list_repo_files(cfg.huggingface_repo, cfg.huggingface_token)
    print(f"Repo listing: {len(entries)} files")

    raw_root = cfg.output_dir / "raw"
    extract_root = cfg.output_dir / "extracted"
    raw_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    result = FullMirrorResult(
        repo=cfg.huggingface_repo,
        splits=splits,
        resolutions=resolutions,
        output_dir=str(cfg.output_dir),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    if args.include_global:
        global_entries = select_global_files(entries)
        for entry in global_entries:
            local_path = raw_root / entry.repo_path
            if not args.overwrite and _should_skip_existing(local_path, entry.size):
                result.global_files.append(
                    {
                        "repo_path": entry.repo_path,
                        "local_path": str(local_path),
                        "downloaded": False,
                        "reason": "exists",
                    }
                )
                continue
            if args.dry_run:
                print(f"[dry-run global] {entry.repo_path}")
                result.global_files.append(
                    {
                        "repo_path": entry.repo_path,
                        "local_path": str(local_path),
                        "downloaded": False,
                    }
                )
                continue
            try:
                local = _download_one(
                    cfg.huggingface_repo, cfg.huggingface_token, entry, raw_root
                )
                result.global_files.append(
                    {
                        "repo_path": entry.repo_path,
                        "local_path": str(local),
                        "downloaded": True,
                        "size": local.stat().st_size if local.exists() else None,
                    }
                )
            except Exception as exc:
                print(f"warning: global file {entry.repo_path} failed: {exc!r}",
                      file=sys.stderr)
                result.global_files.append(
                    {
                        "repo_path": entry.repo_path,
                        "local_path": str(local_path),
                        "downloaded": False,
                        "reason": f"error: {exc!r}",
                    }
                )

    bytes_remaining = int(args.max_bytes) if args.max_bytes is not None else None
    for split in splits:
        split_result, bytes_remaining = mirror_split(
            cfg=cfg,
            args=args,
            entries=entries,
            split=split,
            raw_root=raw_root,
            extract_root=extract_root,
            bytes_remaining=bytes_remaining,
        )
        result.per_split.append(split_result)
        _write_manifest(
            cfg.output_dir / f"manifest_{split}.json", split_result.as_dict()
        )
        if bytes_remaining is not None and bytes_remaining <= 0:
            print(f"--max-bytes budget exhausted; stopping after split {split}.")
            break

    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(cfg.output_dir / "manifest_full.json", result.as_dict())
    print(f"\nWrote top-level manifest: {cfg.output_dir / 'manifest_full.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
