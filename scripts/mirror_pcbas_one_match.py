"""Mirror one match group from the FOOTPASS / SoccerNet PCBAS dataset.

The Hugging Face repository ``SoccerNet/SN-PCBAS-2026`` does not expose
single matches as discrete files. Instead it ships split-level archives
(``*_TRAIN*.zip``, ``*_VAL.zip``, ``*_CHALLENGE.zip``) that each contain
multiple matches. This script therefore mirrors one *split* (default
``VAL``, the smallest one with 3 matches), together with the matching
tactical-data archive and the dataset's format specification. The
SoccerNet NDA password from ``config.toml`` is used to attempt
extraction of password-protected archives.

Usage:
    python scripts/mirror_pcbas_one_match.py --config config.toml --dry-run
    python scripts/mirror_pcbas_one_match.py --config config.toml --split VAL
    python scripts/mirror_pcbas_one_match.py --config config.toml --split VAL --resolution 352x640
"""

from __future__ import annotations

import argparse
import json
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ARCHIVE_EXTS = (".zip", ".7z")

GLOBAL_FILENAMES = (
    "readme.md",
    "readme",
    "license",
    "license.txt",
    "tactical_data_format.txt",
)

VALID_SPLITS = ("TRAIN", "VAL", "CHALLENGE")

SPLIT_RE = re.compile(r"_(TRAIN|VAL|CHALLENGE)(?:_\d+)?\.zip$", re.IGNORECASE)
RESOLUTION_RE = re.compile(r"videos_([^_]+)_(?:TRAIN|VAL|CHALLENGE)", re.IGNORECASE)


@dataclass
class Config:
    huggingface_repo: str
    huggingface_token: str
    soccernet_password: str
    output_dir: Path

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise SystemExit(f"Config file not found: {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)
        section = data.get("pcbas")
        if not isinstance(section, dict):
            raise SystemExit("config.toml must contain a [pcbas] table")
        try:
            repo = section["huggingface_repo"]
        except KeyError:
            raise SystemExit("Missing required key: pcbas.huggingface_repo") from None
        return cls(
            huggingface_repo=str(repo),
            huggingface_token=str(section.get("huggingface_token", "") or ""),
            soccernet_password=str(section.get("soccernet_password", "") or ""),
            output_dir=Path(section.get("output_dir", "data/pcbas_one_match")),
        )


@dataclass
class FileEntry:
    repo_path: str
    size: Optional[int]


@dataclass
class MirrorResult:
    repo: str
    split: str
    resolution: str
    downloaded: list[dict] = field(default_factory=list)
    extracted: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "repo": self.repo,
            "split": self.split,
            "resolution": self.resolution,
            "downloaded": self.downloaded,
            "extracted": self.extracted,
            "skipped": self.skipped,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--split",
        type=str,
        default="VAL",
        choices=VALID_SPLITS,
        help="Which split's archives to mirror. The HF repo bundles whole "
        "splits, not single matches. Default is VAL (smallest, 3 matches).",
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="352x640",
        help="Video resolution variant to include (e.g. 352x640, fullHD). "
        "Default 352x640 keeps the download small.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing files.",
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
    return p.parse_args()


def list_repo_files(repo: str, token: str) -> list[FileEntry]:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install it with `pip install huggingface_hub`."
        ) from e

    api = HfApi()
    kwargs: dict = {"repo_id": repo, "repo_type": "dataset"}
    if token:
        kwargs["token"] = token

    try:
        items = list(api.list_repo_tree(recursive=True, **kwargs))
    except Exception as e:
        raise SystemExit(
            "Failed to list the Hugging Face dataset.\n"
            "  - Make sure you accepted the dataset terms on the repo page\n"
            f"    https://huggingface.co/datasets/{repo}\n"
            "  - Make sure huggingface_token in config.toml is valid.\n"
            f"  - Underlying error: {e}"
        ) from None

    entries: list[FileEntry] = []
    for item in items:
        item_type = getattr(item, "type", None) or item.__class__.__name__.lower()
        path_attr = getattr(item, "path", None) or getattr(item, "rfilename", None)
        if not path_attr:
            continue
        if "directory" in str(item_type).lower() or "folder" in str(item_type).lower():
            continue
        size = getattr(item, "size", None)
        entries.append(FileEntry(repo_path=str(path_attr), size=size))
    return entries


def detect_split_groups(entries: Iterable[FileEntry]) -> dict[str, list[FileEntry]]:
    """Group archive files by split (TRAIN/VAL/CHALLENGE)."""
    groups: dict[str, list[FileEntry]] = {}
    for entry in entries:
        m = SPLIT_RE.search(entry.repo_path)
        if m:
            groups.setdefault(m.group(1).upper(), []).append(entry)
    return groups


def select_files_for_split(
    entries: Iterable[FileEntry], split: str, resolution: str
) -> list[FileEntry]:
    """Select tactical data + chosen-resolution video archives for one split."""
    split_u = split.upper()
    res_l = resolution.lower()
    selected: list[FileEntry] = []
    for entry in entries:
        m = SPLIT_RE.search(entry.repo_path)
        if not m or m.group(1).upper() != split_u:
            continue
        name = entry.repo_path.lower()
        if name.startswith("tactical_data_"):
            selected.append(entry)
            continue
        rm = RESOLUTION_RE.search(entry.repo_path)
        if rm and rm.group(1).lower() == res_l:
            selected.append(entry)
    return selected


def select_global_files(entries: Iterable[FileEntry]) -> list[FileEntry]:
    selected: list[FileEntry] = []
    for entry in entries:
        parts = entry.repo_path.split("/")
        if len(parts) != 1:
            continue
        if parts[0].lower() in GLOBAL_FILENAMES:
            selected.append(entry)
    return selected


def download_files(
    repo: str, token: str, files: Iterable[FileEntry], dest: Path
) -> list[Path]:
    from huggingface_hub import hf_hub_download

    downloaded: list[Path] = []
    for f in files:
        kwargs: dict = {
            "repo_id": repo,
            "repo_type": "dataset",
            "filename": f.repo_path,
            "local_dir": str(dest),
        }
        if token:
            kwargs["token"] = token
        local_path = hf_hub_download(**kwargs)
        downloaded.append(Path(local_path))
    return downloaded


def try_extract(archive: Path, password: str, dest_dir: Path) -> tuple[bool, str]:
    ext = archive.suffix.lower()
    target = dest_dir / archive.stem
    target.mkdir(parents=True, exist_ok=True)
    try:
        if ext == ".zip":
            import pyzipper

            with pyzipper.AESZipFile(str(archive)) as zf:
                if password:
                    zf.setpassword(password.encode("utf-8"))
                zf.extractall(path=str(target))
            return True, f"extracted to {target}"
        if ext == ".7z":
            import py7zr

            with py7zr.SevenZipFile(
                str(archive), mode="r", password=password or None
            ) as zf:
                zf.extractall(path=str(target))
            return True, f"extracted to {target}"
        return False, f"unsupported archive type: {ext}"
    except Exception as e:
        return False, f"extract failed: {e!r}"


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config)

    print(f"Repository : {cfg.huggingface_repo}")
    print(f"Output dir : {cfg.output_dir}")
    print(f"Dry run    : {args.dry_run}")
    if not cfg.huggingface_token:
        print(
            "Note: huggingface_token is empty. Listing a gated dataset will "
            "likely fail until you set it in config.toml."
        )
    if not cfg.soccernet_password:
        print(
            "Note: soccernet_password is empty. Password-protected archives "
            "will be left unextracted and recorded in the manifest."
        )

    entries = list_repo_files(cfg.huggingface_repo, cfg.huggingface_token)
    print(f"Found {len(entries)} files in the repository.")

    groups = detect_split_groups(entries)
    if not groups:
        print(
            "Could not detect any split-level archives (looking for "
            "*_TRAIN/_VAL/_CHALLENGE.zip). First entries:"
        )
        for e in entries[:20]:
            print(f"  - {e.repo_path}")
        return 2

    if args.split not in groups:
        print(
            f"Warning: split '{args.split}' has no matching archives. "
            f"Detected splits: {sorted(groups)}"
        )

    split = args.split
    print(f"Split chosen   : {split} (matches go: VAL=3, CHALLENGE=3, TRAIN=48)")
    print(f"Resolution     : {args.resolution}")

    match_files = select_files_for_split(entries, split, args.resolution)
    global_files = select_global_files(entries) if args.include_global else []

    if not match_files:
        print(
            f"No archives matched split={split} resolution={args.resolution}. "
            f"Files available for this split:"
        )
        for f in groups.get(split, []):
            print(f"  - {f.repo_path}")
        return 2

    print(f"Match-group archives    : {len(match_files)}")
    print(f"Global / metadata files : {len(global_files)}")

    total_bytes = sum((f.size or 0) for f in match_files + global_files)
    print(
        f"Approx total to fetch   : {total_bytes / 1e6:.2f} MB "
        "(sizes may be missing for some files)"
    )

    result = MirrorResult(
        repo=cfg.huggingface_repo, split=split, resolution=args.resolution
    )

    if args.dry_run:
        print()
        print("Archives that would be downloaded:")
        for f in match_files:
            print(f"  archive {f.repo_path}  ({f.size} bytes)")
        print("Global files:")
        for f in global_files:
            print(f"  global  {f.repo_path}  ({f.size} bytes)")
        return 0

    raw_dir = cfg.output_dir / "raw"
    extract_dir = cfg.output_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    all_files = match_files + global_files
    try:
        downloaded_paths = download_files(
            cfg.huggingface_repo, cfg.huggingface_token, all_files, raw_dir
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return 3

    for f, p in zip(all_files, downloaded_paths):
        result.downloaded.append(
            {
                "repo_path": f.repo_path,
                "local_path": str(p),
                "size": p.stat().st_size if p.exists() else None,
            }
        )

    for p in downloaded_paths:
        if p.suffix.lower() in ARCHIVE_EXTS:
            ok, msg = try_extract(p, cfg.soccernet_password, extract_dir)
            entry = {"archive": str(p), "ok": ok, "message": msg}
            if ok:
                result.extracted.append(entry)
            else:
                result.skipped.append(entry)

    manifest_path = cfg.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result.as_dict(), indent=2))
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
