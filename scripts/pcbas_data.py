"""Loaders for extracted PCBAS / FOOTPASS tactical data and videos.

The extracted layout produced by ``mirror_pcbas_one_match.py`` is::

    data/pcbas_one_match/
      extracted/
        tactical_data_<SPLIT>/<split>_tactical_data.h5
        videos_<RES>_<SPLIT>/<match>.mp4
      raw/tactical_data_format.txt

The HDF5 file contains one dataset per match half, e.g. ``game_18_H1``,
shaped ``(N, 14)`` with columns::

    frame, player_id, left_to_right, shirt_number, role_id,
    x, y, speed_x, speed_y,
    roi_x, roi_y, roi_width, roi_height,
    class

Pitch coordinates are normalized in roughly ``[0, 1]``. ROI coordinates
are pixel values in the original fullHD (1920x1080) broadcast frame.
``class == 0`` means no event; non-zero values are the per-frame action
labels.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

COLUMNS = (
    "frame",
    "player_id",
    "left_to_right",
    "shirt_number",
    "role_id",
    "x",
    "y",
    "speed_x",
    "speed_y",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "class",
)
COL = {name: i for i, name in enumerate(COLUMNS)}

ROLE_NAMES = {
    1: "GK",
    2: "LB",
    3: "LCB",
    4: "MCB",
    5: "RCB",
    6: "LM",
    7: "RM",
    8: "DM",
    9: "AM",
    10: "LW",
    11: "RW",
    12: "CF",
    13: "RB",
}

CLASS_NAMES = {
    0: "",
    1: "Drive",
    2: "Pass",
    3: "Cross",
    4: "Shot",
    5: "Header",
    6: "Throw-in",
    7: "Tackle",
    8: "Block",
    9: "High Pass",
    10: "Out",
    11: "Free Kick",
    12: "Goal",
}

FULLHD_W = 1920
FULLHD_H = 1080


@dataclass
class PCBASConfig:
    huggingface_repo: str
    output_dir: Path

    @classmethod
    def load(cls, path: Path) -> "PCBASConfig":
        if not path.exists():
            raise SystemExit(f"Config file not found: {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)
        section = data.get("pcbas")
        if not isinstance(section, dict):
            raise SystemExit("config.toml must contain a [pcbas] table")
        return cls(
            huggingface_repo=str(section.get("huggingface_repo", "")),
            output_dir=Path(section.get("output_dir", "data/pcbas_one_match")),
        )


@dataclass
class MatchAssets:
    match_id: str
    video_path: Path
    halves: list[str]  # dataset keys inside the HDF5 file
    tactical_h5: Path

    @property
    def half_count(self) -> int:
        return len(self.halves)


def find_split_dirs(output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Return (tactical_dir, video_dir) found under extracted/."""
    extracted = output_dir / "extracted"
    if not extracted.exists():
        return (None, None)
    tactical_dir = None
    video_dir = None
    for child in sorted(extracted.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.startswith("tactical_data_") and tactical_dir is None:
            tactical_dir = child
        elif name.startswith("videos_") and video_dir is None:
            video_dir = child
    return tactical_dir, video_dir


def list_matches(output_dir: Path) -> list[MatchAssets]:
    """Discover available matches from extracted tactical+video folders."""
    tactical_dir, video_dir = find_split_dirs(output_dir)
    if tactical_dir is None or video_dir is None:
        return []

    h5_files = sorted(tactical_dir.glob("*.h5"))
    if not h5_files:
        return []

    import h5py

    h5_path = h5_files[0]
    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(f.keys())

    halves_by_match: dict[str, list[str]] = {}
    for k in keys:
        parts = k.rsplit("_", 1)
        if len(parts) == 2 and parts[1].startswith("H") and parts[1][1:].isdigit():
            halves_by_match.setdefault(parts[0], []).append(k)
        else:
            halves_by_match.setdefault(k, []).append(k)

    matches: list[MatchAssets] = []
    for match_id, halves in halves_by_match.items():
        video_path = video_dir / f"{match_id}.mp4"
        matches.append(
            MatchAssets(
                match_id=match_id,
                video_path=video_path,
                halves=sorted(halves),
                tactical_h5=h5_path,
            )
        )
    return matches


def load_tactical_arrays(match: MatchAssets) -> dict[str, np.ndarray]:
    """Load per-half tactical numpy arrays for one match."""
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(str(match.tactical_h5), "r") as f:
        for half_key in match.halves:
            out[half_key] = f[half_key][:]
    return out


def concat_match_array(match: MatchAssets) -> np.ndarray:
    """Return the concatenation of all halves for a match."""
    arrays = list(load_tactical_arrays(match).values())
    if not arrays:
        return np.empty((0, len(COLUMNS)), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def index_by_frame(arr: np.ndarray) -> dict[int, np.ndarray]:
    """Group rows by integer frame index."""
    if arr.size == 0:
        return {}
    frames = arr[:, COL["frame"]].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    arr_sorted = arr[order]
    frames_sorted = frames[order]
    out: dict[int, np.ndarray] = {}
    unique, starts = np.unique(frames_sorted, return_index=True)
    for i, f_idx in enumerate(unique):
        start = starts[i]
        stop = starts[i + 1] if i + 1 < len(starts) else len(arr_sorted)
        out[int(f_idx)] = arr_sorted[start:stop]
    return out


def select_events(arr: np.ndarray) -> np.ndarray:
    """Return only rows where class != 0 (i.e. annotated events)."""
    if arr.size == 0:
        return arr
    return arr[arr[:, COL["class"]] != 0]


def role_label(role_id: float) -> str:
    if np.isnan(role_id):
        return "?"
    return ROLE_NAMES.get(int(role_id), str(int(role_id)))


def class_label(cls: float) -> str:
    if np.isnan(cls):
        return ""
    return CLASS_NAMES.get(int(cls), f"class_{int(cls)}")


def team_of_player(player_id: float) -> int:
    """Heuristic team grouping: ids in 100s -> team 0, ids in 200s -> team 1."""
    if np.isnan(player_id):
        return -1
    return 0 if int(player_id) < 200 else 1


def scale_roi_to_video(
    arr: np.ndarray, video_width: int, video_height: int
) -> np.ndarray:
    """Return an (N, 4) array of (x1, y1, x2, y2) bboxes scaled to video size.

    Rows with NaN ROI values are returned as NaN. Coordinates are clamped to
    the video frame.
    """
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    sx = video_width / FULLHD_W
    sy = video_height / FULLHD_H
    rx = arr[:, COL["roi_x"]] * sx
    ry = arr[:, COL["roi_y"]] * sy
    rw = arr[:, COL["roi_width"]] * sx
    rh = arr[:, COL["roi_height"]] * sy
    x1 = rx
    y1 = ry
    x2 = rx + rw
    y2 = ry + rh
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return boxes


def filter_valid_rois(arr: np.ndarray) -> np.ndarray:
    """Drop rows where any ROI column is NaN."""
    if arr.size == 0:
        return arr
    roi_cols = arr[:, [COL["roi_x"], COL["roi_y"], COL["roi_width"], COL["roi_height"]]]
    mask = ~np.isnan(roi_cols).any(axis=1)
    return arr[mask]


def frame_range(arr: np.ndarray) -> tuple[int, int]:
    if arr.size == 0:
        return (0, 0)
    return int(arr[:, COL["frame"]].min()), int(arr[:, COL["frame"]].max())


def iter_match_summaries(output_dir: Path) -> Iterable[dict]:
    for match in list_matches(output_dir):
        arr = concat_match_array(match)
        fmin, fmax = frame_range(arr)
        events = select_events(arr)
        yield {
            "match_id": match.match_id,
            "video_path": str(match.video_path),
            "video_exists": match.video_path.exists(),
            "tactical_h5": str(match.tactical_h5),
            "halves": match.halves,
            "row_count": int(arr.shape[0]),
            "event_count": int(events.shape[0]),
            "frame_min": fmin,
            "frame_max": fmax,
        }
