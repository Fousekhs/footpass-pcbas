"""FOOTPASS / PCBAS dataset adapter.

Wraps the existing ``scripts/pcbas_data.py`` loader to produce ``Sample``
objects in the schema defined by ``pcspot.data.schema``. The adapter
keeps the FOOTPASS column conventions (player_id, normalized x/y,
ROI bbox in fullHD pixels) and only restructures rows into per-frame
player snapshots and per-event labels.

The adapter does not require the videos themselves: tactical data
plus team/jersey/role columns are enough to drive the graph + temporal
pipeline. Visual features (e.g. CNN crops) can be attached later via
``Sample.global_features`` or, for per-player crop embeddings, by
extending ``PlayerSnapshot``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from pcspot.data.schema import (
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
)
from pcspot.data.windows import Window, iter_windows, split_array_by_windows


_PCBAS_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_PCBAS_SCRIPT_DIR) not in sys.path:
    sys.path.append(str(_PCBAS_SCRIPT_DIR))


# Column indices follow scripts/pcbas_data.py exactly.
COL_FRAME = 0
COL_PLAYER_ID = 1
COL_LEFT_TO_RIGHT = 2
COL_SHIRT = 3
COL_ROLE = 4
COL_X = 5
COL_Y = 6
COL_SPEED_X = 7
COL_SPEED_Y = 8
COL_ROI_X = 9
COL_ROI_Y = 10
COL_ROI_W = 11
COL_ROI_H = 12
COL_CLASS = 13
EXPECTED_NCOLS = 14


@dataclass
class HalfArray:
    """Container for one match-half tactical array."""

    match_id: str
    half_id: str
    array: np.ndarray  # (N, 14) float32

    def __post_init__(self) -> None:
        if self.array.ndim != 2 or self.array.shape[1] != EXPECTED_NCOLS:
            raise ValueError(
                f"tactical array for {self.match_id}/{self.half_id} must be (N, {EXPECTED_NCOLS})"
            )

    @property
    def frame_min(self) -> int:
        if self.array.size == 0:
            return 0
        return int(self.array[:, COL_FRAME].min())

    @property
    def frame_max(self) -> int:
        if self.array.size == 0:
            return -1
        return int(self.array[:, COL_FRAME].max())


def _team_of_player(player_id: float) -> int:
    if not np.isfinite(player_id):
        return -1
    return 0 if int(player_id) < 200 else 1


def _row_to_snapshot(row: np.ndarray) -> PlayerSnapshot:
    pid = int(row[COL_PLAYER_ID])
    bbox = (
        float(row[COL_ROI_X]),
        float(row[COL_ROI_Y]),
        float(row[COL_ROI_W]),
        float(row[COL_ROI_H]),
    )
    visible = all(np.isfinite(b) for b in bbox)
    role_id = int(row[COL_ROLE]) if np.isfinite(row[COL_ROLE]) else 0
    shirt = int(row[COL_SHIRT]) if np.isfinite(row[COL_SHIRT]) else -1
    return PlayerSnapshot(
        player_id=pid,
        team=_team_of_player(row[COL_PLAYER_ID]),
        shirt_number=shirt,
        role_id=role_id,
        x=float(row[COL_X]),
        y=float(row[COL_Y]),
        speed_x=float(row[COL_SPEED_X]),
        speed_y=float(row[COL_SPEED_Y]),
        bbox_xywh=bbox,
        visible=visible,
        left_to_right=float(row[COL_LEFT_TO_RIGHT]),
    )


def _events_from_array(arr: np.ndarray) -> list[EventLabel]:
    if arr.size == 0:
        return []
    cls = arr[:, COL_CLASS]
    mask = cls != 0
    rows = arr[mask]
    return [
        EventLabel(
            frame=int(r[COL_FRAME]),
            player_id=int(r[COL_PLAYER_ID]),
            class_id=int(r[COL_CLASS]),
        )
        for r in rows
    ]


def array_to_sample(
    arr: np.ndarray,
    window: Window,
    *,
    match_id: str,
    half_id: Optional[str] = None,
    fps: float = 25.0,
    fullhd_width: int = 1920,
    fullhd_height: int = 1080,
) -> Sample:
    """Build a ``Sample`` covering ``window`` from a sorted tactical array."""
    if arr.size == 0:
        frames = np.arange(window.start, window.stop, dtype=np.int64)
        return Sample(
            frames=frames,
            players_per_step=[[] for _ in range(window.length)],
            events=[],
            meta=SampleMeta(
                match_id=match_id,
                half_id=half_id,
                fps=fps,
                fullhd_width=fullhd_width,
                fullhd_height=fullhd_height,
            ),
        )

    frames_col = arr[:, COL_FRAME].astype(np.int64)
    order = np.argsort(frames_col, kind="stable")
    arr_sorted = arr[order]
    frames_sorted = frames_col[order]

    frames = np.arange(window.start, window.stop, dtype=np.int64)
    players_per_step: list[list[PlayerSnapshot]] = []
    for f in frames:
        lo = np.searchsorted(frames_sorted, f, side="left")
        hi = np.searchsorted(frames_sorted, f, side="right")
        rows = arr_sorted[lo:hi]
        players_per_step.append([_row_to_snapshot(r) for r in rows])

    lo = np.searchsorted(frames_sorted, window.start, side="left")
    hi = np.searchsorted(frames_sorted, window.stop, side="left")
    events = _events_from_array(arr_sorted[lo:hi])

    return Sample(
        frames=frames,
        players_per_step=players_per_step,
        events=events,
        meta=SampleMeta(
            match_id=match_id,
            half_id=half_id,
            fps=fps,
            fullhd_width=fullhd_width,
            fullhd_height=fullhd_height,
        ),
    )


def iter_samples_from_half(
    half: HalfArray,
    *,
    window_size: int,
    stride: Optional[int] = None,
    drop_last: bool = False,
    fps: float = 25.0,
    fullhd_width: int = 1920,
    fullhd_height: int = 1080,
) -> Iterator[Sample]:
    """Yield ``Sample`` objects for each window in a match-half array."""
    if half.array.size == 0:
        return
    windows = list(
        iter_windows(
            half.frame_min,
            half.frame_max,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
        )
    )
    if not windows:
        return
    chunks = split_array_by_windows(half.array, COL_FRAME, windows)
    for window, chunk in zip(windows, chunks):
        yield array_to_sample(
            chunk,
            window,
            match_id=half.match_id,
            half_id=half.half_id,
            fps=fps,
            fullhd_width=fullhd_width,
            fullhd_height=fullhd_height,
        )


def load_halves_from_pcbas(
    output_dir: Path,
    *,
    match_id: Optional[str] = None,
    include_challenge: bool = False,
) -> list[HalfArray]:
    """Discover halves on disk via ``scripts/pcbas_data.py``.

    Returns one ``HalfArray`` per (match, half). Filters to ``match_id``
    if provided.

    Challenge-split tactical arrays are missing the trailing ``class``
    column (13 columns instead of the usual 14). Pass
    ``include_challenge=True`` to include them anyway, with that column
    zero-padded back in -- safe for inference, where the resulting (empty)
    events are never used.
    """
    import pcbas_data  # type: ignore  # injected from scripts/

    halves: list[HalfArray] = []
    for match in pcbas_data.list_matches(output_dir, include_challenge=include_challenge):
        if match_id is not None and match.match_id != match_id:
            continue
        arrays = pcbas_data.load_tactical_arrays(match)
        for half_key, arr in arrays.items():
            arr = arr.astype(np.float32, copy=False)
            if arr.shape[1] == EXPECTED_NCOLS - 1:
                arr = np.concatenate(
                    [arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1
                )
            halves.append(
                HalfArray(
                    match_id=match.match_id,
                    half_id=half_key,
                    array=arr,
                )
            )
    return halves
