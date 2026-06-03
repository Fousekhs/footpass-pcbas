"""Window slicing utilities for long match sequences.

Match-level tactical arrays cover tens of thousands of frames per
half. Training and evaluation operate on windows of ``window_size``
frames with optional stride / overlap. These helpers are dataset-
agnostic and only use numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class Window:
    """An inclusive-exclusive window over absolute frame indices."""

    start: int
    stop: int  # exclusive

    @property
    def length(self) -> int:
        return self.stop - self.start

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.stop


def iter_windows(
    frame_min: int,
    frame_max: int,
    window_size: int,
    stride: int | None = None,
    drop_last: bool = False,
) -> Iterator[Window]:
    """Yield ``Window`` objects covering ``[frame_min, frame_max + 1)``.

    Args:
        frame_min: smallest frame index that the iterator should cover.
        frame_max: largest frame index (inclusive).
        window_size: number of frames per window.
        stride: hop size; defaults to ``window_size`` (non-overlapping).
        drop_last: if True, drop the last partial window when the total
            range does not divide evenly.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if frame_max < frame_min:
        return

    total = frame_max - frame_min + 1
    last_full_start = frame_min + ((total - window_size) // stride) * stride
    cursor = frame_min
    while cursor + window_size <= frame_max + 1:
        yield Window(cursor, cursor + window_size)
        if cursor == last_full_start:
            break
        cursor += stride

    if not drop_last:
        # Emit a tail window that ends at frame_max + 1 if the previous
        # iterator did not already cover it.
        tail_start = max(frame_min, frame_max + 1 - window_size)
        if tail_start > last_full_start:
            yield Window(tail_start, frame_max + 1)


def split_array_by_windows(
    arr: np.ndarray,
    frame_col: int,
    windows: Iterable[Window],
) -> list[np.ndarray]:
    """Return one numpy slice per window for a sorted-by-frame array."""
    if arr.size == 0:
        return [arr.copy() for _ in windows]
    frames = arr[:, frame_col].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    arr_sorted = arr[order]
    frames_sorted = frames[order]
    out: list[np.ndarray] = []
    for w in windows:
        lo = np.searchsorted(frames_sorted, w.start, side="left")
        hi = np.searchsorted(frames_sorted, w.stop, side="left")
        out.append(arr_sorted[lo:hi])
    return out
