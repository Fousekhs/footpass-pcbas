"""Lazy dataset for player-centric action spotting.

``PCBASDataset`` is a ``torch.utils.data.Dataset`` over (match, half,
window) triples. It materializes a ``StackedSample`` per item on demand
and, optionally, the matching PC-CALF and objectness target tensors.

Two ingestion modes are supported:

1. **HalfArray-backed**: pass already-loaded ``HalfArray`` instances. The
   dataset stores references and slices windows lazily. Good for unit
   tests and small datasets.
2. **Loader-backed**: pass a callable ``half_loader(match_id, half_id)``
   that returns a ``HalfArray``. The dataset keeps an LRU of recently
   loaded halves to avoid re-reading HDF5 for every window.

The dataset enforces split discipline via ``SplitManifest``: each
(match, half) must be assigned to exactly one split, and the dataset is
constructed for one split at a time.

Items emitted by ``__getitem__`` are ``(StackedSample, BatchTargets | None)``
tuples. Building targets is the slow step in CPU training; the dataset
optionally hands off to a ``TargetCache`` so repeated epochs do not pay
the kernel cost twice.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - keeps the module importable without torch
    Dataset = object  # type: ignore[assignment,misc]

from pcspot.data.cache import TargetCache
from pcspot.data.loader import (
    COL_CLASS,
    COL_FRAME,
    HalfArray,
    array_to_sample,
)
from pcspot.data.schema import (
    NUM_PCBAS_CLASSES,
    Sample,
    StackedSample,
    stack_sample,
)
from pcspot.data.splits import SplitManifest
from pcspot.data.targets import (
    CalfConfig,
    build_objectness_targets,
    build_pc_calf_targets,
)
from pcspot.data.windows import Window, iter_windows

try:
    # Visual feature cache is optional; importing lazily here avoids
    # forcing torch/PIL deps on tactical-only training jobs.
    from pcspot.features.cache import (  # type: ignore[import-not-found]
        VisualFeatureCache,
        align_features_to_stacked,
    )

    _VISUAL_CACHE_AVAILABLE = True
except Exception:  # pragma: no cover - keeps the dataset usable without visuals
    VisualFeatureCache = None  # type: ignore[assignment,misc]
    align_features_to_stacked = None  # type: ignore[assignment]
    _VISUAL_CACHE_AVAILABLE = False


HalfLoader = Callable[[str, str], HalfArray]


@dataclass(frozen=True)
class DatasetItemKey:
    """Stable key identifying one ``PCBASDataset`` item."""

    match_id: str
    half_id: str
    window_start: int
    window_size: int

    @property
    def window(self) -> Window:
        return Window(self.window_start, self.window_start + self.window_size)

    def cache_key(self) -> str:
        return (
            f"{self.match_id}|{self.half_id}|{self.window_start}|{self.window_size}"
        )


@dataclass
class SampleTargets:
    """All tensors associated with a single dataset item."""

    class_targets: np.ndarray  # (T, P, C)
    class_weights: np.ndarray  # (T, P, C)
    objectness_targets: np.ndarray  # (T, P)
    objectness_weights: np.ndarray  # (T, P)


class _HalfArrayLRU:
    """Tiny LRU for HalfArray instances. Single-thread safe."""

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._data: OrderedDict[tuple[str, str], HalfArray] = OrderedDict()

    def get_or_load(self, key: tuple[str, str], loader: HalfLoader) -> HalfArray:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        half = loader(key[0], key[1])
        self._data[key] = half
        if len(self._data) > self._capacity:
            self._data.popitem(last=False)
        return half


class PCBASDataset(Dataset):  # type: ignore[misc]
    """Lazy window-level dataset for the player-centric PCBAS pipeline."""

    def __init__(
        self,
        *,
        manifest: SplitManifest,
        split: str,
        window_size: int,
        stride: Optional[int] = None,
        drop_last: bool = False,
        halves: Optional[Sequence[HalfArray]] = None,
        half_loader: Optional[HalfLoader] = None,
        lru_capacity: int = 4,
        calf_config: Optional[CalfConfig] = None,
        target_cache: Optional[TargetCache] = None,
        num_classes: int = NUM_PCBAS_CLASSES,
        compute_targets: bool = True,
        fps: float = 25.0,
        visual_feature_cache: Optional["VisualFeatureCache"] = None,
    ) -> None:
        manifest.ensure_no_overlap()
        self._manifest = manifest
        self._split = split
        self._window_size = int(window_size)
        self._stride = int(stride) if stride is not None else int(window_size)
        self._drop_last = bool(drop_last)
        self._calf_config = calf_config or CalfConfig()
        self._target_cache = target_cache
        self._num_classes = num_classes
        self._compute_targets = bool(compute_targets)
        self._fps = float(fps)
        self._visual_feature_cache = visual_feature_cache
        if visual_feature_cache is not None and not _VISUAL_CACHE_AVAILABLE:
            raise RuntimeError(
                "visual_feature_cache was provided but pcspot.features could "
                "not be imported. Install the visual extras (torch, etc.) "
                "or run without the cache."
            )

        self._halves_index: dict[tuple[str, str], HalfArray] = {}
        if halves is not None:
            for half in halves:
                key = (str(half.match_id), str(half.half_id))
                if key in self._halves_index:
                    raise ValueError(f"duplicate half {key}")
                self._halves_index[key] = half

        self._half_loader = half_loader
        self._lru = _HalfArrayLRU(capacity=lru_capacity) if half_loader else None

        # Build the per-item index: list of (key, half-or-None, event_count).
        self._items: list[DatasetItemKey] = []
        self._event_counts: list[int] = []
        for key in manifest.halves_for(split):
            half = self._halves_index.get(key)
            if half is None and half_loader is None:
                raise ValueError(
                    f"Half {key} is in the manifest for split {split!r} "
                    "but no halves were provided and no half_loader was given."
                )
            if half is None:
                half = self._lru.get_or_load(key, half_loader)  # type: ignore[union-attr]
            for w in iter_windows(
                half.frame_min,
                half.frame_max,
                window_size=self._window_size,
                stride=self._stride,
                drop_last=self._drop_last,
            ):
                item = DatasetItemKey(
                    match_id=key[0],
                    half_id=key[1],
                    window_start=int(w.start),
                    window_size=self._window_size,
                )
                self._items.append(item)
                self._event_counts.append(_count_events_in_window(half, w))

    def __len__(self) -> int:
        return len(self._items)

    @property
    def split(self) -> str:
        return self._split

    @property
    def items(self) -> list[DatasetItemKey]:
        return list(self._items)

    @property
    def event_counts(self) -> list[int]:
        return list(self._event_counts)

    def _half_for(self, match_id: str, half_id: str) -> HalfArray:
        key = (match_id, half_id)
        if key in self._halves_index:
            return self._halves_index[key]
        if self._lru is None or self._half_loader is None:
            raise KeyError(f"No half loader configured for {key}")
        return self._lru.get_or_load(key, self._half_loader)

    def build_sample(self, item: DatasetItemKey) -> Sample:
        half = self._half_for(item.match_id, item.half_id)
        window = item.window
        arr = half.array
        if arr.size:
            frames = arr[:, COL_FRAME].astype(np.int64)
            lo = np.searchsorted(frames, window.start, side="left")
            hi = np.searchsorted(frames, window.stop, side="left")
            chunk = arr[lo:hi]
        else:
            chunk = arr
        return array_to_sample(
            chunk,
            window,
            match_id=item.match_id,
            half_id=item.half_id,
            fps=self._fps,
        )

    def __getitem__(self, idx: int) -> tuple[StackedSample, Optional[SampleTargets]]:
        item = self._items[idx]
        sample = self.build_sample(item)
        stacked = stack_sample(sample)
        if self._visual_feature_cache is not None:
            stacked = align_features_to_stacked(stacked, self._visual_feature_cache)
        if not self._compute_targets:
            return stacked, None
        targets = self._load_or_build_targets(item, stacked)
        return stacked, targets

    def _load_or_build_targets(
        self, item: DatasetItemKey, stacked: StackedSample
    ) -> SampleTargets:
        if self._target_cache is not None:
            cached = self._target_cache.load(item.cache_key())
            if cached is not None:
                ct, cw, ot, ow = cached
                return SampleTargets(
                    class_targets=ct,
                    class_weights=cw,
                    objectness_targets=ot,
                    objectness_weights=ow,
                )
        ct, cw = build_pc_calf_targets(
            stacked, config=self._calf_config, num_classes=self._num_classes
        )
        ot, ow = build_objectness_targets(ct, cw)
        if self._target_cache is not None:
            self._target_cache.save(item.cache_key(), ct, cw, ot, ow)
        return SampleTargets(
            class_targets=ct,
            class_weights=cw,
            objectness_targets=ot,
            objectness_weights=ow,
        )


def _count_events_in_window(half: HalfArray, window: Window) -> int:
    arr = half.array
    if arr.size == 0:
        return 0
    frames = arr[:, COL_FRAME].astype(np.int64)
    lo = np.searchsorted(frames, window.start, side="left")
    hi = np.searchsorted(frames, window.stop, side="left")
    if lo == hi:
        return 0
    return int((arr[lo:hi, COL_CLASS] != 0).sum())
