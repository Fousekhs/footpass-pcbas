"""On-disk cache for per-player visual feature embeddings.

The visual cache stores frozen-backbone embeddings indexed by
``(match_id, half_id, frame, player_id)`` and serves them as
``(T, P, F_visual)`` tensors aligned to a :class:`StackedSample`.

File layout
-----------

The cache is laid out one ``.npz`` file per match-half::

    <root>/<backbone_name>/<match_id>__<half_id>.npz

Each ``.npz`` contains four arrays:

- ``frames``: ``(N,)`` int64 frame indices, monotonically sorted.
- ``player_ids``: ``(N,)`` int64 player ids matching ``frames`` row-by-row.
- ``features``: ``(N, F_visual)`` float32 embeddings (one row per
  ``(frame, player_id)``).
- ``visible``: ``(N,)`` bool flag indicating whether the row was a
  real crop (``True``) or a zero-fill placeholder (``False``).

A small JSON sidecar (``<file>.json``) carries the backbone name and
feature dimension so the dataset can validate cache compatibility
without opening every shard.

Reading works in two phases:

1. The :class:`VisualFeatureCache` is opened with the cache root and
   backbone tag; it lazily memory-maps shards on first access and
   keeps a small LRU of frame-major lookup indices.
2. The dataset asks for a window via :func:`align_features_to_stacked`,
   which builds a ``(T, P, F_visual)`` tensor by looking up each
   ``(frame, player_id)`` pair in the cache.

Missing rows (player not in the cache for that frame) are left as
zeros, matching the embedder's "padded column == zero" convention.
The aligned tensor is therefore safe to attach to a
:class:`StackedSample` even when the cache is incomplete.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from pcspot.data.schema import StackedSample, attach_visual_features


@dataclass(frozen=True)
class VisualFeatureMetadata:
    """Sidecar metadata describing a cache shard / cache root."""

    backbone_name: str
    feature_dim: int
    crop_size: int
    pad_factor: float
    fullhd_width: int
    fullhd_height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "VisualFeatureMetadata":
        return cls(
            backbone_name=str(payload["backbone_name"]),
            feature_dim=int(payload["feature_dim"]),
            crop_size=int(payload["crop_size"]),
            pad_factor=float(payload["pad_factor"]),
            fullhd_width=int(payload["fullhd_width"]),
            fullhd_height=int(payload["fullhd_height"]),
        )


def _shard_basename(match_id: str, half_id: str) -> str:
    safe_match = match_id.replace("/", "_")
    safe_half = half_id.replace("/", "_")
    return f"{safe_match}__{safe_half}"


@dataclass
class _ShardIndex:
    """Lookup table for one match-half shard, frame-major."""

    frames: np.ndarray
    player_ids: np.ndarray
    features: np.ndarray
    visible: np.ndarray
    # frame -> slice into the row arrays (rows for that frame).
    frame_starts: dict[int, tuple[int, int]]

    @classmethod
    def from_arrays(
        cls,
        frames: np.ndarray,
        player_ids: np.ndarray,
        features: np.ndarray,
        visible: np.ndarray,
    ) -> "_ShardIndex":
        if not (frames.ndim == player_ids.ndim == visible.ndim == 1):
            raise ValueError("frames/player_ids/visible must be 1D")
        if frames.shape[0] != features.shape[0]:
            raise ValueError("frames and features must have same N")
        order = np.argsort(frames, kind="stable")
        frames = frames[order]
        player_ids = player_ids[order]
        features = features[order]
        visible = visible[order]
        starts: dict[int, tuple[int, int]] = {}
        if frames.size:
            unique, idx = np.unique(frames, return_index=True)
            ends = np.append(idx[1:], frames.size)
            for f, lo, hi in zip(unique.tolist(), idx.tolist(), ends.tolist()):
                starts[int(f)] = (int(lo), int(hi))
        return cls(
            frames=frames.astype(np.int64, copy=False),
            player_ids=player_ids.astype(np.int64, copy=False),
            features=features.astype(np.float32, copy=False),
            visible=visible.astype(bool, copy=False),
            frame_starts=starts,
        )

    def lookup(self, frame: int, player_id: int) -> np.ndarray | None:
        span = self.frame_starts.get(int(frame))
        if span is None:
            return None
        lo, hi = span
        pids = self.player_ids[lo:hi]
        idx = np.where(pids == int(player_id))[0]
        if idx.size == 0:
            return None
        if not bool(self.visible[lo + int(idx[0])]):
            return None
        return self.features[lo + int(idx[0])]

    def lookup_window(
        self,
        frames: Iterable[int],
        player_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised window lookup.

        Returns ``(features, valid)`` shaped ``(T, P, F)`` and ``(T, P)``
        respectively, where ``valid[t, p]`` is True when the cache had
        a visible row for that ``(frame, player_id)``.
        """
        req_frames = np.asarray(list(frames), dtype=np.int64)
        req_players = np.asarray(list(player_ids), dtype=np.int64)
        T = int(req_frames.shape[0])
        P = int(req_players.shape[0])
        F = int(self.features.shape[1]) if self.features.size else 0
        out = np.zeros((T, P, F), dtype=np.float32)
        valid = np.zeros((T, P), dtype=bool)
        if F == 0 or P == 0 or T == 0:
            return out, valid
        # Vectorise over players per frame instead of a T x P Python loop.
        # For each frame we match the P requested player ids against the rows
        # present for that frame with a single sorted searchsorted, which keeps
        # the cost at O(T * P log n) in NumPy rather than O(T * P) interpreted
        # dict lookups. (frame, player) pairs are unique per shard, so taking
        # the first sorted match is equivalent to the scalar ``lookup``.
        for ti in range(T):
            span = self.frame_starts.get(int(req_frames[ti]))
            if span is None:
                continue
            lo, hi = span
            pids = self.player_ids[lo:hi]
            if pids.size == 0:
                continue
            vis = self.visible[lo:hi]
            sorter = np.argsort(pids, kind="stable")
            sorted_pids = pids[sorter]
            pos = np.searchsorted(sorted_pids, req_players)
            in_range = pos < sorted_pids.shape[0]
            pos_clipped = np.where(in_range, pos, 0)
            matched_local = sorter[pos_clipped]  # row index within [lo:hi]
            found = in_range & (pids[matched_local] == req_players)
            rows_ok = found & vis[matched_local]
            if not rows_ok.any():
                continue
            sel_players = np.nonzero(rows_ok)[0]
            sel_rows = lo + matched_local[sel_players]
            out[ti, sel_players] = self.features[sel_rows]
            valid[ti, sel_players] = True
        return out, valid


class VisualFeatureStore:
    """Writer for visual feature shards.

    Use :meth:`add_frame` once per frame to append visible-player
    embeddings, then :meth:`flush` to write a shard. The store keeps
    rows in memory until ``flush`` to keep the file format simple
    (one ``.npz`` per match-half) and to avoid partial files in case
    of crashes.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: VisualFeatureMetadata,
    ) -> None:
        self.root = Path(root) / metadata.backbone_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self._buffers: dict[
            tuple[str, str], list[tuple[int, int, np.ndarray, bool]]
        ] = {}
        self._write_root_metadata()

    def _write_root_metadata(self) -> None:
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(self.metadata.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def shard_path(self, match_id: str, half_id: str) -> Path:
        return self.root / f"{_shard_basename(match_id, half_id)}.npz"

    def add_frame(
        self,
        match_id: str,
        half_id: str,
        frame: int,
        player_ids: Iterable[int],
        features: np.ndarray,
        visible: Iterable[bool] | None = None,
    ) -> None:
        """Append per-player rows for one frame."""
        pids = [int(p) for p in player_ids]
        N = len(pids)
        if features.ndim != 2 or features.shape[1] != self.metadata.feature_dim:
            raise ValueError(
                f"features must be (N, {self.metadata.feature_dim}), got "
                f"{features.shape}"
            )
        if features.shape[0] != N:
            raise ValueError("features and player_ids must agree on N")
        if visible is None:
            vis_flags = [True] * N
        else:
            vis_flags = [bool(v) for v in visible]
            if len(vis_flags) != N:
                raise ValueError("visible must agree on N")
        key = (str(match_id), str(half_id))
        bucket = self._buffers.setdefault(key, [])
        for i in range(N):
            bucket.append(
                (int(frame), pids[i], features[i].astype(np.float32, copy=False), vis_flags[i])
            )

    def flush(self) -> list[Path]:
        """Write all buffered shards. Returns paths of files written."""
        written: list[Path] = []
        for (match_id, half_id), rows in self._buffers.items():
            if not rows:
                continue
            frames_arr = np.array([r[0] for r in rows], dtype=np.int64)
            pids_arr = np.array([r[1] for r in rows], dtype=np.int64)
            features_arr = np.stack([r[2] for r in rows], axis=0).astype(np.float32, copy=False)
            visible_arr = np.array([r[3] for r in rows], dtype=bool)
            path = self.shard_path(match_id, half_id)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    frames=frames_arr,
                    player_ids=pids_arr,
                    features=features_arr,
                    visible=visible_arr,
                )
            tmp.replace(path)
            written.append(path)
            sidecar = path.with_suffix(".json")
            sidecar.write_text(
                json.dumps(self.metadata.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        self._buffers.clear()
        return written


class VisualFeatureCache:
    """Reader for visual feature shards.

    Keeps a small LRU of decoded shard indices so adjacent windows
    inside the same match-half share the same lookup table.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        backbone_name: str,
        lru_capacity: int = 4,
    ) -> None:
        if lru_capacity < 1:
            raise ValueError("lru_capacity must be >= 1")
        self.root = Path(root) / backbone_name
        self.backbone_name = backbone_name
        self._lru: OrderedDict[tuple[str, str], _ShardIndex] = OrderedDict()
        self._capacity = int(lru_capacity)
        self._metadata = self._load_root_metadata()

    def _load_root_metadata(self) -> Optional[VisualFeatureMetadata]:
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            return None
        return VisualFeatureMetadata.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )

    @property
    def metadata(self) -> Optional[VisualFeatureMetadata]:
        return self._metadata

    def feature_dim(self) -> int:
        if self._metadata is None:
            raise RuntimeError(
                f"Visual feature cache at {self.root} has no metadata.json. "
                "Run the precompute script before training."
            )
        return self._metadata.feature_dim

    def shard_path(self, match_id: str, half_id: str) -> Path:
        return self.root / f"{_shard_basename(match_id, half_id)}.npz"

    def _shard(self, match_id: str, half_id: str) -> _ShardIndex:
        key = (str(match_id), str(half_id))
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        path = self.shard_path(match_id, half_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Visual feature shard not found: {path}. Did you run the "
                "precompute script for this match-half?"
            )
        with np.load(path) as data:
            shard = _ShardIndex.from_arrays(
                frames=data["frames"][:],
                player_ids=data["player_ids"][:],
                features=data["features"][:],
                visible=data["visible"][:],
            )
        self._lru[key] = shard
        if len(self._lru) > self._capacity:
            self._lru.popitem(last=False)
        return shard

    def has_shard(self, match_id: str, half_id: str) -> bool:
        return self.shard_path(match_id, half_id).exists()

    def get_window(
        self,
        match_id: str,
        half_id: str,
        frames: Iterable[int],
        player_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(features, valid)`` for the given window.

        ``features`` is shaped ``(T, P, F_visual)`` float32.
        ``valid`` is shaped ``(T, P)`` bool.
        """
        shard = self._shard(match_id, half_id)
        return shard.lookup_window(frames, player_ids)


def align_features_to_stacked(
    stacked: StackedSample,
    cache: VisualFeatureCache,
) -> StackedSample:
    """Look up cached visual features for a sample and attach them.

    Padded / invisible columns and frames missing from the cache stay
    as zero rows, so the embedder can rely on ``valid_mask`` /
    ``visible`` for masking decisions. Returns a new
    :class:`StackedSample` with ``visual_features`` filled in.
    """
    F = cache.feature_dim()
    feats, valid = cache.get_window(
        match_id=stacked.meta.match_id,
        half_id=stacked.meta.half_id or "",
        frames=stacked.frames.tolist(),
        player_ids=stacked.player_ids.tolist(),
    )
    if feats.shape[-1] != F:
        # Defensive: get_window already returns the full F dim, but
        # if an empty window slipped through reshape it.
        feats = feats.reshape(stacked.num_steps, stacked.num_players, F)
    # Force-zero padded columns so the embedder never sees stale rows.
    feats = feats * stacked.valid_mask[:, :, None].astype(np.float32)
    return attach_visual_features(stacked, feats)
