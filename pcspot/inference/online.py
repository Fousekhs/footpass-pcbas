"""Memory-bounded online inference wrapper.

This module is the streaming counterpart to the offline / batched
training path. It is built around the same
:class:`pcspot.models.pipeline.PlayerCentricSpottingModel`, but ingests
raw video frames + tactical rows one frame at a time and produces
sliding-window predictions without ever materialising the full match
in memory.

Design rules:

- A single :class:`SlidingWindowBuffer` of size ``window_size`` holds
  the most recent ``T = window_size`` frames worth of:
  - per-frame tactical rows (numpy ``(N_t, 14)`` arrays);
  - per-frame visible-player ROI lists;
  - per-frame visible-player visual embeddings (``F_visual``).
  The buffer is a fixed-length deque; each ``push`` evicts the
  oldest frame **and immediately drops the corresponding tensors**
  so they can be garbage collected before the next forward pass.
- All tensors that are large enough to matter (the visual
  embeddings, the model's intermediate activations) are computed
  inside ``torch.inference_mode()`` so no autograd tape is built.
- Visual features are stored on CPU inside the buffer to keep GPU
  memory bounded by the model + the active ``StackedSample`` only;
  they are moved to the model's device once per window.

The public entry point is :class:`OnlineSpotter`. A typical loop
looks like::

    spotter = OnlineSpotter(model, dino, cropper, OnlineInferenceConfig(...))
    for frame_idx, (frame_image, tactical_rows) in enumerate(stream):
        preds = spotter.step(frame_idx, frame_image, tactical_rows)
        for p in preds:
            ...  # write to JSON / Codabench format

``spotter.step`` returns predictions only when the window is full and
the configured stride has elapsed; otherwise it returns an empty list.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - keeps the module importable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from pcspot.data.loader import (
    COL_CLASS,
    COL_FRAME,
    COL_LEFT_TO_RIGHT,
    COL_PLAYER_ID,
    COL_ROI_H,
    COL_ROI_W,
    COL_ROI_X,
    COL_ROI_Y,
    COL_ROLE,
    COL_SHIRT,
    COL_SPEED_X,
    COL_SPEED_Y,
    COL_X,
    COL_Y,
)
from pcspot.data.schema import (
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
    StackedSample,
    attach_visual_features,
    stack_sample,
)
from pcspot.eval.nms import (
    Prediction,
    decode_predictions,
    player_centric_nms,
)
from pcspot.features.cropper import PaddedPlayerCropper
from pcspot.models.pipeline import (
    PlayerCentricSpottingModel,
    StackedSampleBatch,
    stacked_to_batch,
)


@dataclass
class OnlineInferenceConfig:
    """Configuration for :class:`OnlineSpotter`.

    Attributes:
        window_size: temporal window length ``T`` fed to the model.
            Must match the configuration the model was trained with.
        stride: how often to run the model, in frames. ``stride=1``
            re-runs every frame (most accurate, most expensive);
            ``stride=window_size // 2`` halves cost with overlap-1/2.
        nms_radius: temporal radius (frames) used by
            :func:`player_centric_nms`.
        score_threshold: predictions below this score are suppressed.
        device: torch device for the model and for moving the
            visual / kinematic tensors during ``forward``.
        emit_only_last_step: when True, decoder only looks at the
            last frame of each window so we never emit duplicate
            predictions for overlapping windows. Recommended for
            single-pass evaluation.
        match_id / half_id: provenance written into the
            ``StackedSample`` meta on each step. Defaults are placeholders.
        fps: recording fps used for time-feature encoding.
    """

    window_size: int = 64
    stride: int = 1
    nms_radius: int = 3
    score_threshold: float = 0.05
    device: str = "cpu"
    emit_only_last_step: bool = True
    match_id: str = "online"
    half_id: str = "0"
    fps: float = 25.0


@dataclass
class _FrameRecord:
    """Per-frame state buffered in :class:`SlidingWindowBuffer`."""

    frame_index: int
    snapshots: list[PlayerSnapshot]
    events: list[EventLabel]
    # Per-player visual features keyed by player_id -> (F,) float32.
    visual_features: dict[int, np.ndarray] = field(default_factory=dict)


class SlidingWindowBuffer:
    """Fixed-length deque of :class:`_FrameRecord`s.

    Pushing a new frame past ``window_size`` evicts the oldest record
    and explicitly drops the inner numpy arrays so Python can GC them
    immediately. Read access is via :meth:`build_sample`, which
    constructs a fresh :class:`Sample` covering the current window.
    """

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self._capacity = int(window_size)
        self._records: deque[_FrameRecord] = deque(maxlen=self._capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._records)

    def is_full(self) -> bool:
        return len(self._records) == self._capacity

    def push(self, record: _FrameRecord) -> None:
        if self._records.maxlen is not None and len(self._records) == self._records.maxlen:
            evicted = self._records[0]
            evicted.snapshots = []
            evicted.events = []
            evicted.visual_features.clear()
        self._records.append(record)

    def latest_frame_index(self) -> int:
        if not self._records:
            raise RuntimeError("buffer is empty")
        return self._records[-1].frame_index

    def frames(self) -> list[int]:
        return [r.frame_index for r in self._records]

    def build_sample(
        self,
        match_id: str,
        half_id: str,
        fps: float,
    ) -> tuple[Sample, dict[int, dict[int, np.ndarray]]]:
        """Return ``(sample, visual_by_frame)`` for the current window.

        ``visual_by_frame`` maps ``frame_index -> {player_id: vec}``
        so the caller can attach features after stacking. Sample
        events come from buffered tactical rows (rare in online use).
        """
        frames = np.array([r.frame_index for r in self._records], dtype=np.int64)
        per_step = [list(r.snapshots) for r in self._records]
        events: list[EventLabel] = []
        for r in self._records:
            events.extend(r.events)
        sample = Sample(
            frames=frames,
            players_per_step=per_step,
            events=events,
            meta=SampleMeta(match_id=match_id, half_id=half_id, fps=fps),
        )
        visual_by_frame = {r.frame_index: dict(r.visual_features) for r in self._records}
        return sample, visual_by_frame


def _row_to_snapshot(row: np.ndarray) -> PlayerSnapshot:
    """Lightweight in-line version of ``loader._row_to_snapshot``.

    We avoid importing the loader's private helper to keep this
    module self-contained for streaming jobs.
    """
    pid = int(row[COL_PLAYER_ID])
    bbox = (
        float(row[COL_ROI_X]),
        float(row[COL_ROI_Y]),
        float(row[COL_ROI_W]),
        float(row[COL_ROI_H]),
    )
    visible = all(np.isfinite(b) for b in bbox)
    role_id_raw = float(row[COL_ROLE])
    role_id = int(role_id_raw) if np.isfinite(role_id_raw) else 0
    shirt_raw = float(row[COL_SHIRT])
    shirt = int(shirt_raw) if np.isfinite(shirt_raw) else -1
    return PlayerSnapshot(
        player_id=pid,
        team=0 if pid < 200 else 1,
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


class OnlineSpotter:
    """Streaming inference wrapper for ``PlayerCentricSpottingModel``.

    The wrapper owns the sliding window, the visual extractor, and
    the cropper. It does **not** own the video reader; callers feed
    decoded RGB frames one at a time. This keeps the wrapper
    independent of OpenCV / PyAV and makes it trivial to plug into
    Codabench-style evaluation harnesses.
    """

    def __init__(
        self,
        model: PlayerCentricSpottingModel,
        extractor,
        cropper: PaddedPlayerCropper,
        config: OnlineInferenceConfig | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for OnlineSpotter")
        self.model = model.eval()
        self.extractor = extractor
        self.cropper = cropper
        self.config = config or OnlineInferenceConfig()
        self.buffer = SlidingWindowBuffer(self.config.window_size)
        self._step_count = 0
        self._last_emitted_frame: int | None = None
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.config.device)

    def reset(self) -> None:
        """Clear the buffer between halves / matches."""
        self.buffer = SlidingWindowBuffer(self.config.window_size)
        self._step_count = 0
        self._last_emitted_frame = None

    def _ingest(
        self,
        frame_index: int,
        frame: np.ndarray,
        tactical_rows: np.ndarray,
    ) -> None:
        """Extract crops and visual embeddings for one frame, then push."""
        if tactical_rows.size == 0:
            self.buffer.push(_FrameRecord(frame_index=frame_index, snapshots=[], events=[]))
            return
        snapshots: list[PlayerSnapshot] = []
        events: list[EventLabel] = []
        rois: list[tuple[float, float, float, float]] = []
        pids: list[int] = []
        for row in tactical_rows:
            snap = _row_to_snapshot(row)
            snapshots.append(snap)
            if snap.visible:
                rois.append(tuple(float(v) for v in snap.bbox_xywh))  # type: ignore[arg-type]
                pids.append(int(snap.player_id))
            cls_val = float(row[COL_CLASS]) if row.shape[0] > COL_CLASS else 0.0
            if np.isfinite(cls_val) and int(cls_val) != 0:
                events.append(
                    EventLabel(
                        frame=int(row[COL_FRAME]),
                        player_id=int(snap.player_id),
                        class_id=int(cls_val),
                    )
                )
        visual: dict[int, np.ndarray] = {}
        if rois:
            crops, mask = self.cropper.extract(frame, np.asarray(rois, dtype=np.float32))
            if mask.any():
                feats = self.extractor.extract_features(crops[mask])
                kept_pids = [pid for pid, ok in zip(pids, mask.tolist()) if ok]
                for pid, vec in zip(kept_pids, feats):
                    visual[pid] = vec.astype(np.float32, copy=False)
        self.buffer.push(
            _FrameRecord(
                frame_index=frame_index,
                snapshots=snapshots,
                events=events,
                visual_features=visual,
            )
        )

    def _attach_visuals(
        self,
        stacked: StackedSample,
        visual_by_frame: dict[int, dict[int, np.ndarray]],
    ) -> StackedSample:
        if not getattr(self.model, "visual_dim", 0):
            return stacked
        F = int(self.model.visual_dim)
        T, P = stacked.num_steps, stacked.num_players
        feats = np.zeros((T, P, F), dtype=np.float32)
        pid_to_col = {int(pid): i for i, pid in enumerate(stacked.player_ids.tolist())}
        for ti, frame in enumerate(stacked.frames.tolist()):
            row_visual = visual_by_frame.get(int(frame), {})
            for pid, vec in row_visual.items():
                col = pid_to_col.get(int(pid))
                if col is None:
                    continue
                if vec.shape[0] != F:
                    raise ValueError(
                        f"visual feature dim mismatch: got {vec.shape[0]}, "
                        f"model expects {F}"
                    )
                feats[ti, col] = vec
        feats = feats * stacked.valid_mask[:, :, None].astype(np.float32)
        return attach_visual_features(stacked, feats)

    def _should_emit(self) -> bool:
        if not self.buffer.is_full():
            return False
        return (self._step_count % max(1, int(self.config.stride))) == 0

    def _decode(
        self,
        outputs: dict[str, "torch.Tensor"],  # type: ignore[name-defined]
        stacked: StackedSample,
    ) -> list[Prediction]:
        logits = outputs["logits"][0].detach().cpu()
        confidence = (
            outputs["confidence"][0].detach().cpu()
            if "confidence" in outputs
            else None
        )
        valid = stacked.valid_mask
        preds = decode_predictions(
            logits=logits,
            confidence=confidence,
            valid_mask=valid,
            player_ids=stacked.player_ids,
            score_threshold=self.config.score_threshold,
        )
        if self.config.emit_only_last_step:
            last_t = stacked.num_steps - 1
            preds = [p for p in preds if p.time == last_t]
        # Translate window-local time back to absolute frames.
        window_frames = stacked.frames.tolist()
        absolute = []
        for p in preds:
            abs_frame = int(window_frames[p.time])
            if (
                self._last_emitted_frame is not None
                and abs_frame <= self._last_emitted_frame
            ):
                continue
            absolute.append(
                Prediction(
                    time=abs_frame,
                    class_id=p.class_id,
                    player_id=p.player_id,
                    score=p.score,
                )
            )
        suppressed = player_centric_nms(absolute, window_radius=self.config.nms_radius)
        if suppressed:
            self._last_emitted_frame = max(p.time for p in suppressed)
        return suppressed

    def step(
        self,
        frame_index: int,
        frame: np.ndarray,
        tactical_rows: np.ndarray,
    ) -> list[Prediction]:
        """Ingest one frame and emit (possibly empty) predictions.

        Args:
            frame_index: absolute frame index.
            frame: ``(H, W, 3)`` RGB image.
            tactical_rows: ``(N_t, 14)`` tactical rows for this frame
                (matching ``pcspot.data.loader`` column layout).

        Returns:
            List of :class:`Prediction` for predictions whose absolute
            time is past ``self._last_emitted_frame``. May be empty.
        """
        self._ingest(frame_index, frame, tactical_rows)
        self._step_count += 1
        if not self._should_emit():
            return []
        sample, visual_by_frame = self.buffer.build_sample(
            match_id=self.config.match_id,
            half_id=self.config.half_id,
            fps=self.config.fps,
        )
        stacked = stack_sample(sample)
        if stacked.num_players == 0:
            return []
        stacked = self._attach_visuals(stacked, visual_by_frame)
        batch = stacked_to_batch([stacked])
        batch = self._move_batch(batch)
        with torch.inference_mode():
            outputs = self.model(batch)
        preds = self._decode(outputs, stacked)
        # Aggressive eviction: drop intermediate references so Python
        # can free the GPU tensors before the next ``step`` allocates.
        del batch
        del outputs
        return preds

    def _move_batch(self, batch: StackedSampleBatch) -> StackedSampleBatch:
        device = self.config.device

        def _opt(t):
            return t.to(device) if t is not None else None

        return StackedSampleBatch(
            pitch_xy=batch.pitch_xy.to(device),
            velocity=batch.velocity.to(device),
            bbox=batch.bbox.to(device),
            roles=batch.roles.to(device),
            teams=batch.teams.to(device),
            valid_mask=batch.valid_mask.to(device),
            targets_class=batch.targets_class.to(device),
            global_features=_opt(batch.global_features),
            acceleration=_opt(batch.acceleration),
            time_features=_opt(batch.time_features),
            frames=_opt(batch.frames),
            visual_features=_opt(batch.visual_features),
            left_to_right=_opt(batch.left_to_right),
            shirt_numbers=_opt(batch.shirt_numbers),
        )
