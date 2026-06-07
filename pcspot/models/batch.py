"""Model-agnostic batch container and per-player feature utilities.

Holds the ``StackedSampleBatch`` consumed by every model variant plus the
helpers that turn a list of ``StackedSample`` into one, derive acceleration
(finite differences of velocity along T) and match-time positional features
(sinusoidal encodings of the absolute frame index at multiple periods), and
compute oriented goal/sideline distances.

Acceleration and time features are optional from a model's perspective: an
embedder ignores them unless ``use_acceleration`` / ``time_dim`` are set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from pcspot.data.schema import StackedSample


EXTRA_SCALAR_DIM: int = 5
"""Channels emitted into the embedder's ``extra_scalars`` branch:
``[dist_own_goal, dist_opp_goal, dist_nearest_sideline,
n_same_within_r, n_opp_within_r]``.

Distances are measured in normalized pitch units in the attacking-
canonical frame (so ``dist_opp_goal`` is consistent across both
squads). Degree counts come from the ``radius`` edge type and provide
the raw cardinality signal the kNN-censored ``near`` edge cannot.
"""


DEFAULT_NUM_JERSEYS: int = 100
"""Embedding-table size for jersey numbers. Index 0 is reserved for the
``-1`` / unknown sentinel; valid shirt numbers in PCBAS fit comfortably
into ``[1, 99]`` so 100 slots are enough."""


# Sinusoidal time-encoding periods, in seconds. Three octaves spanning
# short-range tactical context (5s), medium phases of play (60s), and
# half-scale match position (300s).
TIME_FEATURE_PERIODS_SEC: tuple[float, ...] = (5.0, 60.0, 300.0)
TIME_FEATURE_DIM: int = 2 * len(TIME_FEATURE_PERIODS_SEC)


def _compute_acceleration(velocity: torch.Tensor) -> torch.Tensor:
    """Return finite-difference acceleration with t=0 zero-padded."""
    if velocity.shape[1] == 0:
        return velocity.clone()
    a = torch.zeros_like(velocity)
    a[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    return a


def _compute_time_features(
    frames: torch.Tensor,  # (B, T) int64
    fps: float,
    periods_sec: Sequence[float] = TIME_FEATURE_PERIODS_SEC,
) -> torch.Tensor:
    """Return ``(B, T, 2 * len(periods))`` sinusoidal match-time encoding."""
    t_sec = frames.to(torch.float32) / float(fps)
    parts: list[torch.Tensor] = []
    for period in periods_sec:
        ang = (2.0 * math.pi) * t_sec / float(period)
        parts.append(torch.sin(ang))
        parts.append(torch.cos(ang))
    return torch.stack(parts, dim=-1)


def compute_goal_distances(
    pitch_xy: torch.Tensor,  # (B, T, P, 2)
    left_to_right: torch.Tensor,  # (B, T, P)
) -> torch.Tensor:
    """Return ``(B, T, P, 3)`` ``[dist_own_goal, dist_opp_goal, dist_sideline]``.

    Distances are in normalized pitch units and oriented to attacking
    direction via ``left_to_right`` (``+1`` = attacking +x,
    ``-1`` = attacking -x; the loader passes the FOOTPASS column
    through unchanged). The opponent goal is along the attacking
    direction; the own goal is the opposite goal-line.

    ``dist_sideline`` is the distance to the *nearer* of the two
    sidelines (``y = 0`` and ``y = 1``); it is symmetric across the
    pitch and does not depend on attacking direction.
    """
    x = pitch_xy[..., 0]
    y = pitch_xy[..., 1]
    # Attacking-canonical x: when attacking left-to-right (ltr >= 0),
    # the opponent goal is at x=1; otherwise it is at x=0.
    attacking_right = left_to_right >= 0
    dist_opp_goal = torch.where(attacking_right, 1.0 - x, x)
    dist_own_goal = torch.where(attacking_right, x, 1.0 - x)
    dist_sideline = torch.minimum(y, 1.0 - y)
    return torch.stack([dist_own_goal, dist_opp_goal, dist_sideline], dim=-1)


@dataclass
class StackedSampleBatch:
    """Padded batch of ``StackedSample`` instances ready for the model.

    Adds ``acceleration``, ``time_features``, and ``visual_features``
    over the original schema. All of them default to ``None`` so older
    callers keep working.

    ``visual_features`` is an optional ``(B, T, P, F_visual)`` tensor of
    per-player visual embeddings (e.g. frozen DINOv2 features extracted
    from padded player crops). It is consumed by ``PlayerNodeEmbedder``
    when ``visual_dim > 0`` is configured on the model.
    """

    pitch_xy: torch.Tensor  # (B, T, P, 2)
    velocity: torch.Tensor  # (B, T, P, 2)
    bbox: torch.Tensor  # (B, T, P, 4)
    roles: torch.Tensor  # (B, T, P) long
    teams: torch.Tensor  # (B, P) long
    valid_mask: torch.Tensor  # (B, T, P) bool
    targets_class: torch.Tensor  # (B, T, P) long
    global_features: torch.Tensor | None = None  # (B, T, F)
    acceleration: torch.Tensor | None = None  # (B, T, P, 2)
    time_features: torch.Tensor | None = None  # (B, T, F_time)
    frames: torch.Tensor | None = None  # (B, T) int64
    visual_features: torch.Tensor | None = None  # (B, T, P, F_visual)
    left_to_right: torch.Tensor | None = None  # (B, T, P) float32
    shirt_numbers: torch.Tensor | None = None  # (B, P) long


def stacked_to_batch(samples: Sequence[StackedSample]) -> StackedSampleBatch:
    """Pad-and-stack a list of ``StackedSample`` into a ``StackedSampleBatch``.

    All samples must share ``T``. Player columns are padded to the max
    ``P`` across the batch. Padded columns are masked invalid.
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    T = samples[0].num_steps
    if any(s.num_steps != T for s in samples):
        raise ValueError("all samples must share the same number of steps T")
    P = max(s.num_players for s in samples)
    B = len(samples)

    pitch_xy = torch.zeros(B, T, P, 2, dtype=torch.float32)
    velocity = torch.zeros(B, T, P, 2, dtype=torch.float32)
    bbox = torch.zeros(B, T, P, 4, dtype=torch.float32)
    roles = torch.zeros(B, T, P, dtype=torch.long)
    teams = torch.zeros(B, P, dtype=torch.long)
    valid_mask = torch.zeros(B, T, P, dtype=torch.bool)
    targets_class = torch.zeros(B, T, P, dtype=torch.long)
    frames = torch.zeros(B, T, dtype=torch.int64)
    left_to_right = torch.zeros(B, T, P, dtype=torch.float32)
    shirt_numbers = torch.full((B, P), -1, dtype=torch.long)
    fps_values: list[float] = []

    has_global = samples[0].global_features is not None
    if has_global:
        F = samples[0].global_features.shape[-1]
        if any(
            s.global_features is None or s.global_features.shape[-1] != F for s in samples
        ):
            raise ValueError("global_features must be present and same dim for all samples")
        global_features: torch.Tensor | None = torch.zeros(B, T, F, dtype=torch.float32)
    else:
        global_features = None

    has_visual = samples[0].visual_features is not None
    if has_visual:
        Fv = samples[0].visual_features.shape[-1]  # type: ignore[union-attr]
        for s in samples:
            if s.visual_features is None or s.visual_features.shape[-1] != Fv:
                raise ValueError(
                    "visual_features must be present and have the same last "
                    "dim for all samples in a batch"
                )
            if s.visual_features.shape[0] != T:
                raise ValueError(
                    "visual_features must share T with the kinematic arrays "
                    f"({s.visual_features.shape[0]} != {T})"
                )
        visual_features: torch.Tensor | None = torch.zeros(
            B, T, P, Fv, dtype=torch.float32
        )
    else:
        visual_features = None

    for b, s in enumerate(samples):
        p = s.num_players
        pitch_xy[b, :, :p] = torch.from_numpy(s.pitch_xy)
        velocity[b, :, :p] = torch.from_numpy(s.velocity)
        bbox[b, :, :p] = torch.from_numpy(s.bbox_xywh)
        roles[b, :, :p] = torch.from_numpy(s.roles.astype(np.int64))
        team_ids = np.where(s.teams < 0, 2, s.teams).astype(np.int64)
        teams[b, :p] = torch.from_numpy(team_ids)
        valid_mask[b, :, :p] = torch.from_numpy(s.valid_mask)
        targets_class[b, :, :p] = torch.from_numpy(s.targets_class.astype(np.int64))
        frames[b] = torch.from_numpy(s.frames.astype(np.int64))
        fps_values.append(float(getattr(s.meta, "fps", 25.0)))
        if has_global and s.global_features is not None:
            global_features[b] = torch.from_numpy(s.global_features.astype(np.float32))  # type: ignore[index]
        if has_visual and s.visual_features is not None:
            visual_features[b, :, :p] = torch.from_numpy(  # type: ignore[index]
                s.visual_features.astype(np.float32)
            )
        if s.left_to_right is not None:
            left_to_right[b, :, :p] = torch.from_numpy(
                s.left_to_right.astype(np.float32)
            )
        if s.shirt_numbers is not None:
            shirt_numbers[b, :p] = torch.from_numpy(
                s.shirt_numbers.astype(np.int64)
            )

    acceleration = _compute_acceleration(velocity)
    # Use the median fps across the batch as the time-feature scale; in
    # practice all samples in one batch share the same fps.
    fps = float(np.median(fps_values)) if fps_values else 25.0
    time_features = _compute_time_features(frames, fps=fps)

    return StackedSampleBatch(
        pitch_xy=pitch_xy,
        velocity=velocity,
        bbox=bbox,
        roles=roles,
        teams=teams,
        valid_mask=valid_mask,
        targets_class=targets_class,
        global_features=global_features,
        acceleration=acceleration,
        time_features=time_features,
        frames=frames,
        visual_features=visual_features,
        left_to_right=left_to_right,
        shirt_numbers=shirt_numbers,
    )
