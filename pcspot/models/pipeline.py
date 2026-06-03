"""End-to-end player-centric action spotting model.

Wires the embedder, HGT encoder, MS-TCN++ temporal bridge, and head
into a single ``nn.Module`` whose ``forward`` consumes a small
``StackedSampleBatch`` produced from ``StackedSample`` instances.

``stacked_to_batch`` also derives acceleration (finite differences of
velocity along T) and match-time positional features (sinusoidal
encodings of the absolute frame index at multiple periods). Both are
optional from the model's perspective: the embedder ignores them unless
``use_acceleration`` / ``time_dim`` are set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from pcspot.data.schema import NUM_PCBAS_CLASSES, StackedSample
from pcspot.models.graph import (
    DEFAULT_RADIUS,
    EDGE_FEATURE_DIM,
    HGTEncoder,
    PLAYER_EDGE_TYPES,
    PlayerNodeEmbedder,
    ZONE_GRID,
    build_player_graphs,
    build_zone_assignment,
    zone_adjacency,
)
from pcspot.models.heads import PlayerActionHead
from pcspot.models.mstcn import PlayerMSTCN


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


class PlayerCentricSpottingModel(nn.Module):
    """Full pipeline: embedder -> HGT -> MS-TCN++ -> per-stage head.

    Optional per-player visual features (e.g. frozen DINOv2 ViT-S/14
    embeddings of padded player crops) are fused into the node embedder
    when ``visual_dim > 0``. The embedder applies a dedicated
    LayerNorm + Linear projection (``visual_proj_dim``, defaulting to
    ``hidden_dim // 2``) before concatenating the projected visual
    embedding with the kinematic / role / team / time / global features
    and feeding the combined vector through the hidden projection.

    Heterogeneous zone nodes can be enabled with ``use_zone_nodes``
    (default on). They give the block a notion of physical congestion
    and local numerical superiority that softmax / mean aggregators
    structurally cannot represent (they are count-invariant). The pitch
    is binned into a ``zone_grid`` of ``(Gx, Gy)`` cells and players
    are soft-splatted into the 4 surrounding cell centers; zone nodes
    SUM their occupants per team, diffuse across their grid neighbors,
    and report back into each player's update.

    Jersey numbers and a small bundle of derived scalars (goal
    distances + radius-edge degree counts) are fed through dedicated
    branches in the embedder, controlled by ``use_jersey``,
    ``use_goal_distances`` and ``use_radius_edges`` / ``radius``.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = NUM_PCBAS_CLASSES,
        num_hgt_layers: int = 2,
        num_heads: int = 4,
        num_mstcn_stages: int = 3,
        num_mstcn_layers: int = 10,
        knn: int = 4,
        edge_types: Sequence[str] = PLAYER_EDGE_TYPES,
        global_dim: int = 0,
        visual_dim: int = 0,
        visual_proj_dim: int | None = None,
        with_confidence: bool = True,
        use_acceleration: bool = True,
        use_time_features: bool = True,
        use_edge_features: bool = True,
        use_zone_nodes: bool = True,
        zone_grid: tuple[int, int] = ZONE_GRID,
        use_jersey: bool = True,
        num_jerseys: int = DEFAULT_NUM_JERSEYS,
        jersey_dim: int | None = None,
        use_goal_distances: bool = True,
        use_radius_edges: bool = True,
        radius: float = DEFAULT_RADIUS,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.knn = knn
        self.use_acceleration = bool(use_acceleration)
        self.use_time_features = bool(use_time_features)
        self.use_edge_features = bool(use_edge_features)
        self.visual_dim = int(visual_dim)
        self.use_zone_nodes = bool(use_zone_nodes)
        self.zone_grid = (int(zone_grid[0]), int(zone_grid[1]))
        self.num_zones = self.zone_grid[0] * self.zone_grid[1] if self.use_zone_nodes else 0
        self.use_jersey = bool(use_jersey)
        self.use_goal_distances = bool(use_goal_distances)
        self.use_radius_edges = bool(use_radius_edges)
        self.radius = float(radius)
        time_dim = TIME_FEATURE_DIM if use_time_features else 0
        edge_feat_dim = EDGE_FEATURE_DIM if use_edge_features else 0
        # Extra scalars: 3 goal-distance channels (when on) + 2 degree
        # counts (when radius edges are on). Both contribute via the
        # same branch so the LayerNorm sees them together.
        extra_scalar_dim = 0
        if self.use_goal_distances:
            extra_scalar_dim += 3
        if self.use_radius_edges:
            extra_scalar_dim += 2
        self.extra_scalar_dim = extra_scalar_dim
        self.embedder = PlayerNodeEmbedder(
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            time_dim=time_dim,
            use_acceleration=use_acceleration,
            visual_dim=visual_dim,
            visual_proj_dim=visual_proj_dim,
            num_jerseys=num_jerseys if self.use_jersey else 0,
            jersey_dim=jersey_dim,
            extra_scalar_dim=self.extra_scalar_dim,
        )
        self.hgt = HGTEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_hgt_layers,
            num_heads=num_heads,
            edge_types=edge_types,
            edge_feat_dim=edge_feat_dim,
            num_zones=self.num_zones,
        )
        if self.use_zone_nodes:
            # Buffer (not parameter): row-normalized 8-neighborhood
            # over the grid. Moves with the model via .to(device).
            self.register_buffer(
                "_zone_adj",
                zone_adjacency(self.zone_grid),
                persistent=False,
            )
        self.temporal = PlayerMSTCN(
            hidden_dim=hidden_dim,
            num_stages=num_mstcn_stages,
            num_layers_per_stage=num_mstcn_layers,
        )
        self.head = PlayerActionHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            with_confidence=with_confidence,
        )

    def _build_extra_scalars(
        self,
        batch: StackedSampleBatch,
        degree_counts: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Assemble the embedder's ``extra_scalars`` input or return ``None``.

        Order matches :data:`EXTRA_SCALAR_DIM` documentation:
        ``[dist_own_goal, dist_opp_goal, dist_sideline,
        n_same_within_r, n_opp_within_r]``. When a sub-branch is
        disabled its channels are omitted (so the embedder is
        configured with the matching width).
        """
        if self.extra_scalar_dim == 0:
            return None
        parts: list[torch.Tensor] = []
        if self.use_goal_distances:
            ltr = batch.left_to_right
            if ltr is None:
                ltr = torch.zeros(
                    batch.pitch_xy.shape[:3],
                    device=batch.pitch_xy.device,
                    dtype=batch.pitch_xy.dtype,
                )
            parts.append(compute_goal_distances(batch.pitch_xy, ltr))
        if self.use_radius_edges:
            if degree_counts is None:
                # build_player_graphs was called without
                # compute_degree_counts; fall back to zeros so the
                # branch shape still matches.
                degree_counts = torch.zeros(
                    (*batch.pitch_xy.shape[:3], 2),
                    device=batch.pitch_xy.device,
                    dtype=batch.pitch_xy.dtype,
                )
            parts.append(degree_counts)
        return torch.cat(parts, dim=-1)

    def forward(self, batch: StackedSampleBatch) -> dict[str, torch.Tensor]:
        edges = build_player_graphs(
            teams=batch.teams,
            pitch_xy=batch.pitch_xy,
            valid=batch.valid_mask,
            knn=self.knn,
            velocity=batch.velocity if self.use_edge_features else None,
            compute_edge_features_flag=self.use_edge_features,
            use_radius=self.use_radius_edges,
            radius=self.radius,
            compute_degree_counts=self.use_radius_edges,
        )
        if self.use_zone_nodes:
            edges.zone_assignment = build_zone_assignment(
                pitch_xy=batch.pitch_xy,
                valid=batch.valid_mask,
                grid=self.zone_grid,
            )
            edges.zone_adjacency = self._zone_adj
        extra_scalars = self._build_extra_scalars(batch, edges.degree_counts)
        h = self.embedder(
            pitch_xy=batch.pitch_xy,
            velocity=batch.velocity,
            bbox=batch.bbox,
            roles=batch.roles,
            teams=batch.teams,
            acceleration=batch.acceleration if self.use_acceleration else None,
            global_features=batch.global_features,
            time_features=batch.time_features if self.use_time_features else None,
            visual_features=batch.visual_features if self.visual_dim > 0 else None,
            shirt_numbers=batch.shirt_numbers if self.use_jersey else None,
            extra_scalars=extra_scalars,
        )
        h = self.hgt(h, edges, teams=batch.teams)
        outs = self.temporal(h, valid=batch.valid_mask)
        # Run the head on every stage's output (deep supervision a-la MS-TCN).
        stage_logits = []
        stage_conf = []
        for stage_h in outs.stages:
            head_out = self.head(stage_h)
            stage_logits.append(head_out["logits"])
            if "confidence" in head_out:
                stage_conf.append(head_out["confidence"])
        out: dict[str, torch.Tensor] = {
            "stage_logits": torch.stack(stage_logits, dim=0),  # (S, B, T, P, C)
            "logits": stage_logits[-1],
        }
        if stage_conf:
            out["stage_confidence"] = torch.stack(stage_conf, dim=0)  # (S, B, T, P)
            out["confidence"] = stage_conf[-1]
        return out
