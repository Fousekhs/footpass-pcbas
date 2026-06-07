"""Graph-based player-centric action spotting model (the "main" model).

Wires the embedder, HGT graph encoder, MS-TCN++ temporal bridge, and
head into a single ``nn.Module`` whose ``forward`` consumes a
``StackedSampleBatch`` produced by ``stacked_to_batch``.

See ``pcspot.models.no_graph_model.NoGraphSpottingModel`` for the
graph-ablation sibling that drops the ``HGTEncoder`` while keeping the
same per-player tactical + visual inputs and output contract.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from pcspot.data.schema import NUM_PCBAS_CLASSES
from pcspot.models.batch import StackedSampleBatch, compute_goal_distances
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

from pcspot.models.batch import DEFAULT_NUM_JERSEYS, TIME_FEATURE_DIM


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
