"""Graph-ablation player-centric action spotting model.

A pure per-player tower — ``embedder -> MS-TCN++ -> head`` — with the
``HGTEncoder`` removed entirely: no edge attention, no zone nodes, no
team/frame pooling, i.e. **no inter-player message passing of any
kind**. Tactical (kinematic) features and per-player visual features
(e.g. frozen DINOv2 embeddings) still flow into the node embedder
exactly as in :class:`pcspot.models.graph_model.PlayerCentricSpottingModel`.

This sibling exists to measure the added value of the graph: train one
run with each model (everything else equal) and compare validation
metrics.

Differences from the graph model:

* No ``HGTEncoder`` / zone nodes / edge features / radius edges — the
  embedder output is fed straight into the temporal stack.
* The radius-edge degree counts (``[n_same_within_r,
  n_opp_within_r]``) are graph-derived and therefore dropped; the
  ``extra_scalars`` branch carries goal distances only (when enabled).
* ``forward`` returns the same output dict shape (``stage_logits``,
  ``logits``, optionally ``stage_confidence`` / ``confidence``) so the
  ``Trainer``, loss function, and validation pipeline are reused
  unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pcspot.data.schema import NUM_PCBAS_CLASSES
from pcspot.models.batch import (
    DEFAULT_NUM_JERSEYS,
    StackedSampleBatch,
    TIME_FEATURE_DIM,
    compute_goal_distances,
)
from pcspot.models.graph import PlayerNodeEmbedder
from pcspot.models.heads import PlayerActionHead
from pcspot.models.mstcn import PlayerMSTCN


class NoGraphSpottingModel(nn.Module):
    """Graph-ablation pipeline: embedder -> MS-TCN++ -> per-stage head.

    Keeps the tactical (kinematic), role/team, jersey, goal-distance,
    and per-player visual branches of :class:`PlayerNodeEmbedder`
    intact, but drops every form of inter-player communication: no
    HGT, no zone nodes, no radius-edge degree counts.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = NUM_PCBAS_CLASSES,
        num_mstcn_stages: int = 3,
        num_mstcn_layers: int = 10,
        global_dim: int = 0,
        visual_dim: int = 0,
        visual_proj_dim: int | None = None,
        with_confidence: bool = True,
        use_acceleration: bool = True,
        use_time_features: bool = True,
        use_jersey: bool = True,
        num_jerseys: int = DEFAULT_NUM_JERSEYS,
        jersey_dim: int | None = None,
        use_goal_distances: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_acceleration = bool(use_acceleration)
        self.use_time_features = bool(use_time_features)
        self.visual_dim = int(visual_dim)
        self.use_jersey = bool(use_jersey)
        self.use_goal_distances = bool(use_goal_distances)
        time_dim = TIME_FEATURE_DIM if use_time_features else 0
        # No degree counts (graph-derived): extra scalars are goal
        # distances only, when enabled.
        self.extra_scalar_dim = 3 if self.use_goal_distances else 0
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

    def _build_extra_scalars(self, batch: StackedSampleBatch) -> torch.Tensor | None:
        """Assemble the embedder's ``extra_scalars`` input or return ``None``.

        Goal distances only — ``[dist_own_goal, dist_opp_goal,
        dist_sideline]``. Unlike the graph model there are no
        radius-edge degree counts to append (no graph to derive them
        from).
        """
        if self.extra_scalar_dim == 0:
            return None
        ltr = batch.left_to_right
        if ltr is None:
            ltr = torch.zeros(
                batch.pitch_xy.shape[:3],
                device=batch.pitch_xy.device,
                dtype=batch.pitch_xy.dtype,
            )
        return compute_goal_distances(batch.pitch_xy, ltr)

    def forward(self, batch: StackedSampleBatch) -> dict[str, torch.Tensor]:
        extra_scalars = self._build_extra_scalars(batch)
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
