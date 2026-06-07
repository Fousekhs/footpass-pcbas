"""Player-graph (no zone nodes) action spotting model.

A middle ground between the two existing variants: it keeps the HGT
graph over **player nodes** — player-player edges (including the
variable-degree ``radius`` edge type) and the matching degree counts —
but removes the heterogeneous **zone nodes** entirely.

* :class:`pcspot.models.graph_model.PlayerCentricSpottingModel` — full
  model (player nodes + zone nodes).
* :class:`pcspot.models.no_zones_model.NoZonesSpottingModel` — *this*
  variant (player nodes only).
* :class:`pcspot.models.no_graph_model.NoGraphSpottingModel` — drops the
  HGT entirely (no inter-player message passing).

Training one run of each isolates the contribution of zone nodes alone
(the full model minus this variant). Because zone construction in
``PlayerCentricSpottingModel`` is already fully gated by
``use_zone_nodes``, this variant is a thin subclass that hard-wires
``use_zone_nodes=False`` and drops the two zone-only ``__init__`` kwargs
(``use_zone_nodes`` / ``zone_grid``); ``forward`` and the output-dict
contract are inherited unchanged, so the ``Trainer``, loss, and
validation pipeline are reused as-is.
"""

from __future__ import annotations

from typing import Sequence

from pcspot.data.schema import NUM_PCBAS_CLASSES
from pcspot.models.batch import DEFAULT_NUM_JERSEYS
from pcspot.models.graph import DEFAULT_RADIUS, PLAYER_EDGE_TYPES
from pcspot.models.graph_model import PlayerCentricSpottingModel


class NoZonesSpottingModel(PlayerCentricSpottingModel):
    """Player-graph pipeline: embedder -> HGT (player nodes only) -> MS-TCN++ -> head.

    Identical to :class:`PlayerCentricSpottingModel` but with the zone
    nodes removed. The HGT, player-player edges (including ``radius``
    edges) and the radius-edge degree counts are all retained, so the
    only difference from the full model is the absence of the
    count-aware zone view. With ``use_zone_nodes`` forced off, the
    instance has ``num_zones == 0``, no ``_zone_adj`` buffer, and the
    ``HGTEncoder`` is built with ``num_zones=0`` — but ``self.hgt`` is
    still present (this is *not* the no-graph ablation).
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
        use_jersey: bool = True,
        num_jerseys: int = DEFAULT_NUM_JERSEYS,
        jersey_dim: int | None = None,
        use_goal_distances: bool = True,
        use_radius_edges: bool = True,
        radius: float = DEFAULT_RADIUS,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_hgt_layers=num_hgt_layers,
            num_heads=num_heads,
            num_mstcn_stages=num_mstcn_stages,
            num_mstcn_layers=num_mstcn_layers,
            knn=knn,
            edge_types=edge_types,
            global_dim=global_dim,
            visual_dim=visual_dim,
            visual_proj_dim=visual_proj_dim,
            with_confidence=with_confidence,
            use_acceleration=use_acceleration,
            use_time_features=use_time_features,
            use_edge_features=use_edge_features,
            use_zone_nodes=False,  # the ablation: player nodes only
            use_jersey=use_jersey,
            num_jerseys=num_jerseys,
            jersey_dim=jersey_dim,
            use_goal_distances=use_goal_distances,
            use_radius_edges=use_radius_edges,
            radius=radius,
        )
