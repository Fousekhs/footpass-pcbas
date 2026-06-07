"""Model components: HGT graph encoder, MS-TCN++ bridge, heads, pipelines.

Three end-to-end model variants are available:

* :class:`pcspot.models.graph_model.PlayerCentricSpottingModel` — the
  main model (embedder -> HGT graph encoder -> MS-TCN++ -> head) with
  both player nodes and zone nodes.
* :class:`pcspot.models.no_zones_model.NoZonesSpottingModel` — the
  player-graph variant (HGT over player nodes only, zone nodes removed),
  used to measure the contribution of zone nodes alone.
* :class:`pcspot.models.no_graph_model.NoGraphSpottingModel` — its
  graph-ablation sibling (embedder -> MS-TCN++ -> head, no inter-player
  message passing) used to measure the added value of the graph.

All three share :mod:`pcspot.models.batch` (``StackedSampleBatch`` /
``stacked_to_batch``) and the same output-dict contract, so they are
interchangeable from the trainer / loss / validation pipeline's
perspective.
"""

from pcspot.models.graph_model import PlayerCentricSpottingModel
from pcspot.models.no_graph_model import NoGraphSpottingModel
from pcspot.models.no_zones_model import NoZonesSpottingModel

__all__ = [
    "NoGraphSpottingModel",
    "NoZonesSpottingModel",
    "PlayerCentricSpottingModel",
]
