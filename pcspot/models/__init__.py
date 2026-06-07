"""Model components: HGT graph encoder, MS-TCN++ bridge, heads, pipelines.

Two end-to-end model variants are available:

* :class:`pcspot.models.graph_model.PlayerCentricSpottingModel` — the
  main model (embedder -> HGT graph encoder -> MS-TCN++ -> head).
* :class:`pcspot.models.no_graph_model.NoGraphSpottingModel` — its
  graph-ablation sibling (embedder -> MS-TCN++ -> head, no inter-player
  message passing) used to measure the added value of the graph.

Both share :mod:`pcspot.models.batch` (``StackedSampleBatch`` /
``stacked_to_batch``) and the same output-dict contract, so they are
interchangeable from the trainer / loss / validation pipeline's
perspective.
"""

from pcspot.models.graph_model import PlayerCentricSpottingModel
from pcspot.models.no_graph_model import NoGraphSpottingModel

__all__ = [
    "NoGraphSpottingModel",
    "PlayerCentricSpottingModel",
]
