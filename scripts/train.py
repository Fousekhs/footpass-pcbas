"""Deprecated shim — use ``scripts/graph/train.py`` directly.

Kept so existing commands and docs that reference ``python
scripts/train.py`` keep working: it delegates to the **graph** model
variant's trainer (:class:`pcspot.models.graph_model.PlayerCentricSpottingModel`,
the project's main model). For the graph-ablation variant, use
``scripts/no_graph/train.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_DIR = SCRIPT_DIR / "graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

from train import main  # noqa: E402  (scripts/graph/train.py via sys.path)

if __name__ == "__main__":
    raise SystemExit(main())
