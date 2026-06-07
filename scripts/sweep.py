"""Deprecated shim — use ``scripts/graph/sweep.py`` directly.

Kept so existing commands and docs that reference ``python
scripts/sweep.py`` keep working: it delegates to the **graph** model
variant's sweep agent. For the graph-ablation variant, use
``scripts/no_graph/sweep.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_DIR = SCRIPT_DIR / "graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

from sweep import main  # noqa: E402  (scripts/graph/sweep.py via sys.path)

if __name__ == "__main__":
    raise SystemExit(main())
