"""W&B sweep agent entry point for the no_zones model variant.

Usage — two modes:

1. **Create + run** (sweep.py creates the sweep and immediately starts an agent):

    python scripts/no_zones/sweep.py \\
        --sweep-config configs/sweeps/no_zones.yaml \\
        --splits data/splits.json \\
        --output-dir checkpoints/sweeps/no_zones \\
        --device cuda:0 --epochs 10 --count 20

2. **Agent-only** (sweep was already created with ``wandb sweep``):

    python scripts/no_zones/sweep.py \\
        --sweep-id <entity/project/sweep_id> \\
        --splits data/splits.json \\
        --output-dir checkpoints/sweeps/no_zones \\
        --device cuda:0 --epochs 10 --count 20

Fixed args (paths, device, epochs, visual cache, validation settings) are
passed here on the command line and stay constant across all trials. Per-trial
hyperparameters (learning_rate, hidden_dim, …) are injected by the W&B agent
via ``wandb.config`` and applied through ``scripts.no_zones.train._build_argparser``'s
``defaults`` dict, which is keyed by argparse dest names (underscores).

Each trial writes its checkpoints to ``<output_dir>/<run_id>/`` so concurrent
agents never collide. Shared sweep-agent machinery lives in
:mod:`pcspot.train.sweep_common`; this script only wires it to the no_zones
variant's argparser/run pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make the sibling train.py importable without a scripts/ package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcspot.train import sweep_common
from train import _build_argparser, run as _run_training  # noqa: E402


def main() -> int:
    return sweep_common.main(
        description=__doc__.splitlines()[0],
        build_argparser=_build_argparser,
        run_training=_run_training,
    )


if __name__ == "__main__":
    raise SystemExit(main())
