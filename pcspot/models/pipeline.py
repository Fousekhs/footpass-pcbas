"""Backward-compatible re-exports for the (now split) model pipeline.

The batch container / feature utilities moved to
:mod:`pcspot.models.batch` and the graph model moved to
:mod:`pcspot.models.graph_model` (its graph-ablation sibling lives in
:mod:`pcspot.models.no_graph_model`). This module re-exports the
original names so existing imports of ``pcspot.models.pipeline`` keep
working unchanged.
"""

from __future__ import annotations

from pcspot.models.batch import (
    DEFAULT_NUM_JERSEYS,
    EXTRA_SCALAR_DIM,
    StackedSampleBatch,
    TIME_FEATURE_DIM,
    TIME_FEATURE_PERIODS_SEC,
    _compute_acceleration,
    _compute_time_features,
    compute_goal_distances,
    stacked_to_batch,
)
from pcspot.models.graph_model import PlayerCentricSpottingModel

__all__ = [
    "DEFAULT_NUM_JERSEYS",
    "EXTRA_SCALAR_DIM",
    "PlayerCentricSpottingModel",
    "StackedSampleBatch",
    "TIME_FEATURE_DIM",
    "TIME_FEATURE_PERIODS_SEC",
    "_compute_acceleration",
    "_compute_time_features",
    "compute_goal_distances",
    "stacked_to_batch",
]
