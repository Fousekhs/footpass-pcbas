"""Evaluation: player-centric NMS, average-mAP, joint correctness."""

from pcspot.eval.nms import (
    Prediction,
    decode_predictions,
    player_centric_nms,
)
from pcspot.eval.metrics import (
    EvalSummary,
    average_map_at_tolerances,
    match_predictions,
    player_identity_accuracy,
)

__all__ = [
    "Prediction",
    "decode_predictions",
    "player_centric_nms",
    "EvalSummary",
    "average_map_at_tolerances",
    "match_predictions",
    "player_identity_accuracy",
]
