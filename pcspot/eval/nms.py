"""Player-centric prediction decoding and non-maximum suppression.

Predictions live in a four-tuple space ``(time, class, player_id,
score)``. Two NMS modes are supported:

- ``"per_player_class"`` (default, player-centric): suppress overlapping
  predictions per ``(class, player_id)`` pair. This keeps independent
  events for different players or different classes intact.
- ``"per_player"`` (cross-class): within a time radius and per
  ``player_id``, keep only the highest-scoring prediction across all
  classes. Useful when downstream consumers want at most one event per
  player per moment (e.g. a player rarely performs two actions
  simultaneously, so a "Pass" + "Drive" prediction at the same frame is
  likely a class ambiguity that should be collapsed).
- ``"per_class"`` (back-compat with the classic SoccerNet NMS): suppress
  overlapping predictions per class regardless of player.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Prediction:
    """One spotting prediction.

    Attributes:
        time: frame index (relative to the start of the window or
            absolute, as long as ground-truth uses the same convention).
        class_id: 1-based class id matching ``PCBAS_CLASS_NAMES``.
        player_id: predicted player who performed the action.
        score: classification * confidence score in ``[0, 1]``.
    """

    time: int
    class_id: int
    player_id: int
    score: float


def _local_maxima_1d(scores: np.ndarray) -> np.ndarray:
    """Indices where ``scores[t]`` is a strict local maximum along the time axis."""
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if scores.size == 1:
        return np.array([0], dtype=np.int64)
    out = []
    for t in range(scores.size):
        left_ok = t == 0 or scores[t] > scores[t - 1]
        right_ok = t == scores.size - 1 or scores[t] >= scores[t + 1]
        if left_ok and right_ok:
            out.append(t)
    return np.asarray(out, dtype=np.int64)


def decode_predictions(
    logits: torch.Tensor,  # (T, P, C)
    confidence: torch.Tensor | None,  # (T, P) or None
    valid_mask: np.ndarray,  # (T, P) bool
    player_ids: np.ndarray,  # (P,) int
    *,
    score_threshold: float = 0.0,
    apply_sigmoid: bool = True,
) -> list[Prediction]:
    """Turn per-frame, per-player logits into local-maximum predictions.

    The decoder picks, for each ``(class, player)`` pair independently,
    the local maxima along the time axis whose score exceeds
    ``score_threshold``. Confidence (if provided) multiplies the
    sigmoid class probability.
    """
    if logits.ndim != 3:
        raise ValueError("logits must be (T, P, C)")
    T, P, C = logits.shape
    if apply_sigmoid:
        probs = torch.sigmoid(logits)
    else:
        probs = logits
    if confidence is not None:
        if confidence.shape != (T, P):
            raise ValueError("confidence must be (T, P)")
        if apply_sigmoid:
            conf = torch.sigmoid(confidence)
        else:
            conf = confidence
        probs = probs * conf.unsqueeze(-1)
    probs_np = probs.detach().cpu().numpy()

    out: list[Prediction] = []
    for p in range(P):
        if not valid_mask[:, p].any():
            continue
        for c in range(C):
            scores = probs_np[:, p, c]
            scores = np.where(valid_mask[:, p], scores, 0.0)
            peaks = _local_maxima_1d(scores)
            for t in peaks:
                s = float(scores[t])
                if s < score_threshold:
                    continue
                out.append(
                    Prediction(
                        time=int(t),
                        class_id=int(c) + 1,
                        player_id=int(player_ids[p]),
                        score=s,
                    )
                )
    return out


NMSMode = str  # "per_player_class" | "per_player" | "per_class"


def player_centric_nms(
    predictions: list[Prediction],
    window_radius: int,
    *,
    mode: NMSMode = "per_player_class",
) -> list[Prediction]:
    """Non-maximum suppression with selectable grouping.

    Args:
        predictions: input predictions.
        window_radius: time radius (in frames) for the suppression check.
        mode: see module docstring.
    """
    if window_radius < 0:
        raise ValueError("window_radius must be >= 0")
    if mode not in ("per_player_class", "per_player", "per_class"):
        raise ValueError(
            f"unknown NMS mode {mode!r}; expected one of "
            "'per_player_class', 'per_player', 'per_class'"
        )

    def key_fn(p: Prediction) -> tuple:
        if mode == "per_player_class":
            return (p.class_id, p.player_id)
        if mode == "per_player":
            return (p.player_id,)
        return (p.class_id,)

    by_key: dict[tuple, list[Prediction]] = {}
    for pred in predictions:
        by_key.setdefault(key_fn(pred), []).append(pred)

    kept: list[Prediction] = []
    for preds in by_key.values():
        preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
        used: list[Prediction] = []
        for pred in preds_sorted:
            if any(abs(pred.time - q.time) <= window_radius for q in used):
                continue
            used.append(pred)
        kept.extend(used)
    return kept
