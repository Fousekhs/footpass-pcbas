"""Player-centric spotting metrics.

We expose three primary signals:

- ``average_map_at_tolerances``: SoccerNet-style Average-mAP across a
  list of frame tolerances. Predictions are matched to ground-truth
  events of the same class within the tolerance window. Per-class AP
  is computed over the precision-recall curve obtained by sweeping
  the score threshold.
- ``player_identity_accuracy``: of the predictions that match a
  ground-truth event in time *and* class, what fraction also match
  the responsible player id. This isolates the player-centric signal
  on top of the temporal/class signal.
- ``joint_average_map``: the same Average-mAP but with a stricter
  matching rule that *also* requires matching ``player_id``.

Ground-truth events use the same ``(time, class_id, player_id)``
shape as predictions, except they have no score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from pcspot.data.schema import EventLabel
from pcspot.eval.nms import Prediction


@dataclass
class EvalSummary:
    """Bundle of evaluation numbers for one tolerance setting."""

    tolerance: int
    average_map: float
    average_map_joint: float
    per_class_ap: dict[int, float]
    per_class_ap_joint: dict[int, float]
    player_identity_accuracy: float
    matched_count: int
    gt_count: int


APInterpolation = str  # "11-point" | "101-point" | "continuous"

DEFAULT_AP_INTERPOLATION: APInterpolation = "11-point"
"""Default AP interpolation. Kept at ``"11-point"`` to match the
SoccerNet reference convention; ``"101-point"`` is the COCO convention
and tends to produce slightly higher numbers; ``"continuous"`` evaluates
the precision-recall area exactly (no recall quantization)."""


def _ap_11_point(precisions: np.ndarray, recalls: np.ndarray) -> float:
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        mask = recalls >= r
        ap += float(np.max(precisions[mask])) if mask.any() else 0.0
    return ap / 11.0


def _ap_101_point(precisions: np.ndarray, recalls: np.ndarray) -> float:
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        mask = recalls >= r
        ap += float(np.max(precisions[mask])) if mask.any() else 0.0
    return ap / 101.0


def _ap_continuous(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Exact PR-AUC after enforcing the precision-monotonicity convention."""
    if precisions.size == 0:
        return 0.0
    # Append (0, 0) at the start and (max_recall, 0) at the end so we
    # integrate the staircase area correctly.
    mrec = np.concatenate(([0.0], recalls, [recalls[-1]]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Make precisions monotonically non-increasing (the PASCAL VOC trick).
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Sum widths * heights where recall changes.
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def _ap_from_pr(
    precisions: np.ndarray,
    recalls: np.ndarray,
    *,
    mode: APInterpolation = DEFAULT_AP_INTERPOLATION,
) -> float:
    """AP from a precision-recall curve with configurable interpolation."""
    if mode == "11-point":
        return _ap_11_point(precisions, recalls)
    if mode == "101-point":
        return _ap_101_point(precisions, recalls)
    if mode == "continuous":
        return _ap_continuous(precisions, recalls)
    raise ValueError(
        f"unknown AP interpolation mode {mode!r}; "
        "expected one of '11-point', '101-point', 'continuous'"
    )


def _match_class_only(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match predictions to ground-truth events of the same class.

    Returns ``(scores_sorted, is_tp, num_gt)``. ``is_tp`` indicates
    whether each prediction (after sorting by descending score) was a
    true positive.
    """
    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    n_gt = len(gts)
    matched_gt = [False] * n_gt
    is_tp = np.zeros(len(preds_sorted), dtype=bool)
    for i, pred in enumerate(preds_sorted):
        best_j = -1
        best_dt = tolerance + 1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            if pred.class_id != gt.class_id:
                continue
            dt = abs(pred.time - gt.frame)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            matched_gt[best_j] = True
            is_tp[i] = True
    scores = np.asarray([p.score for p in preds_sorted], dtype=np.float64)
    return scores, is_tp, n_gt


def _match_class_and_player(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    n_gt = len(gts)
    matched_gt = [False] * n_gt
    is_tp = np.zeros(len(preds_sorted), dtype=bool)
    for i, pred in enumerate(preds_sorted):
        best_j = -1
        best_dt = tolerance + 1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            if pred.class_id != gt.class_id:
                continue
            if pred.player_id != gt.player_id:
                continue
            dt = abs(pred.time - gt.frame)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            matched_gt[best_j] = True
            is_tp[i] = True
    scores = np.asarray([p.score for p in preds_sorted], dtype=np.float64)
    return scores, is_tp, n_gt


def _per_class_ap(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerance: int,
    class_ids: Iterable[int],
    matcher,
    *,
    ap_interpolation: APInterpolation = DEFAULT_AP_INTERPOLATION,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in class_ids:
        cls_preds = [p for p in preds if p.class_id == c]
        cls_gts = [g for g in gts if g.class_id == c]
        if not cls_gts:
            out[c] = float("nan")
            continue
        scores, is_tp, n_gt = matcher(cls_preds, cls_gts, tolerance)
        if scores.size == 0:
            out[c] = 0.0
            continue
        tp = is_tp.astype(np.float64)
        fp = 1.0 - tp
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recalls = cum_tp / max(n_gt, 1)
        out[c] = _ap_from_pr(precisions, recalls, mode=ap_interpolation)
    return out


def match_predictions(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerance: int,
    *,
    require_player: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Public matcher used by tests and downstream evaluators."""
    matcher = _match_class_and_player if require_player else _match_class_only
    return matcher(preds, gts, tolerance)


def player_identity_accuracy(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerance: int,
) -> tuple[float, int]:
    """Fraction of class-matched predictions that also match player.

    Returns ``(accuracy, matched_count)``. ``matched_count`` is the
    number of predictions matched to a GT event by class+time only;
    accuracy is undefined when ``matched_count == 0`` and reported as
    0.0 in that case.
    """
    if not preds or not gts:
        return 0.0, 0
    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    matched_gt = [False] * len(gts)
    matched = 0
    correct = 0
    for pred in preds_sorted:
        best_j = -1
        best_dt = tolerance + 1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            if pred.class_id != gt.class_id:
                continue
            dt = abs(pred.time - gt.frame)
            if dt <= tolerance and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            matched_gt[best_j] = True
            matched += 1
            if pred.player_id == gts[best_j].player_id:
                correct += 1
    if matched == 0:
        return 0.0, 0
    return correct / matched, matched


def average_map_at_tolerances(
    preds: list[Prediction],
    gts: list[EventLabel],
    tolerances: list[int],
    class_ids: Iterable[int],
    *,
    ap_interpolation: APInterpolation = DEFAULT_AP_INTERPOLATION,
) -> list[EvalSummary]:
    """Compute Average-mAP and joint Average-mAP across tolerances."""
    summaries: list[EvalSummary] = []
    for tol in tolerances:
        per_class = _per_class_ap(
            preds, gts, tol, class_ids, _match_class_only,
            ap_interpolation=ap_interpolation,
        )
        per_class_joint = _per_class_ap(
            preds, gts, tol, class_ids, _match_class_and_player,
            ap_interpolation=ap_interpolation,
        )
        valid_class = [v for v in per_class.values() if not np.isnan(v)]
        valid_class_joint = [v for v in per_class_joint.values() if not np.isnan(v)]
        avg_map = float(np.mean(valid_class)) if valid_class else 0.0
        avg_map_joint = (
            float(np.mean(valid_class_joint)) if valid_class_joint else 0.0
        )
        acc, matched = player_identity_accuracy(preds, gts, tol)
        summaries.append(
            EvalSummary(
                tolerance=tol,
                average_map=avg_map,
                average_map_joint=avg_map_joint,
                per_class_ap=per_class,
                per_class_ap_joint=per_class_joint,
                player_identity_accuracy=acc,
                matched_count=matched,
                gt_count=len(gts),
            )
        )
    return summaries
