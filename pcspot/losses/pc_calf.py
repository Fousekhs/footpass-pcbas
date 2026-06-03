"""Player-aware CALF loss with objectness supervision and stage weighting.

This module computes the supervision signal for the player-centric
spotting model. It bundles three terms:

1. **Per-class CALF BCE** on ``(B, T, P, C)`` logits.
2. **Objectness BCE** on ``(B, T, P)`` confidence logits, using the
   max-over-classes targets from ``pcspot.data.targets.build_objectness_targets``.
   The confidence/objectness head was previously trained implicitly (i.e.
   never), so without this term the decoder multiplies class probabilities
   by random sigmoid noise.
3. **T-MSE smoothing** (MS-TCN++ Eq. 4) on the class logits, masked by the
   *effective* weight mask rather than only ``valid_mask`` so it does not
   pull predictions toward smoothness inside CALF uncertain-before zones.

Each term is averaged across MS-TCN++ stages. Stage weighting is
configurable -- default is geometric (later stages weighted higher), which
matches MS-TCN++'s observation that the last stage is what really matters
at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted BCE-with-logits reduced to a scalar."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = bce * weight
    denom = weight.sum().clamp(min=1.0)
    return weighted.sum() / denom


def _tmse_smoothing(
    logits: torch.Tensor,
    pair_weight: torch.Tensor,
    threshold: float = 4.0,
) -> torch.Tensor:
    """T-MSE smoothing penalty masked by a precomputed pair weight.

    ``pair_weight`` has shape ``(B, T-1, P, C)`` (or broadcastable to it)
    and is 0 where the smoothing should be ignored. Callers compose it
    from the effective CALF weight mask.
    """
    log_probs = F.logsigmoid(logits)  # (B, T, P, C)
    diff = log_probs[:, 1:] - log_probs[:, :-1]
    diff = torch.clamp(torch.abs(diff), max=threshold) ** 2
    denom = pair_weight.sum().clamp(min=1.0)
    return (diff * pair_weight).sum() / denom


def _resolve_stage_weights(
    num_stages: int,
    stage_weights: Sequence[float] | str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(stage_weights, str):
        if stage_weights == "uniform":
            w = [1.0] * num_stages
        elif stage_weights == "geometric":
            w = [2.0 ** s for s in range(num_stages)]
        else:
            raise ValueError(
                f"Unknown stage_weights mode {stage_weights!r}; "
                "use 'uniform', 'geometric' or a sequence of floats."
            )
    elif stage_weights is None:
        w = [2.0 ** s for s in range(num_stages)]
    else:
        w = list(stage_weights)
        if len(w) != num_stages:
            raise ValueError(
                f"stage_weights length {len(w)} != num_stages {num_stages}"
            )
        if any(x < 0 for x in w):
            raise ValueError("stage_weights entries must be non-negative")
    t = torch.tensor(w, device=device, dtype=dtype)
    total = t.sum().clamp(min=1e-9)
    return t / total


@dataclass
class CalfLossOutputs:
    """All loss components plus their weighted total."""

    total: torch.Tensor
    bce: torch.Tensor
    tmse: torch.Tensor
    objectness: torch.Tensor


def pc_calf_loss(
    stage_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    tmse_lambda: float = 0.15,
    stage_weights: Sequence[float] | str | None = None,
    stage_objectness_logits: torch.Tensor | None = None,
    objectness_targets: torch.Tensor | None = None,
    objectness_weights: torch.Tensor | None = None,
    objectness_lambda: float = 1.0,
) -> CalfLossOutputs:
    """Compute the player-aware CALF loss across MS-TCN++ stages.

    Args:
        stage_logits: ``(S, B, T, P, C)`` per-class logits.
        targets: ``(B, T, P, C)`` CALF segmentation targets in [0, 1].
        weights: ``(B, T, P, C)`` per-element weights (>= 0).
        valid: ``(B, T, P)`` bool valid mask.
        tmse_lambda: weight on the T-MSE smoothing penalty.
        stage_weights: ``"uniform"``, ``"geometric"``, an explicit sequence,
            or ``None`` (defaults to geometric).
        stage_objectness_logits: optional ``(S, B, T, P)`` confidence logits;
            when provided together with ``objectness_targets``, an objectness
            BCE term is added.
        objectness_targets: ``(B, T, P)`` float32 objectness targets.
        objectness_weights: ``(B, T, P)`` float32 objectness weights.
        objectness_lambda: weight on the objectness BCE term.
    """
    if stage_logits.ndim != 5:
        raise ValueError(
            f"stage_logits must have shape (S, B, T, P, C); got {tuple(stage_logits.shape)}"
        )
    S = stage_logits.shape[0]
    device = stage_logits.device
    dtype = stage_logits.dtype
    stage_w = _resolve_stage_weights(S, stage_weights, device, dtype)

    valid_f = valid.unsqueeze(-1).to(dtype)
    full_weight = weights.to(dtype) * valid_f
    pair_weight = torch.minimum(full_weight[:, 1:], full_weight[:, :-1])

    has_obj = (
        stage_objectness_logits is not None
        and objectness_targets is not None
        and objectness_weights is not None
    )
    if has_obj:
        if stage_objectness_logits.shape[0] != S:
            raise ValueError(
                f"stage_objectness_logits stage count {stage_objectness_logits.shape[0]} "
                f"!= class stage count {S}"
            )
        obj_full_w = objectness_weights.to(dtype) * valid.to(dtype)
    else:
        obj_full_w = None  # silence type checker; unused below

    bce_terms: list[torch.Tensor] = []
    tmse_terms: list[torch.Tensor] = []
    obj_terms: list[torch.Tensor] = []
    for s in range(S):
        logits_s = stage_logits[s]
        bce_terms.append(_masked_bce(logits_s, targets, full_weight))
        if tmse_lambda > 0.0:
            tmse_terms.append(_tmse_smoothing(logits_s, pair_weight))
        if has_obj:
            obj_logits = stage_objectness_logits[s]
            obj_terms.append(
                _masked_bce(obj_logits, objectness_targets, obj_full_w)
            )

    def _weighted_mean(terms: list[torch.Tensor]) -> torch.Tensor:
        if not terms:
            return stage_logits.new_zeros(())
        stacked = torch.stack(terms)
        return (stacked * stage_w).sum()

    bce = _weighted_mean(bce_terms)
    tmse = _weighted_mean(tmse_terms) if tmse_terms else stage_logits.new_zeros(())
    obj = _weighted_mean(obj_terms) if obj_terms else stage_logits.new_zeros(())
    total = bce + tmse_lambda * tmse + objectness_lambda * obj
    return CalfLossOutputs(total=total, bce=bce, tmse=tmse, objectness=obj)


class PlayerAwareCalfLoss(nn.Module):
    """``nn.Module`` wrapper around ``pc_calf_loss`` for trainer use."""

    def __init__(
        self,
        tmse_lambda: float = 0.15,
        objectness_lambda: float = 1.0,
        stage_weights: Sequence[float] | str | None = None,
    ) -> None:
        super().__init__()
        self.tmse_lambda = float(tmse_lambda)
        self.objectness_lambda = float(objectness_lambda)
        self.stage_weights = stage_weights

    def forward(
        self,
        stage_logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        *,
        stage_objectness_logits: torch.Tensor | None = None,
        objectness_targets: torch.Tensor | None = None,
        objectness_weights: torch.Tensor | None = None,
    ) -> CalfLossOutputs:
        return pc_calf_loss(
            stage_logits,
            targets,
            weights,
            valid,
            tmse_lambda=self.tmse_lambda,
            stage_weights=self.stage_weights,
            stage_objectness_logits=stage_objectness_logits,
            objectness_targets=objectness_targets,
            objectness_weights=objectness_weights,
            objectness_lambda=self.objectness_lambda,
        )
