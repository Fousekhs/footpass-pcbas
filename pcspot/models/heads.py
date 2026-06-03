"""Prediction heads for player-centric action spotting."""

from __future__ import annotations

import torch
import torch.nn as nn


class PlayerActionHead(nn.Module):
    """Per ``(t, p)`` classification head.

    Outputs raw logits over ``num_classes`` action classes (background
    excluded). Optionally returns a confidence score per ``(t, p)`` for
    CALF-style spotting decoding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        with_confidence: bool = True,
        with_offset: bool = False,
    ) -> None:
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.with_confidence = with_confidence
        self.with_offset = with_offset
        if with_confidence:
            self.conf = nn.Linear(hidden_dim, 1)
        if with_offset:
            self.offset = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"logits": self.cls(h)}
        if self.with_confidence:
            out["confidence"] = self.conf(h).squeeze(-1)
        if self.with_offset:
            out["offset"] = self.offset(h).squeeze(-1)
        return out
