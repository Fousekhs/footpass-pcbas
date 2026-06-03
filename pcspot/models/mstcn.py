"""MS-TCN++ adapted to per-player temporal sequences.

Reference: Li et al. "MS-TCN++: Multi-Stage Temporal Convolutional
Network for Action Segmentation", TPAMI 2020. The original network
operates on a single ``(F, T)`` per video; here we run it independently
per player by flattening the player axis into the batch, which matches
the plan's "(B * P) x D x T" idea.

The first stage uses *dual-dilated* layers (dilations grow then shrink
within each block) and the refinement stages use the original
single-dilated MS-TCN layers. A boolean ``time_mask`` can be passed
through the forward pass; while the convolutions themselves are
agnostic to the mask, downstream loss code uses it to ignore padded
timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DualDilatedLayer(nn.Module):
    """Dual-dilated residual block from MS-TCN++.

    Combines features from a small dilation (2**i) and a large dilation
    (2**(L-1-i)) in parallel, fuses them, then adds a 1x1 projection
    plus residual.
    """

    def __init__(self, in_dim: int, hidden_dim: int, small_dilation: int, large_dilation: int) -> None:
        super().__init__()
        self.conv_small = nn.Conv1d(
            in_dim, hidden_dim, kernel_size=3, padding=small_dilation, dilation=small_dilation
        )
        self.conv_large = nn.Conv1d(
            in_dim, hidden_dim, kernel_size=3, padding=large_dilation, dilation=large_dilation
        )
        self.fuse = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        self.residual = (
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1) if in_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.conv_small(x))
        b = F.relu(self.conv_large(x))
        out = self.fuse(torch.cat([a, b], dim=1))
        out = self.dropout(out)
        return out + self.residual(x)


class _SingleDilatedLayer(nn.Module):
    """Original MS-TCN dilated residual block (used in refinement stages)."""

    def __init__(self, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv(x))
        out = self.proj(out)
        out = self.dropout(out)
        return out + x


class _DualDilatedStage(nn.Module):
    """First stage: ``num_layers`` dual-dilated layers stacked."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 10) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            small = 2 ** i
            large = 2 ** (num_layers - 1 - i)
            layers.append(
                _DualDilatedLayer(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    small_dilation=small,
                    large_dilation=large,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _RefinementStage(nn.Module):
    """Refinement stage: ``num_layers`` single-dilated layers."""

    def __init__(self, num_classes_proxy_dim: int, hidden_dim: int, num_layers: int = 10) -> None:
        super().__init__()
        # Refinement input is the previous stage's logits-space embedding,
        # projected back to hidden_dim.
        self.in_proj = nn.Conv1d(num_classes_proxy_dim, hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList(
            [_SingleDilatedLayer(hidden_dim, dilation=2 ** i) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MSTCNOutputs:
    """Per-stage outputs from the MS-TCN++ stack."""

    stages: list[torch.Tensor]  # each (B*P, hidden_dim, T)


class PlayerMSTCN(nn.Module):
    """MS-TCN++ adapted to per-player temporal sequences.

    Forward expects ``(B, T, P, D)`` and returns the same shape per
    stage, plus a final ``(B, T, P, D)`` that callers feed into the
    classification head.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_stages: int = 3,
        num_layers_per_stage: int = 10,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        self.hidden_dim = hidden_dim
        self.first_stage = _DualDilatedStage(
            in_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers_per_stage
        )
        self.refinements = nn.ModuleList(
            [
                _RefinementStage(
                    num_classes_proxy_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers_per_stage,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,  # (B, T, P, D)
        valid: torch.Tensor | None = None,  # (B, T, P) bool, optional
    ) -> MSTCNOutputs:
        if h.ndim != 4:
            raise ValueError(f"expected (B, T, P, D), got shape {tuple(h.shape)}")
        B, T, P, D = h.shape
        if D != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: model={self.hidden_dim} input={D}")

        # Flatten (B, P) into batch and put time last for Conv1d.
        x = h.permute(0, 2, 3, 1).reshape(B * P, D, T)

        if valid is not None:
            v = valid.permute(0, 2, 1).reshape(B * P, 1, T).float()
            x = x * v
        else:
            v = None

        outs: list[torch.Tensor] = []
        out = self.first_stage(x)
        if v is not None:
            out = out * v
        outs.append(out)

        for refine in self.refinements:
            out = refine(out)
            if v is not None:
                out = out * v
            outs.append(out)

        # Reshape back to (B, T, P, D).
        outs_btpd = [o.reshape(B, P, D, T).permute(0, 3, 1, 2).contiguous() for o in outs]
        return MSTCNOutputs(stages=outs_btpd)
