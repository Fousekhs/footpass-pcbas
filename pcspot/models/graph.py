"""Heterogeneous graph encoder for per-frame multi-agent context.

The plan calls for a Heterogeneous Graph Transformer over node types
``player``, ``team``, ``frame`` (and optionally ``ball`` once tracking
is available) with edge types ``same_team``, ``opponent``, ``near``,
``has_player``, ``in_frame``.

PyTorch Geometric's ``HGTConv`` is the documented production target,
but this project's CPU-only venv does not ship ``torch_geometric``. To
keep the package runnable today, this module implements a faithful
HGT-style message-passing block in pure PyTorch:

- Per-edge-type multi-head attention with type-specific Q/K/V
  projections (Hu et al. 2020).
- A learnable relation prior ``mu`` per edge type (the HGT relation
  gain), broadcast into attention scores.
- Auxiliary node types ``team`` and ``frame`` are realised as pooled
  prototypes that broadcast a message back to player nodes.

The whole stack works on dense ``(B, T, P, D)`` tensors plus boolean
masks. ``P`` is at most ~22 in soccer, so dense attention is cheap
and avoids ragged-graph plumbing.

If/when ``torch_geometric`` becomes available, ``PlayerHGTBlock`` can
be swapped for ``torch_geometric.nn.HGTConv`` while keeping the same
edge-type metadata; the design doc tracks that swap-out plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PLAYER_EDGE_TYPES: tuple[str, ...] = ("self", "same_team", "opponent", "near", "radius")


DEFAULT_RADIUS: float = 0.15
"""Default pitch distance (normalized units) for the ``radius`` edge type
and for the per-player degree-count features.

The ``radius`` edge complements ``near`` (which is fixed-k kNN and so
censors any cardinality past ``k``) by giving each player a variable-
degree set of neighbors. Per-player ``[n_same_within_r, n_opp_within_r]``
counts are returned alongside the masks so the embedder can feed them
directly into the player feature vector.
"""


ZONE_GRID: tuple[int, int] = (6, 4)
"""Default pitch grid for zone nodes: ``(Gx, Gy)``.

The pitch is normalized to ``[0, 1] x [0, 1]``. A ``6 x 4`` grid gives
24 zone nodes which is robust for counting (each zone sees several
players on average) without being so fine that cells become noisy.

Zones are indexed in row-major order: ``z = j * Gx + i`` for cell
``(i, j)`` with ``i in [0, Gx)`` and ``j in [0, Gy)``. The same
convention is used by ``build_zone_assignment`` and ``zone_adjacency``.
"""


def build_zone_assignment(
    pitch_xy: torch.Tensor,  # (B, T, P, 2)
    valid: torch.Tensor,  # (B, T, P) bool
    grid: tuple[int, int] = ZONE_GRID,
) -> torch.Tensor:
    """Soft (bilinear) splat of players into pitch zones.

    Returns ``A`` of shape ``(B, T, P, Z)`` where ``Z = Gx * Gy``. For
    every valid player, ``A[b, t, p, :]`` holds bilinear weights that
    sum to 1 across the 4 cell centers surrounding the player's pitch
    position. Padded players are all-zero.

    Splatting is done in **physical** pitch coordinates so co-located
    players (regardless of team or attacking direction) end up in the
    same zone(s); this is what makes per-team zone counts meaningful for
    detecting physical congestion. Attacking-direction canonicalization
    is handled downstream by callers that need it (e.g. goal-distance
    scalars), not at the splat.

    Cell ``(i, j)`` has its center at
    ``((i + 0.5) / Gx, (j + 0.5) / Gy)``. Players whose normalized
    position falls outside ``[0, 1]^2`` have their bilinear contributions
    clamped onto the boundary cell; weights still sum to 1.
    """
    if pitch_xy.shape[-1] != 2:
        raise ValueError(
            f"pitch_xy last dim must be 2, got {pitch_xy.shape[-1]}"
        )
    if valid.shape != pitch_xy.shape[:3]:
        raise ValueError(
            f"valid shape {valid.shape} must match pitch_xy[:3]="
            f"{tuple(pitch_xy.shape[:3])}"
        )
    Gx, Gy = int(grid[0]), int(grid[1])
    if Gx < 1 or Gy < 1:
        raise ValueError(f"grid must have positive sizes, got {grid}")
    B, T, P, _ = pitch_xy.shape
    Z = Gx * Gy
    dtype = pitch_xy.dtype
    device = pitch_xy.device

    # Map pitch coords to grid coords where cell-i center has u=i.
    u = pitch_xy[..., 0] * Gx - 0.5  # (B, T, P)
    v = pitch_xy[..., 1] * Gy - 0.5
    i0 = torch.floor(u).long()
    j0 = torch.floor(v).long()
    fu = u - i0.to(dtype)
    fv = v - j0.to(dtype)
    i1 = i0 + 1
    j1 = j0 + 1

    # Clamp into [0, G-1]. Out-of-bounds bilinear contributions collapse
    # onto the boundary cell; the four weights still sum to 1.
    i0c = i0.clamp(0, Gx - 1)
    i1c = i1.clamp(0, Gx - 1)
    j0c = j0.clamp(0, Gy - 1)
    j1c = j1.clamp(0, Gy - 1)

    w00 = (1.0 - fu) * (1.0 - fv)
    w10 = fu * (1.0 - fv)
    w01 = (1.0 - fu) * fv
    w11 = fu * fv

    A = torch.zeros(B, T, P, Z, device=device, dtype=dtype)
    idx00 = (j0c * Gx + i0c).unsqueeze(-1)
    idx10 = (j0c * Gx + i1c).unsqueeze(-1)
    idx01 = (j1c * Gx + i0c).unsqueeze(-1)
    idx11 = (j1c * Gx + i1c).unsqueeze(-1)
    A.scatter_add_(-1, idx00, w00.unsqueeze(-1))
    A.scatter_add_(-1, idx10, w10.unsqueeze(-1))
    A.scatter_add_(-1, idx01, w01.unsqueeze(-1))
    A.scatter_add_(-1, idx11, w11.unsqueeze(-1))

    # Zero out padded players; renormalize valid rows defensively in
    # case any numerical drift accumulated above.
    valid_f = valid.to(dtype).unsqueeze(-1)
    A = A * valid_f
    s = A.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    A = torch.where(valid.unsqueeze(-1), A / s, A)
    return A


def zone_adjacency(
    grid: tuple[int, int] = ZONE_GRID,
    *,
    include_self: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Row-normalized 8-neighborhood adjacency for a ``Gx x Gy`` grid.

    Returns ``(Z, Z)`` with rows summing to 1. When ``include_self`` is
    True (the default), each zone's self-loop is included in the
    neighborhood, so one diffusion step keeps the zone's own state at
    a meaningful weight. Used by ``HGTEncoder`` as a model buffer to
    diffuse zone embeddings to their grid neighbors.
    """
    Gx, Gy = int(grid[0]), int(grid[1])
    Z = Gx * Gy
    adj = torch.zeros(Z, Z, dtype=dtype, device=device)
    for j in range(Gy):
        for i in range(Gx):
            z = j * Gx + i
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= Gx or nj < 0 or nj >= Gy:
                        continue
                    if di == 0 and dj == 0 and not include_self:
                        continue
                    adj[z, nj * Gx + ni] = 1.0
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that treats fully-masked rows as zeros instead of NaN."""
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg_inf)
    # Rows with no True entries would produce NaN under plain softmax; detect
    # and zero them out cleanly.
    has_any = mask.any(dim=dim, keepdim=True)
    weights = torch.softmax(scores, dim=dim)
    weights = torch.where(has_any, weights, torch.zeros_like(weights))
    return weights


@dataclass
class GraphEdges:
    """Boolean adjacency masks per edge type, all shaped ``(B, T, P, P)``.

    Convention: ``edges[type][b, t, i, j] == True`` means there is an
    edge of ``type`` from player ``j`` (source) to player ``i``
    (destination) at batch ``b``, time ``t``.

    Optionally carries dense edge features ``edge_features`` of shape
    ``(B, T, P, P, F_edge)``. When present, ``_TypedAttention`` uses them
    to condition attention scores on the geometric relationship between
    source and destination players (pitch distance, relative position,
    relative velocity, approach/closing speed, same-team flag).

    Optionally also carries zone-node plumbing for the heterogeneous
    extension:

    - ``zone_assignment``: ``(B, T, P, Z)`` soft splat of players into
      pitch zones (rows sum to 1 for valid players, 0 for padded).
    - ``zone_adjacency``: ``(Z, Z)`` row-normalized adjacency used to
      diffuse zone embeddings to grid neighbors.
    - ``degree_counts``: ``(B, T, P, 2)`` raw counts of same-team and
      opposing-team players within the radius-edge threshold. Provides
      the cardinality signal that the kNN-censored ``near`` edge and
      the mean-pooled prototypes structurally cannot recover.
    """

    masks: dict[str, torch.Tensor]
    valid_player: torch.Tensor  # (B, T, P) bool
    edge_features: torch.Tensor | None = None  # (B, T, P, P, F_edge)
    zone_assignment: torch.Tensor | None = None  # (B, T, P, Z)
    zone_adjacency: torch.Tensor | None = None  # (Z, Z)
    degree_counts: torch.Tensor | None = None  # (B, T, P, 2)

    @property
    def edge_types(self) -> tuple[str, ...]:
        return tuple(self.masks.keys())

    @property
    def edge_feature_dim(self) -> int:
        if self.edge_features is None:
            return 0
        return int(self.edge_features.shape[-1])

    @property
    def num_zones(self) -> int:
        if self.zone_assignment is None:
            return 0
        return int(self.zone_assignment.shape[-1])


class _TypedAttention(nn.Module):
    """Single-edge-type multi-head attention (player <- player).

    Optionally consumes per-edge features ``e (B, T, P, P, F_edge)`` and
    projects them to a per-head additive bias on the attention scores.
    The same edge features are shared across edge types upstream, but
    each type owns its own ``edge_proj`` so it can choose how to mix
    them (e.g. ``opponent`` may weigh distance heavily, ``same_team``
    may weigh shared-velocity more).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_feat_dim = int(edge_feat_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.relation_gain = nn.Parameter(torch.ones(num_heads))
        if self.edge_feat_dim > 0:
            self.edge_proj = nn.Linear(self.edge_feat_dim, num_heads)
        else:
            self.edge_proj = None  # type: ignore[assignment]

    def forward(
        self,
        h: torch.Tensor,  # (B, T, P, D)
        mask: torch.Tensor,  # (B, T, P, P) bool
        edge_features: torch.Tensor | None = None,  # (B, T, P, P, F_edge)
    ) -> torch.Tensor:
        B, T, P, _ = h.shape
        H, D = self.num_heads, self.head_dim
        q = self.q(h).view(B, T, P, H, D)
        k = self.k(h).view(B, T, P, H, D)
        v = self.v(h).view(B, T, P, H, D)
        scores = torch.einsum("btihd,btjhd->btijh", q, k) / (D ** 0.5)
        scores = scores * self.relation_gain.view(1, 1, 1, 1, H)
        if self.edge_proj is not None and edge_features is not None:
            if edge_features.shape[-1] != self.edge_feat_dim:
                raise ValueError(
                    f"edge_features last dim {edge_features.shape[-1]} "
                    f"!= configured edge_feat_dim {self.edge_feat_dim}"
                )
            bias = self.edge_proj(edge_features)  # (B, T, P, P, H)
            scores = scores + bias
        mask_h = mask.unsqueeze(-1).expand_as(scores)
        weights = _masked_softmax(scores, mask_h, dim=3)
        out = torch.einsum("btijh,btjhd->btihd", weights, v)
        return out.reshape(B, T, P, H * D)


class PlayerHGTBlock(nn.Module):
    """One HGT-style block updating player node embeddings.

    Combines:
        - One typed attention head per player-player edge type.
        - A team prototype message: mean over same-team valid players.
        - A frame prototype message: mean over all valid players at t.
        - Optionally, a zone-node update (player->zone SUM aggregation,
          zone->zone adjacency diffusion, zone->player read-back). Zones
          give the block a notion of physical congestion that the
          softmax / mean aggregators above structurally cannot represent
          (they are count-invariant convex combinations).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        edge_types: Sequence[str] = PLAYER_EDGE_TYPES,
        dropout: float = 0.1,
        edge_feat_dim: int = 0,
        num_zones: int = 0,
    ) -> None:
        super().__init__()
        if not edge_types:
            raise ValueError("edge_types must be non-empty")
        self.hidden_dim = hidden_dim
        self.edge_types = tuple(edge_types)
        self.edge_feat_dim = int(edge_feat_dim)
        self.num_zones = int(num_zones)
        self.attentions = nn.ModuleDict(
            {
                et: _TypedAttention(hidden_dim, num_heads, edge_feat_dim=edge_feat_dim)
                for et in self.edge_types
            }
        )
        self.team_proj = nn.Linear(hidden_dim, hidden_dim)
        self.frame_proj = nn.Linear(hidden_dim, hidden_dim)
        # One scalar gate per edge/aux source so the block can learn to
        # ignore unhelpful types.
        self.edge_gain = nn.Parameter(torch.ones(len(self.edge_types)))
        self.team_gain = nn.Parameter(torch.ones(()))
        self.frame_gain = nn.Parameter(torch.ones(()))
        if self.num_zones > 0:
            # Zone-node bookkeeping.
            self.zone_pos_embed = nn.Parameter(
                torch.zeros(self.num_zones, hidden_dim)
            )
            nn.init.normal_(self.zone_pos_embed, std=0.02)
            self.zone_value = nn.Linear(hidden_dim, hidden_dim)
            # [cnt_team0, cnt_team1, total, balance] -> hidden_dim
            self.zone_count_proj = nn.Linear(4, hidden_dim)
            self.zone_norm = nn.LayerNorm(hidden_dim)
            self.zone_diffuse_proj = nn.Linear(hidden_dim, hidden_dim)
            self.z_to_p_proj = nn.Linear(hidden_dim, hidden_dim)
            # [own_cnt, opp_cnt] -> hidden_dim
            self.player_count_proj = nn.Linear(2, hidden_dim)
            self.zone_gain = nn.Parameter(torch.ones(()))
        else:
            self.zone_pos_embed = None  # type: ignore[assignment]
            self.zone_value = None  # type: ignore[assignment]
            self.zone_count_proj = None  # type: ignore[assignment]
            self.zone_norm = None  # type: ignore[assignment]
            self.zone_diffuse_proj = None  # type: ignore[assignment]
            self.z_to_p_proj = None  # type: ignore[assignment]
            self.player_count_proj = None  # type: ignore[assignment]
            self.zone_gain = None  # type: ignore[assignment]
        self.combine = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def _team_message(
        self,
        h: torch.Tensor,  # (B, T, P, D)
        teams: torch.Tensor,  # (B, P) int
        valid: torch.Tensor,  # (B, T, P) bool
    ) -> torch.Tensor:
        # Build (B, T, P, P) same-team mask combined with valid.
        same = (teams.unsqueeze(2) == teams.unsqueeze(1)).unsqueeze(1)
        same = same & valid.unsqueeze(2) & valid.unsqueeze(3)
        denom = same.sum(dim=-1).clamp(min=1).unsqueeze(-1)
        # Mean of teammates' embeddings.
        msg = torch.einsum("btij,btjd->btid", same.float(), h) / denom
        return self.team_proj(msg)

    def _frame_message(
        self,
        h: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # Mean over all valid players at time t.
        denom = valid.sum(dim=-1).clamp(min=1).unsqueeze(-1).unsqueeze(-1)
        masked = h * valid.unsqueeze(-1).float()
        frame_proto = masked.sum(dim=2, keepdim=True) / denom  # (B, T, 1, D)
        msg = frame_proto.expand_as(h)
        return self.frame_proj(msg)

    def _update_zone(
        self,
        h: torch.Tensor,  # (B, T, P, D)
        z_prev: torch.Tensor | None,  # (B, T, Z, D) or None
        A: torch.Tensor,  # (B, T, P, Z)
        adj: torch.Tensor,  # (Z, Z)
        teams: torch.Tensor,  # (B, P) int
        valid: torch.Tensor,  # (B, T, P) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_new, cnt0, cnt1).

        Player->zone aggregation uses **sum** (via the soft assignment
        ``A``), which is what makes the zone node able to distinguish
        "3 players here" from "13 players here" - softmax/mean would
        normalize the count away. Zone->zone diffusion happens via the
        row-normalized 8-neighborhood adjacency so each zone's update
        already reflects its grid neighbors.
        """
        B, T, P, D = h.shape
        dtype = h.dtype
        valid_f = valid.to(dtype)
        # Per-team binary "valid AND on team k" masks broadcast over T.
        team0 = ((teams == 0).unsqueeze(1).expand(B, T, P) & valid).to(dtype)
        team1 = ((teams == 1).unsqueeze(1).expand(B, T, P) & valid).to(dtype)
        cnt0 = torch.einsum("btpz,btp->btz", A, team0)
        cnt1 = torch.einsum("btpz,btp->btz", A, team1)
        zfeat = torch.stack(
            [cnt0, cnt1, cnt0 + cnt1, cnt0 - cnt1], dim=-1
        )  # (B, T, Z, 4)
        # Mask invalid players' values before SUM so padding cannot leak.
        zmsg_v = self.zone_value(h) * valid_f.unsqueeze(-1)
        zmsg = torch.einsum("btpz,btpd->btzd", A, zmsg_v)
        z_init = self.zone_pos_embed.view(1, 1, self.num_zones, D)
        z_new = self.zone_norm(zmsg + self.zone_count_proj(zfeat) + z_init)
        if z_prev is not None:
            z_new = z_new + z_prev
        # Adjacency diffusion (Z, Z) @ (B, T, Z, D) -> (B, T, Z, D).
        z_diff = torch.einsum("zZ,btZd->btzd", adj, z_new)
        z_new = z_new + self.zone_diffuse_proj(z_diff)
        return z_new, cnt0, cnt1

    def _zone_to_player_message(
        self,
        z: torch.Tensor,  # (B, T, Z, D)
        A: torch.Tensor,  # (B, T, P, Z)
        teams: torch.Tensor,  # (B, P) int
        cnt0: torch.Tensor,  # (B, T, Z)
        cnt1: torch.Tensor,  # (B, T, Z)
        T: int,
    ) -> torch.Tensor:
        """Return ``(B, T, P, D)`` zone->player message.

        Combines the soft-weighted zone embedding read back through
        ``A`` with explicit team-oriented occupancy counts (own and
        opposing) gathered from the same zones the player splats into.
        """
        z_to_p = torch.einsum("btpz,btzd->btpd", A, z)
        cnt0_at_p = torch.einsum("btpz,btz->btp", A, cnt0)
        cnt1_at_p = torch.einsum("btpz,btz->btp", A, cnt1)
        team0_mask = (teams == 0).unsqueeze(1).expand(-1, T, -1)
        own_cnt = torch.where(team0_mask, cnt0_at_p, cnt1_at_p)
        opp_cnt = torch.where(team0_mask, cnt1_at_p, cnt0_at_p)
        team_counts = torch.stack([own_cnt, opp_cnt], dim=-1)  # (B,T,P,2)
        return self.z_to_p_proj(z_to_p) + self.player_count_proj(team_counts)

    def forward(
        self,
        h: torch.Tensor,  # (B, T, P, D)
        edges: GraphEdges,
        teams: torch.Tensor,  # (B, P) int
        z_prev: torch.Tensor | None = None,  # (B, T, Z, D)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        valid = edges.valid_player
        if valid.shape != h.shape[:3]:
            raise ValueError(
                f"valid_player shape {valid.shape} must match h[:3]={tuple(h.shape[:3])}"
            )

        # Player <- player typed attention.
        agg = torch.zeros_like(h)
        edge_feats = edges.edge_features if self.edge_feat_dim > 0 else None
        for i, et in enumerate(self.edge_types):
            mask = edges.masks.get(et)
            if mask is None:
                continue
            full_mask = mask & valid.unsqueeze(2) & valid.unsqueeze(3)
            if not bool(full_mask.any()):
                continue
            agg = agg + self.edge_gain[i] * self.attentions[et](
                h, full_mask, edge_features=edge_feats
            )

        agg = agg + self.team_gain * self._team_message(h, teams, valid)
        agg = agg + self.frame_gain * self._frame_message(h, valid)

        z_out: torch.Tensor | None = None
        if self.num_zones > 0 and edges.zone_assignment is not None:
            if edges.zone_adjacency is None:
                raise ValueError(
                    "zone_assignment requires zone_adjacency on the same GraphEdges"
                )
            T = h.shape[1]
            z_out, cnt0, cnt1 = self._update_zone(
                h=h,
                z_prev=z_prev,
                A=edges.zone_assignment,
                adj=edges.zone_adjacency,
                teams=teams,
                valid=valid,
            )
            zmsg_to_p = self._zone_to_player_message(
                z=z_out,
                A=edges.zone_assignment,
                teams=teams,
                cnt0=cnt0,
                cnt1=cnt1,
                T=T,
            )
            agg = agg + self.zone_gain * zmsg_to_p

        agg = self.combine(agg)
        h = self.norm(h + self.dropout(agg))
        h = self.norm_ff(h + self.dropout(self.ff(h)))
        # Zero out padded columns so downstream temporal stages do not
        # learn from invalid slots.
        h = h * valid.unsqueeze(-1).to(h.dtype)
        return h, z_out


class PlayerNodeEmbedder(nn.Module):
    """Per-player feature -> hidden_dim embedding.

    Concatenates pitch (x, y), velocity, acceleration (optional), bbox,
    role embedding, team embedding, optional jersey-number embedding,
    optional per-player visual embedding, an optional ``extra_scalars``
    branch (goal-distance / radius degree-count features), and optional
    global / match-time features, then projects into ``hidden_dim``.

    Visual features (e.g. frozen DINOv2 ViT-S/14 embeddings of padded
    player crops) are passed through a dedicated LayerNorm + Linear +
    GELU projection of width ``visual_proj_dim`` (defaults to
    ``hidden_dim // 2``) before concatenation. Set ``visual_dim = 0``
    to disable the visual branch entirely; in that case
    ``forward(visual_features=...)`` is silently ignored.

    The jersey branch (``num_jerseys > 0``) embeds the FOOTPASS
    ``shirt_number`` column as a categorical with width
    ``hidden_dim // 8``. Index 0 is reserved for "unknown" so callers
    must remap the ``-1`` sentinel before passing it in. The branch
    is more identity-stable than ``player_id`` across tracklet switches.

    The extra-scalars branch (``extra_scalar_dim > 0``) is a dedicated
    LayerNorm + Linear (+ GELU + Dropout) projection over a small bundle
    of derived per-player scalars (e.g. distance to own/opp goal,
    nearest sideline, plus the radius-edge degree counts). Keeping it
    on its own branch avoids distorting the kinematic ``feat_norm`` with
    features that have very different statistics (raw counts).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_roles: int = 14,
        num_teams: int = 3,
        global_dim: int = 0,
        time_dim: int = 0,
        use_acceleration: bool = True,
        visual_dim: int = 0,
        visual_proj_dim: int | None = None,
        visual_dropout: float = 0.0,
        num_jerseys: int = 0,
        jersey_dim: int | None = None,
        extra_scalar_dim: int = 0,
        extra_scalar_proj_dim: int | None = None,
        extra_scalar_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_embed = nn.Embedding(num_roles, hidden_dim // 4)
        self.team_embed = nn.Embedding(num_teams, hidden_dim // 4)
        feat_dim = 2 + 2 + 4  # xy + velocity + bbox
        self.use_acceleration = bool(use_acceleration)
        if self.use_acceleration:
            feat_dim += 2  # acceleration
        self.feat_dim = feat_dim
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.global_dim = int(global_dim)
        self.time_dim = int(time_dim)
        self.visual_dim = int(visual_dim)
        if visual_proj_dim is None:
            visual_proj_dim = hidden_dim // 2
        self.visual_proj_dim = int(visual_proj_dim) if self.visual_dim > 0 else 0
        if self.visual_dim > 0:
            self.visual_norm = nn.LayerNorm(self.visual_dim)
            self.visual_proj = nn.Sequential(
                nn.Linear(self.visual_dim, self.visual_proj_dim),
                nn.GELU(),
                nn.Dropout(float(visual_dropout)),
            )
        else:
            self.visual_norm = None  # type: ignore[assignment]
            self.visual_proj = None  # type: ignore[assignment]

        self.num_jerseys = int(num_jerseys)
        if jersey_dim is None:
            jersey_dim = hidden_dim // 8
        self.jersey_dim = int(jersey_dim) if self.num_jerseys > 0 else 0
        if self.num_jerseys > 0:
            self.jersey_embed = nn.Embedding(self.num_jerseys, self.jersey_dim)
        else:
            self.jersey_embed = None  # type: ignore[assignment]

        self.extra_scalar_dim = int(extra_scalar_dim)
        if extra_scalar_proj_dim is None:
            extra_scalar_proj_dim = hidden_dim // 4
        self.extra_scalar_proj_dim = (
            int(extra_scalar_proj_dim) if self.extra_scalar_dim > 0 else 0
        )
        if self.extra_scalar_dim > 0:
            self.extra_scalar_norm = nn.LayerNorm(self.extra_scalar_dim)
            self.extra_scalar_proj = nn.Sequential(
                nn.Linear(self.extra_scalar_dim, self.extra_scalar_proj_dim),
                nn.GELU(),
                nn.Dropout(float(extra_scalar_dropout)),
            )
        else:
            self.extra_scalar_norm = None  # type: ignore[assignment]
            self.extra_scalar_proj = None  # type: ignore[assignment]

        self.proj = nn.Linear(
            feat_dim
            + hidden_dim // 4
            + hidden_dim // 4
            + self.global_dim
            + self.time_dim
            + self.visual_proj_dim
            + self.jersey_dim
            + self.extra_scalar_proj_dim,
            hidden_dim,
        )

    def forward(
        self,
        pitch_xy: torch.Tensor,  # (B, T, P, 2)
        velocity: torch.Tensor,  # (B, T, P, 2)
        bbox: torch.Tensor,  # (B, T, P, 4)
        roles: torch.Tensor,  # (B, T, P) long
        teams: torch.Tensor,  # (B, P) long, broadcast over T
        acceleration: torch.Tensor | None = None,  # (B, T, P, 2)
        global_features: torch.Tensor | None = None,  # (B, T, F)
        time_features: torch.Tensor | None = None,  # (B, T, F_time)
        visual_features: torch.Tensor | None = None,  # (B, T, P, F_visual)
        shirt_numbers: torch.Tensor | None = None,  # (B, P) long
        extra_scalars: torch.Tensor | None = None,  # (B, T, P, F_extra)
    ) -> torch.Tensor:
        B, T, P, _ = pitch_xy.shape
        feat_parts = [pitch_xy, velocity, bbox]
        if self.use_acceleration:
            if acceleration is None:
                acceleration = torch.zeros_like(velocity)
            feat_parts.insert(2, acceleration)
        feat = torch.cat(feat_parts, dim=-1)
        feat = self.feat_norm(feat)
        role_emb = self.role_embed(roles.clamp(min=0))
        team_emb = self.team_embed(teams.clamp(min=0)).unsqueeze(1).expand(B, T, P, -1)
        parts = [feat, role_emb, team_emb]
        if self.global_dim > 0:
            if global_features is None:
                raise ValueError(
                    f"global_features required (global_dim={self.global_dim})"
                )
            if global_features.shape[-1] != self.global_dim:
                raise ValueError(
                    f"global_features last dim must be {self.global_dim}, "
                    f"got {global_features.shape[-1]}"
                )
            parts.append(global_features.unsqueeze(2).expand(B, T, P, self.global_dim))
        if self.time_dim > 0:
            if time_features is None:
                raise ValueError(
                    f"time_features required (time_dim={self.time_dim})"
                )
            if time_features.shape[-1] != self.time_dim:
                raise ValueError(
                    f"time_features last dim must be {self.time_dim}, "
                    f"got {time_features.shape[-1]}"
                )
            parts.append(time_features.unsqueeze(2).expand(B, T, P, self.time_dim))
        if self.visual_dim > 0:
            if visual_features is None:
                raise ValueError(
                    f"visual_features required (visual_dim={self.visual_dim})"
                )
            if visual_features.shape[-1] != self.visual_dim:
                raise ValueError(
                    f"visual_features last dim must be {self.visual_dim}, "
                    f"got {visual_features.shape[-1]}"
                )
            if visual_features.shape[:3] != (B, T, P):
                raise ValueError(
                    f"visual_features must be (B={B}, T={T}, P={P}, F), got "
                    f"{tuple(visual_features.shape)}"
                )
            v = self.visual_norm(visual_features)
            v = self.visual_proj(v)
            parts.append(v)
        if self.num_jerseys > 0:
            if shirt_numbers is None:
                # Treat as "all unknown" rather than erroring, so callers
                # that have not yet plumbed shirt numbers still work.
                shirt_idx = torch.zeros(
                    (B, P), dtype=torch.long, device=pitch_xy.device
                )
            else:
                if shirt_numbers.shape != (B, P):
                    raise ValueError(
                        f"shirt_numbers must be (B={B}, P={P}), got "
                        f"{tuple(shirt_numbers.shape)}"
                    )
                # Reserve embedding slot 0 for unknown / out-of-range.
                shirt_idx = torch.where(
                    (shirt_numbers < 0) | (shirt_numbers >= self.num_jerseys),
                    torch.zeros_like(shirt_numbers),
                    shirt_numbers,
                )
            jersey_emb = self.jersey_embed(shirt_idx).unsqueeze(1).expand(
                B, T, P, self.jersey_dim
            )
            parts.append(jersey_emb)
        if self.extra_scalar_dim > 0:
            if extra_scalars is None:
                raise ValueError(
                    f"extra_scalars required (extra_scalar_dim={self.extra_scalar_dim})"
                )
            if extra_scalars.shape[-1] != self.extra_scalar_dim:
                raise ValueError(
                    f"extra_scalars last dim must be {self.extra_scalar_dim}, "
                    f"got {extra_scalars.shape[-1]}"
                )
            if extra_scalars.shape[:3] != (B, T, P):
                raise ValueError(
                    f"extra_scalars must be (B={B}, T={T}, P={P}, F), got "
                    f"{tuple(extra_scalars.shape)}"
                )
            e = self.extra_scalar_norm(extra_scalars)
            e = self.extra_scalar_proj(e)
            parts.append(e)
        x = torch.cat(parts, dim=-1)
        return self.proj(x)


EDGE_FEATURE_DIM: int = 7
"""Number of channels emitted by ``compute_edge_features``.

Layout: ``[dist, dx, dy, dvx, dvy, closing_speed, same_team_flag]``.
"""


def compute_edge_features(
    teams: torch.Tensor,  # (B, P)
    pitch_xy: torch.Tensor,  # (B, T, P, 2)
    valid: torch.Tensor,  # (B, T, P) bool
    velocity: torch.Tensor | None = None,  # (B, T, P, 2)
) -> torch.Tensor:
    """Return dense per-edge features ``(B, T, P, P, EDGE_FEATURE_DIM)``.

    The ``i``-th destination receives messages from the ``j``-th source,
    so deltas are ``source - dest``: ``dx = pitch_j - pitch_i``.

    Closing speed is the projection of relative velocity onto the unit
    vector from ``j`` to ``i`` (positive when ``j`` is approaching ``i``).
    All channels are zeroed on invalid endpoints so padding rows do not
    bias the per-head projection.
    """
    B, T, P, _ = pitch_xy.shape
    device = pitch_xy.device
    dtype = pitch_xy.dtype

    dx = pitch_xy.unsqueeze(3) - pitch_xy.unsqueeze(2)  # (B, T, P, P, 2)
    dist = torch.sqrt((dx * dx).sum(-1) + 1e-9).unsqueeze(-1)  # (B, T, P, P, 1)

    if velocity is None:
        dv = torch.zeros_like(dx)
    else:
        dv = velocity.unsqueeze(3) - velocity.unsqueeze(2)

    # Closing speed: -d/dt(distance). With dx = source - dest, the unit
    # vector from dest to source is dx / |dx|; closing speed = -<dv, dx>/|dx|.
    norm = (dist + 1e-6)
    closing = (-(dv * dx).sum(-1, keepdim=True)) / norm  # (B, T, P, P, 1)

    same_team = (teams.unsqueeze(2) == teams.unsqueeze(1)).unsqueeze(1).expand(B, T, P, P)
    same_team_f = same_team.to(dtype).unsqueeze(-1)

    feats = torch.cat([dist, dx, dv, closing, same_team_f], dim=-1)
    # Zero edges that touch an invalid endpoint so the bias channel does
    # not leak signal from padded slots into valid attention heads.
    valid_pair = (valid.unsqueeze(2) & valid.unsqueeze(3)).to(dtype).unsqueeze(-1)
    return feats * valid_pair


def build_player_graphs(
    teams: torch.Tensor,  # (B, P)
    pitch_xy: torch.Tensor,  # (B, T, P, 2)
    valid: torch.Tensor,  # (B, T, P) bool
    *,
    knn: int = 4,
    add_self_loop: bool = True,
    velocity: torch.Tensor | None = None,
    compute_edge_features_flag: bool = False,
    use_radius: bool = False,
    radius: float = DEFAULT_RADIUS,
    compute_degree_counts: bool = False,
) -> GraphEdges:
    """Construct per-step, per-edge-type adjacency masks.

    - ``self``: identity per valid player (optional).
    - ``same_team``: connect each player to teammates (excluding self).
    - ``opponent``: connect each player to opponents.
    - ``near``: kNN by pitch distance among valid players (excluding
      self), capped at ``knn`` nearest neighbours.
    - ``radius`` (when ``use_radius=True``): all valid players within
      ``radius`` (normalized pitch units), variable degree. Unlike
      ``near``, this edge type does not censor at a fixed k; it pairs
      with ``compute_degree_counts`` to give the embedder a real
      cardinality signal.

    When ``compute_edge_features_flag`` is True, also attach a dense
    edge feature tensor to the returned ``GraphEdges`` (see
    ``compute_edge_features`` for the channel layout).

    When ``compute_degree_counts`` is True, also attach a
    ``(B, T, P, 2)`` tensor of ``[n_same_within_r, n_opp_within_r]``
    where the counts exclude the player themselves and treat padded
    players as absent.
    """
    B, T, P, _ = pitch_xy.shape
    device = pitch_xy.device

    same_team = (teams.unsqueeze(2) == teams.unsqueeze(1)).unsqueeze(1).expand(B, T, P, P)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(3)
    same_team = same_team & valid_pair
    eye = torch.eye(P, dtype=torch.bool, device=device).view(1, 1, P, P)
    same_no_self = same_team & ~eye
    opponent = (~same_team) & valid_pair

    dx = pitch_xy.unsqueeze(2) - pitch_xy.unsqueeze(3)  # (B, T, P, P, 2)
    dist = torch.sqrt((dx * dx).sum(-1) + 1e-9)
    big = torch.finfo(dist.dtype).max / 4
    dist_for_knn = torch.where(valid_pair, dist, torch.full_like(dist, big))
    dist_for_knn = torch.where(eye.expand_as(dist), torch.full_like(dist, big), dist_for_knn)

    eff_k = min(knn, P - 1) if P > 1 else 0
    near = torch.zeros((B, T, P, P), dtype=torch.bool, device=device)
    if eff_k > 0:
        _, idx = torch.topk(dist_for_knn, eff_k, dim=-1, largest=False)
        near.scatter_(-1, idx, True)
    near = near & valid_pair & ~eye

    masks: dict[str, torch.Tensor] = {
        "same_team": same_no_self,
        "opponent": opponent,
        "near": near,
    }
    if add_self_loop:
        self_mask = eye.expand(B, T, P, P) & valid_pair
        masks["self"] = self_mask

    if use_radius:
        within = (dist <= float(radius)) & valid_pair & ~eye
        masks["radius"] = within

    degree_counts: torch.Tensor | None = None
    if compute_degree_counts:
        # Excludes self via eye mask; treats padded entries as absent
        # via valid_pair. Counts are unnormalized integers cast to the
        # pitch_xy dtype so they can flow straight into a Linear.
        within_count = (dist <= float(radius)) & valid_pair & ~eye
        same_count = (within_count & same_team).sum(dim=-1)  # (B, T, P)
        opp_count = (within_count & ~same_team).sum(dim=-1)
        degree_counts = torch.stack(
            [same_count.to(pitch_xy.dtype), opp_count.to(pitch_xy.dtype)],
            dim=-1,
        )

    edge_features = None
    if compute_edge_features_flag:
        edge_features = compute_edge_features(
            teams=teams, pitch_xy=pitch_xy, valid=valid, velocity=velocity
        )
    return GraphEdges(
        masks=masks,
        valid_player=valid,
        edge_features=edge_features,
        degree_counts=degree_counts,
    )


class HGTEncoder(nn.Module):
    """Stack of ``PlayerHGTBlock`` layers operating on ``(B, T, P, D)``.

    When ``num_zones > 0`` the encoder also threads a zone-node tensor
    ``z: (B, T, Z, D)`` through the stack. ``z`` is initialised to
    zeros; each layer's ``_update_zone`` adds the per-team SUM counts
    and the player-aggregated message, then the result is carried into
    the next layer as a residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        edge_types: Sequence[str] = PLAYER_EDGE_TYPES,
        dropout: float = 0.1,
        edge_feat_dim: int = 0,
        num_zones: int = 0,
    ) -> None:
        super().__init__()
        self.edge_feat_dim = int(edge_feat_dim)
        self.num_zones = int(num_zones)
        self.layers = nn.ModuleList(
            [
                PlayerHGTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_types=edge_types,
                    dropout=dropout,
                    edge_feat_dim=edge_feat_dim,
                    num_zones=num_zones,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        edges: GraphEdges,
        teams: torch.Tensor,
    ) -> torch.Tensor:
        z: torch.Tensor | None = None
        for layer in self.layers:
            h, z = layer(h, edges, teams, z_prev=z)
        return h
