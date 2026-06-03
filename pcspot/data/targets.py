"""Player-aware CALF target generation.

The original SoccerNet CALF (Cioppa et al. 2020) supervises a
``(T, C)`` temporal segmentation tensor: for every frame ``t`` and
class ``c`` the target depends on the time-shift to the nearest
ground-truth action. We extend that target to ``(T, P, C)``, where
``P`` is the per-window stable player column produced by
``StackedSample``. Only the player who is *responsible* for an event
gets a positive temporal kernel. Other players are supervised as
negatives, but with two physically-motivated downweights:

- a Gaussian distance falloff that softens nearby players, and
- a class-aware team policy that softens teammates for ball-progression
  actions (Drive/Pass/Cross/Shot/Header) and softens opponents for
  duel actions (Tackle/Block).

Zones per class ``c`` (in frames; ``K1`` and ``K2`` are class-tunable):

- ``[t0 - inf, t0 - K2)``     : far-before, target=0, weight=1
- ``[t0 - K2, t0)``           : uncertain-before, target=0, weight=0
- ``t0``                      : event, target=1, weight=W_pos
- ``(t0, t0 + K1]``           : near-after, target=1, weight=W_pos
- ``(t0 + K1, t0 + K1 + K2]`` : decay-after, target=ramp 1->0, weight=1
- ``(t0 + K1 + K2, +inf)``    : far-after, target=0, weight=1

A second helper, ``build_objectness_targets``, derives a per-(t, p)
"is any event happening" target from the per-class targets. It is used
to train the confidence/objectness head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pcspot.data.schema import NUM_PCBAS_CLASSES, StackedSample


# Class id sets used by the team-aware policy. Class ids are 1-based,
# following ``PCBAS_CLASS_NAMES``.
ATTACKING_CLASSES_DEFAULT: frozenset[int] = frozenset(
    {
        1,  # Drive
        2,  # Pass
        3,  # Cross
        4,  # Shot
        5,  # Header
        9,  # High Pass
    }
)
DUEL_CLASSES_DEFAULT: frozenset[int] = frozenset(
    {
        7,  # Tackle
        8,  # Block
    }
)


DistanceFalloff = Literal["hard", "gaussian"]


@dataclass
class CalfConfig:
    """Tunable CALF window radii, positive weights, and ambiguity policy."""

    k1_default: int = 2
    """Frames after the event still treated as fully positive."""

    k2_default: int = 20
    """Frames forming the symmetric uncertainty / decay tail."""

    per_class_window: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Optional ``{class_id: (K1, K2)}`` overrides; class ids are 1-based."""

    positive_weight: float = 4.0
    """Default loss weight applied to ``[t0, t0 + K1]`` for the responsible player."""

    per_class_positive_weight: dict[int, float] = field(default_factory=dict)
    """Optional ``{class_id: W_pos}`` overrides; class ids are 1-based."""

    # Distance falloff.
    distance_falloff: DistanceFalloff = "gaussian"
    """``"hard"`` reproduces the legacy radius cutoff; ``"gaussian"`` smooths it."""

    ambiguity_weight: float = 0.3
    """Floor weight for the closest non-responsible players (d -> 0)."""

    ambiguity_radius: float = 0.05
    """Pitch distance (normalized in [0, 1] units) at which the floor lifts to ~1."""

    gaussian_sigma: float | None = None
    """Optional sigma override (normalized pitch units). Defaults to
    ``ambiguity_radius`` so the Gaussian recovers ``ambiguity_weight`` at
    ``d == 0`` and approaches ``1`` for ``d >> radius``."""

    # Team-aware policy.
    teammate_floor: float = 0.5
    """Floor weight applied to teammates of the actor for attacking classes."""

    opponent_floor: float = 0.5
    """Floor weight applied to opponents of the actor for duel classes."""

    attacking_classes: frozenset[int] = field(default_factory=lambda: ATTACKING_CLASSES_DEFAULT)
    """Classes where teammates near the actor receive the ``teammate_floor`` softening."""

    duel_classes: frozenset[int] = field(default_factory=lambda: DUEL_CLASSES_DEFAULT)
    """Classes where opponents near the actor receive the ``opponent_floor`` softening."""

    # Time window for ambiguity downweight, expressed as a multiplier of (K1 + K2).
    ambiguity_time_scale: float = 1.0
    """Multiplier applied to the ``(K1 + K2)`` half-window used to bound the
    temporal extent of the ambiguity downweight. Increase if a class's after-event
    ambiguity is longer than the CALF tail."""

    def window_for(self, class_id_one_based: int) -> tuple[int, int]:
        return self.per_class_window.get(
            class_id_one_based, (self.k1_default, self.k2_default)
        )

    def positive_weight_for(self, class_id_one_based: int) -> float:
        return float(
            self.per_class_positive_weight.get(class_id_one_based, self.positive_weight)
        )

    def effective_sigma(self) -> float:
        if self.gaussian_sigma is not None and self.gaussian_sigma > 0:
            return float(self.gaussian_sigma)
        return float(max(self.ambiguity_radius, 1e-6))


def _zone_kernel(delta: np.ndarray, k1: int, k2: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute (target, weight) for an array of time-shifts ``delta = t - t0``.

    The weight returned here uses ``1.0`` everywhere except the uncertain-before
    zone, which is set to ``0.0``. Positive-plateau elevation to the class's
    ``positive_weight`` is applied separately by the caller so we can keep this
    helper class-agnostic.
    """
    target = np.zeros_like(delta, dtype=np.float32)
    weight = np.ones_like(delta, dtype=np.float32)

    pre_mask = (delta >= -k2) & (delta < 0)
    weight[pre_mask] = 0.0

    pos_mask = (delta >= 0) & (delta <= k1)
    target[pos_mask] = 1.0

    decay_mask = (delta > k1) & (delta <= k1 + k2)
    if decay_mask.any():
        decay_delta = delta[decay_mask].astype(np.float32) - k1
        target[decay_mask] = 1.0 - decay_delta / max(float(k2), 1e-6)

    return target, weight


def _distance_weight(d: np.ndarray, cfg: CalfConfig) -> np.ndarray:
    """Per-player distance-graded weight in ``[ambiguity_weight, 1.0]``.

    For ``"gaussian"`` (default), uses ``floor + (1 - floor) * (1 - exp(-d^2 / (2 sigma^2)))``
    so ``w(0) == ambiguity_weight`` and ``w(d>>sigma) -> 1``.

    For ``"hard"`` (back-compat), reproduces the legacy radius cutoff: ``ambiguity_weight``
    inside ``ambiguity_radius`` and ``1.0`` outside.
    """
    floor = float(cfg.ambiguity_weight)
    if cfg.distance_falloff == "hard":
        inside = d < float(cfg.ambiguity_radius)
        return np.where(inside, floor, 1.0).astype(np.float32)
    sigma = cfg.effective_sigma()
    ramp = 1.0 - np.exp(-(d.astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    return (floor + (1.0 - floor) * ramp).astype(np.float32)


def _team_weight(
    teams: np.ndarray,
    p_resp: int,
    cls_one: int,
    cfg: CalfConfig,
) -> np.ndarray:
    """Per-player class-aware team softening in ``[floor, 1.0]``."""
    P = teams.shape[0]
    w = np.ones((P,), dtype=np.float32)
    if cls_one in cfg.attacking_classes:
        same_team = teams == teams[p_resp]
        w = np.where(same_team, np.float32(cfg.teammate_floor), w)
    if cls_one in cfg.duel_classes:
        opponent = (teams != teams[p_resp]) & (teams >= 0)
        w = np.where(opponent, np.float32(cfg.opponent_floor), w)
    return w


def build_pc_calf_targets(
    stacked: StackedSample,
    *,
    config: CalfConfig | None = None,
    num_classes: int = NUM_PCBAS_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(T, P, C)`` segmentation targets and weights for a sample.

    Returns:
        targets: float32 array of shape ``(T, P, C)``.
        weights: float32 array of shape ``(T, P, C)`` with the same layout.
    """
    cfg = config or CalfConfig()
    T = stacked.num_steps
    P = stacked.num_players
    C = num_classes
    targets = np.zeros((T, P, C), dtype=np.float32)
    weights = np.ones((T, P, C), dtype=np.float32)

    column_alive = stacked.valid_mask.any(axis=0)  # (P,)
    if not column_alive.any() or T == 0:
        weights[:, ~column_alive, :] = 0.0
        return targets, weights

    event_t, event_p = np.where(stacked.targets_class > 0)
    if event_t.size == 0:
        weights[:, ~column_alive, :] = 0.0
        weights[~stacked.valid_mask, :] = 0.0
        return targets, weights

    event_classes = stacked.targets_class[event_t, event_p]
    time_grid = np.arange(T, dtype=np.int64)

    for evt_idx in range(event_t.size):
        t_local = int(event_t[evt_idx])
        p_resp = int(event_p[evt_idx])
        cls_one = int(event_classes[evt_idx])
        if cls_one < 1 or cls_one > C:
            continue
        k1, k2 = cfg.window_for(cls_one)
        w_pos = cfg.positive_weight_for(cls_one)
        delta = time_grid - t_local
        tgt, w = _zone_kernel(delta, k1=k1, k2=k2)
        c_idx = cls_one - 1

        # 1. Positive temporal kernel for the responsible player only.
        targets[:, p_resp, c_idx] = np.maximum(targets[:, p_resp, c_idx], tgt)

        plateau = (delta >= 0) & (delta <= k1)
        existing = weights[:, p_resp, c_idx]
        new_weight = np.where(plateau, np.float32(w_pos), w)
        weights[:, p_resp, c_idx] = np.maximum(existing, new_weight)
        # Uncertain-before keeps 0 weight only where nothing higher has been written.
        uncertain_before = (delta >= -k2) & (delta < 0)
        zero_mask = uncertain_before & (existing == 1.0)
        weights[zero_mask, p_resp, c_idx] = 0.0

        # 2. Non-responsible player ambiguity downweight.
        # Compute distance-graded weight per player (relative to actor's pitch
        # position at the event time), then combine multiplicatively with the
        # class-aware team softening.
        pos_resp = stacked.pitch_xy[t_local, p_resp]
        pos_all = stacked.pitch_xy[t_local]  # (P, 2)
        d = np.linalg.norm(pos_all - pos_resp, axis=-1)
        w_dist = _distance_weight(d, cfg)  # (P,)
        w_team = _team_weight(stacked.teams, p_resp, cls_one, cfg)  # (P,)
        per_player_factor = (w_dist * w_team).astype(np.float32)  # (P,)
        # Exclude the actor: the actor's own weight was already set above.
        per_player_factor[p_resp] = 1.0

        # Restrict the downweight to a temporal window around the event.
        half = max(int(round(cfg.ambiguity_time_scale * (k1 + k2))), 1)
        t_lo = max(t_local - half, 0)
        t_hi = min(t_local + half, T - 1)
        if t_lo > t_hi:
            continue

        # Bug fix: do NOT lower the weight in positions where another event has
        # already set a positive target for this (t, p, c) cell. That would
        # clobber a real positive plateau.
        slab_existing = weights[t_lo : t_hi + 1, :, c_idx]  # (Tw, P)
        slab_targets = targets[t_lo : t_hi + 1, :, c_idx]  # (Tw, P)
        candidate = np.minimum(slab_existing, per_player_factor[None, :])
        # Where the cell holds a positive target, keep the (higher) existing
        # weight unchanged.
        is_positive = slab_targets > 0.0
        weights[t_lo : t_hi + 1, :, c_idx] = np.where(
            is_positive, slab_existing, candidate
        )

    # Mask out columns that are never valid in the window.
    weights[:, ~column_alive, :] = 0.0
    # Mask out per-step invalid (t, p) slots entirely.
    invalid_tp = ~stacked.valid_mask  # (T, P)
    weights[invalid_tp, :] = 0.0

    return targets, weights


def build_objectness_targets(
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-class CALF targets to a per-(t, p) objectness signal.

    Args:
        targets: ``(T, P, C)`` CALF targets from ``build_pc_calf_targets``.
        weights: ``(T, P, C)`` CALF weights from ``build_pc_calf_targets``.

    Returns:
        obj_targets: ``(T, P)`` float32 max-over-classes of ``targets``.
            A position is a positive objectness target whenever any class
            has positive CALF supervision there.
        obj_weights: ``(T, P)`` float32 hybrid reduction:

            - At positions where *any* class is positive (``target > 0``),
              take ``max`` across classes so the actor's strong positive
              plateau (e.g. weight 4.0) propagates as a strong objectness
              positive supervision instead of being washed down by the
              other classes' background weight 1.0.
            - At all other positions, take ``min`` across classes so the
              actor's uncertain-before zone (``weight == 0`` for the
              active class) and padded ``(t, p)`` slots (all weights 0)
              propagate to objectness as "ignored".

            Non-actor background positions keep ``obj_weight == 1.0`` and
            non-actor ambiguity-downweighted positions keep their reduced
            class weight, both of which are sensible "negative" supervisions
            for objectness.
    """
    if targets.shape != weights.shape:
        raise ValueError(
            f"targets and weights must share shape; got {targets.shape} vs {weights.shape}"
        )
    if targets.ndim != 3:
        raise ValueError(f"targets must be (T, P, C); got {targets.shape}")
    obj_targets = targets.max(axis=-1).astype(np.float32)
    has_pos = (targets > 0.0).any(axis=-1)  # (T, P)
    max_w = weights.max(axis=-1)
    min_w = weights.min(axis=-1)
    obj_weights = np.where(has_pos, max_w, min_w).astype(np.float32)
    return obj_targets, obj_weights


def stack_pc_calf_targets(
    samples_targets: list[np.ndarray],
    samples_weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad-and-stack a list of per-sample target/weight arrays into a batch."""
    if not samples_targets:
        raise ValueError("samples_targets must be non-empty")
    if len(samples_targets) != len(samples_weights):
        raise ValueError("targets and weights lists must align")
    T = samples_targets[0].shape[0]
    C = samples_targets[0].shape[2]
    P = max(t.shape[1] for t in samples_targets)
    B = len(samples_targets)
    out_t = np.zeros((B, T, P, C), dtype=np.float32)
    out_w = np.zeros((B, T, P, C), dtype=np.float32)
    for b, (t, w) in enumerate(zip(samples_targets, samples_weights)):
        if t.shape[0] != T or t.shape[2] != C:
            raise ValueError("inconsistent T or C among samples")
        p = t.shape[1]
        out_t[b, :, :p, :] = t
        out_w[b, :, :p, :] = w
    return out_t, out_w


def stack_objectness_targets(
    samples_obj_t: list[np.ndarray],
    samples_obj_w: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad-and-stack ``(T, P)`` objectness targets/weights into ``(B, T, P)``."""
    if not samples_obj_t:
        raise ValueError("samples_obj_t must be non-empty")
    if len(samples_obj_t) != len(samples_obj_w):
        raise ValueError("targets and weights lists must align")
    T = samples_obj_t[0].shape[0]
    P = max(t.shape[1] for t in samples_obj_t)
    B = len(samples_obj_t)
    out_t = np.zeros((B, T, P), dtype=np.float32)
    out_w = np.zeros((B, T, P), dtype=np.float32)
    for b, (t, w) in enumerate(zip(samples_obj_t, samples_obj_w)):
        if t.shape[0] != T:
            raise ValueError("inconsistent T among samples")
        p = t.shape[1]
        out_t[b, :, :p] = t
        out_w[b, :, :p] = w
    return out_t, out_w
