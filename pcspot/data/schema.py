"""Normalized player-centered sample schema.

The whole pipeline is built around a single, well-typed sample format
so that data loaders, graph builders, the temporal bridge, the loss and
the evaluator all agree on layout.

Two complementary representations are defined:

1. ``Sample`` -- a *ragged*, variable-player snapshot, close to the raw
   FOOTPASS / PCBAS rows. Different timesteps can hold a different
   number of players (broadcast tracking does drop players in and out
   of frame). This is the natural representation for graph
   construction, where every timestep gets its own sub-graph.

2. ``StackedSample`` -- a *padded*, fixed ``(T, P)`` grid backed by
   numpy arrays plus a ``valid_mask``. The HGT encoder produces tensors
   on this grid, MS-TCN++ runs over its time axis, and the player-aware
   CALF loss masks invalid ``(t, p)`` slots before reducing.

A small helper ``stack_sample`` converts between the two. Tests live in
``tests/test_schema.py`` and only depend on numpy, so they run with the
project's CPU-only venv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PCBAS_CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "Drive",
    2: "Pass",
    3: "Cross",
    4: "Shot",
    5: "Header",
    6: "Throw-in",
    7: "Tackle",
    8: "Block",
    9: "High Pass",
    10: "Out",
    11: "Free Kick",
    12: "Goal",
}

NUM_PCBAS_CLASSES = 12

PCBAS_ROLE_NAMES: dict[int, str] = {
    1: "GK",
    2: "LB",
    3: "LCB",
    4: "MCB",
    5: "RCB",
    6: "LM",
    7: "RM",
    8: "DM",
    9: "AM",
    10: "LW",
    11: "RW",
    12: "CF",
    13: "RB",
}

NUM_ROLES = 13


@dataclass
class PlayerSnapshot:
    """One player's state at a single timestep.

    The fields mirror the FOOTPASS tactical schema documented in
    ``data/pcbas_one_match/raw/tactical_data_format.txt``. Pitch
    coordinates are normalized in roughly ``[0, 1]``. ROI / bbox
    fields are pixel coordinates in the original fullHD broadcast
    (1920x1080); they may be ``NaN`` when the player is off-screen.
    """

    player_id: int
    team: int
    shirt_number: int
    role_id: int
    x: float
    y: float
    speed_x: float
    speed_y: float
    bbox_xywh: tuple[float, float, float, float]
    visible: bool
    left_to_right: float = 0.0


@dataclass
class EventLabel:
    """Player-centered ground-truth event."""

    frame: int
    player_id: int
    class_id: int


@dataclass
class SampleMeta:
    """Provenance information for a sample window."""

    match_id: str
    half_id: Optional[str] = None
    fps: float = 25.0
    fullhd_width: int = 1920
    fullhd_height: int = 1080
    extra: dict = field(default_factory=dict)


@dataclass
class Sample:
    """Ragged sample: one window of frames with per-frame player lists.

    Attributes:
        frames: monotonically increasing frame indices, length ``T``.
        players_per_step: list of length ``T`` of ``PlayerSnapshot``
            lists. Outer length is ``T``; inner length is the number of
            tracked players at that timestep and may vary.
        events: list of player-centered events that fall inside this
            window. Frame indices are absolute (matching ``frames``).
        global_features: optional ``(T, F_global)`` array of frame-level
            features (e.g. CNN embeddings, audio); ``None`` if unused.
        meta: provenance.
    """

    frames: np.ndarray
    players_per_step: list[list[PlayerSnapshot]]
    events: list[EventLabel]
    global_features: Optional[np.ndarray] = None
    meta: SampleMeta = field(default_factory=lambda: SampleMeta(match_id="?"))

    def __post_init__(self) -> None:
        if self.frames.ndim != 1:
            raise ValueError("frames must be 1D")
        if len(self.players_per_step) != self.frames.shape[0]:
            raise ValueError(
                "players_per_step must have one entry per frame "
                f"(got T={self.frames.shape[0]}, players_per_step={len(self.players_per_step)})"
            )
        if self.global_features is not None:
            if self.global_features.ndim != 2:
                raise ValueError("global_features must be 2D (T, F)")
            if self.global_features.shape[0] != self.frames.shape[0]:
                raise ValueError(
                    "global_features T must match frames length "
                    f"({self.global_features.shape[0]} vs {self.frames.shape[0]})"
                )

    @property
    def num_steps(self) -> int:
        return int(self.frames.shape[0])

    @property
    def max_players(self) -> int:
        return max((len(ps) for ps in self.players_per_step), default=0)

    def unique_player_ids(self) -> list[int]:
        seen: dict[int, None] = {}
        for snaps in self.players_per_step:
            for snap in snaps:
                seen.setdefault(snap.player_id, None)
        return list(seen.keys())


@dataclass
class StackedSample:
    """Padded ``(T, P)`` grid view of a ``Sample``.

    All arrays use float32 except ``player_ids``/``teams``/``roles``
    (int32) and the boolean masks. The ``player_ids`` row is a stable
    mapping of column index ``p`` -> player id for the whole window;
    a player that disappears for some frames keeps its column and is
    masked out via ``valid_mask`` for those frames.

    ``visual_features`` is an optional ``(T, P, F_visual)`` tensor of
    per-player visual embeddings (e.g. frozen DINOv2 features extracted
    from padded player crops). It is aligned to the ``frames`` and
    ``player_ids`` axes so missing crops or padded columns can be left
    as zeros without disturbing the kinematic columns. The dataset, the
    online-inference wrapper, or the visual-feature cache populate this
    field; ``None`` means visual features are disabled for this sample.

    ``left_to_right`` is a per-``(t, p)`` float that records each
    player's attacking direction as reported by the FOOTPASS tactical
    rows (typically ``+1`` for the squad attacking left-to-right at this
    moment and ``-1`` for the other squad; the loader passes the raw
    column through). The zone-node builder uses it to orient the pitch
    grid into an attacking-canonical frame so that "final third" /
    "own third" is consistent across the two squads.

    ``shirt_numbers`` is a per-player int that mirrors the FOOTPASS
    ``shirt_number`` column. It is more identity-stable than
    ``player_ids`` across tracklet identity switches, so the embedder
    uses it as a categorical signal. Missing values are stored as
    ``-1`` and are remapped onto a reserved embedding slot at the
    embedder boundary.
    """

    frames: np.ndarray  # (T,)
    player_ids: np.ndarray  # (P,) int32; same column = same player across T
    teams: np.ndarray  # (P,) int32 in {0, 1}
    roles: np.ndarray  # (T, P) int32; 0 = unknown
    pitch_xy: np.ndarray  # (T, P, 2) float32
    velocity: np.ndarray  # (T, P, 2) float32
    bbox_xywh: np.ndarray  # (T, P, 4) float32; NaN replaced with 0
    visible: np.ndarray  # (T, P) bool; True if player observed at that step
    valid_mask: np.ndarray  # (T, P) bool; column has non-zero rows at this step
    targets_class: np.ndarray  # (T, P) int32; 0 = no event, else class id
    global_features: Optional[np.ndarray]
    meta: SampleMeta
    visual_features: Optional[np.ndarray] = None  # (T, P, F_visual) float32
    left_to_right: Optional[np.ndarray] = None  # (T, P) float32
    shirt_numbers: Optional[np.ndarray] = None  # (P,) int32; -1 = unknown

    @property
    def num_steps(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_players(self) -> int:
        return int(self.player_ids.shape[0])

    def per_step_event_targets(
        self, num_classes: int = NUM_PCBAS_CLASSES
    ) -> np.ndarray:
        """Return ``(T, P, C)`` one-hot event presence (background excluded)."""
        T, P = self.targets_class.shape
        out = np.zeros((T, P, num_classes), dtype=np.float32)
        cls = self.targets_class
        ts, ps = np.where(cls > 0)
        if ts.size:
            out[ts, ps, cls[ts, ps] - 1] = 1.0
        return out


def _player_columns(sample: Sample) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return (player_ids, teams, id->column_index) over the whole window."""
    pid_to_col: dict[int, int] = {}
    pid_team: dict[int, int] = {}
    for snaps in sample.players_per_step:
        for snap in snaps:
            if snap.player_id not in pid_to_col:
                pid_to_col[snap.player_id] = len(pid_to_col)
                pid_team[snap.player_id] = snap.team
    if not pid_to_col:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            pid_to_col,
        )
    ordered = sorted(pid_to_col.items(), key=lambda kv: kv[1])
    pids = np.array([pid for pid, _ in ordered], dtype=np.int32)
    teams = np.array([pid_team[pid] for pid, _ in ordered], dtype=np.int32)
    return pids, teams, pid_to_col


def stack_sample(sample: Sample) -> StackedSample:
    """Convert a ragged ``Sample`` into a padded ``StackedSample``.

    Padding is per-window: every player who appears at least once in
    the window gets a stable column. Steps where the player is missing
    are filled with zeros and masked via ``valid_mask`` and ``visible``.
    """
    T = sample.num_steps
    pids, teams, pid_to_col = _player_columns(sample)
    P = pids.shape[0]

    roles = np.zeros((T, P), dtype=np.int32)
    pitch_xy = np.zeros((T, P, 2), dtype=np.float32)
    velocity = np.zeros((T, P, 2), dtype=np.float32)
    bbox_xywh = np.zeros((T, P, 4), dtype=np.float32)
    visible = np.zeros((T, P), dtype=bool)
    valid_mask = np.zeros((T, P), dtype=bool)
    targets_class = np.zeros((T, P), dtype=np.int32)
    left_to_right = np.zeros((T, P), dtype=np.float32)
    shirt_numbers = np.full((P,), -1, dtype=np.int32)

    for t, snaps in enumerate(sample.players_per_step):
        for snap in snaps:
            p = pid_to_col[snap.player_id]
            roles[t, p] = snap.role_id
            pitch_xy[t, p, 0] = snap.x
            pitch_xy[t, p, 1] = snap.y
            velocity[t, p, 0] = snap.speed_x
            velocity[t, p, 1] = snap.speed_y
            bbox = np.asarray(snap.bbox_xywh, dtype=np.float32)
            bbox = np.where(np.isnan(bbox), 0.0, bbox)
            bbox_xywh[t, p] = bbox
            visible[t, p] = bool(snap.visible)
            valid_mask[t, p] = True
            left_to_right[t, p] = float(snap.left_to_right)
            # Shirt is identity-stable across the window for a given
            # player column; overwrite if the loader ever yields a
            # different value (the last seen wins).
            if snap.shirt_number is not None:
                shirt_numbers[p] = int(snap.shirt_number)

    if sample.events:
        frame_to_idx = {int(f): i for i, f in enumerate(sample.frames)}
        for ev in sample.events:
            if ev.player_id not in pid_to_col:
                continue
            t = frame_to_idx.get(int(ev.frame))
            if t is None:
                continue
            p = pid_to_col[ev.player_id]
            targets_class[t, p] = int(ev.class_id)

    return StackedSample(
        frames=sample.frames.astype(np.int64, copy=False),
        player_ids=pids,
        teams=teams,
        roles=roles,
        pitch_xy=pitch_xy,
        velocity=velocity,
        bbox_xywh=bbox_xywh,
        visible=visible,
        valid_mask=valid_mask,
        targets_class=targets_class,
        global_features=sample.global_features,
        meta=sample.meta,
        visual_features=None,
        left_to_right=left_to_right,
        shirt_numbers=shirt_numbers,
    )


def attach_visual_features(
    stacked: StackedSample,
    visual_features: np.ndarray,
) -> StackedSample:
    """Return a copy of ``stacked`` with ``visual_features`` attached.

    The tensor must be shaped ``(T, P, F_visual)`` and aligned to
    ``stacked.frames`` (time axis) and ``stacked.player_ids`` (player
    axis). Caller is responsible for zeroing the rows of invisible /
    padded players if they are not already zero.
    """
    if visual_features.ndim != 3:
        raise ValueError(
            f"visual_features must be 3D (T, P, F_visual), got shape "
            f"{visual_features.shape}"
        )
    T, P = stacked.num_steps, stacked.num_players
    if visual_features.shape[0] != T or visual_features.shape[1] != P:
        raise ValueError(
            f"visual_features shape {visual_features.shape} does not match "
            f"stacked sample (T={T}, P={P})"
        )
    return StackedSample(
        frames=stacked.frames,
        player_ids=stacked.player_ids,
        teams=stacked.teams,
        roles=stacked.roles,
        pitch_xy=stacked.pitch_xy,
        velocity=stacked.velocity,
        bbox_xywh=stacked.bbox_xywh,
        visible=stacked.visible,
        valid_mask=stacked.valid_mask,
        targets_class=stacked.targets_class,
        global_features=stacked.global_features,
        meta=stacked.meta,
        visual_features=visual_features.astype(np.float32, copy=False),
        left_to_right=stacked.left_to_right,
        shirt_numbers=stacked.shirt_numbers,
    )
