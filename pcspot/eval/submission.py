"""PCBAS / FOOTPASS Codabench submission formatting and schema validation.

The Codabench evaluator expects a single JSON document mapping each
match id to a flat list of player-centric action predictions, where each
prediction is a positional array::

    {
        "game_01_H1": [
            [1425, 0, 10, 2, 0.87],
            [1580, 1, 4, 7, 0.64],
            ...
        ],
        "game_01_H2": [...],
        ...
    }

Each row is ``[frame, team, jersey_number, class_id, score]``:

- ``frame``        — integer frame index within the match-half
- ``team``         — 0 or 1
- ``jersey_number``— shirt number (-1 if unknown)
- ``class_id``     — 1-based PCBAS action class id (see ``PCBAS_CLASS_NAMES``)
- ``score``        — confidence in [0, 1]

This module deliberately keeps the conversion logic out of the CLI so
tests can exercise the schema and grouping without spinning up a full
training pipeline. ``validate_submission_payload`` (and its per-match
counterpart ``validate_match_predictions``) are the single source of
truth for the schema rules and are reused by both the writer and the
test suite.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pcspot.data.schema import PCBAS_CLASS_NAMES


SUBMISSION_FILE_NAME = "predictions.json"

# Each prediction is a fixed-length positional row:
#   [frame, team, jersey_number, class_id, score]
ROW_LEN = 5

# Valid ``class_id`` values. ``class_id == 0`` ("background") is an
# internal decoder sentinel and never appears in submitted predictions.
_VALID_CLASS_IDS = {k for k in PCBAS_CLASS_NAMES if k != 0}


@dataclass(frozen=True)
class InternalPrediction:
    """Frame-level prediction as emitted by ``scripts/infer.py``.

    Internal representation only. ``half`` is 1-based and used solely to
    order a match's predictions (H1 before H2); it is not part of the
    serialized schema.
    """

    match_id: str
    half: int
    frame: int
    team: int
    jersey_number: int
    class_id: int
    score: float


def prediction_to_payload(pred: InternalPrediction) -> list:
    """Convert an ``InternalPrediction`` to the positional row shape."""
    return [
        int(pred.frame),
        int(pred.team),
        int(pred.jersey_number),
        int(pred.class_id),
        float(pred.score),
    ]


def group_predictions_by_match(
    predictions: Iterable[InternalPrediction],
) -> dict[str, list[InternalPrediction]]:
    """Bucket predictions by match-id, preserving deterministic order."""
    by_match: dict[str, list[InternalPrediction]] = {}
    for pred in predictions:
        by_match.setdefault(str(pred.match_id), []).append(pred)
    for match_id, preds in by_match.items():
        preds.sort(
            key=lambda p: (
                int(p.half),
                int(p.frame),
                int(p.team),
                int(p.jersey_number),
            )
        )
    return by_match


def build_match_document(preds: Sequence[InternalPrediction]) -> list[list]:
    """Build the flat list of prediction rows for one match."""
    return [prediction_to_payload(p) for p in preds]


def validate_match_predictions(preds: object) -> list[str]:
    """Return a list of validation errors for one match's prediction list."""
    if not isinstance(preds, list):
        return ["predictions must be a list"]

    errors: list[str] = []
    for i, pred in enumerate(preds):
        if not isinstance(pred, (list, tuple)):
            errors.append(f"predictions[{i}] must be a list")
            continue
        if len(pred) != ROW_LEN:
            errors.append(
                f"predictions[{i}] must have {ROW_LEN} elements "
                "[frame, team, jersey_number, class_id, score]"
            )
            continue
        frame, team, jersey, class_id, score = pred
        try:
            if int(frame) < 0:
                errors.append(f"predictions[{i}][0] frame must be >= 0")
        except (TypeError, ValueError):
            errors.append(f"predictions[{i}][0] frame must be an integer")
        try:
            if int(team) not in (0, 1):
                errors.append(f"predictions[{i}][1] team {team!r} must be 0 or 1")
        except (TypeError, ValueError):
            errors.append(f"predictions[{i}][1] team must be an integer")
        try:
            int(jersey)
        except (TypeError, ValueError):
            errors.append(f"predictions[{i}][2] jersey_number must be an integer")
        try:
            if int(class_id) not in _VALID_CLASS_IDS:
                errors.append(
                    f"predictions[{i}][3] class_id {class_id!r} is not a known "
                    "PCBAS class id"
                )
        except (TypeError, ValueError):
            errors.append(f"predictions[{i}][3] class_id must be an integer")
        try:
            s = float(score)
        except (TypeError, ValueError):
            errors.append(f"predictions[{i}][4] score must parse as a float")
        else:
            if not 0.0 <= s <= 1.0:
                errors.append(f"predictions[{i}][4] score {s} outside [0, 1]")
    return errors


def validate_submission_payload(payload: Mapping) -> list[str]:
    """Return a list of validation errors (empty == valid).

    ``payload`` is the full ``{match_id: [predictions...]}`` document.
    """
    if not isinstance(payload, Mapping):
        return ["submission payload must be a dict keyed by match id"]

    errors: list[str] = []
    for match_id, preds in payload.items():
        if not isinstance(match_id, str) or not match_id:
            errors.append(f"match key {match_id!r} must be a non-empty string")
            continue
        for err in validate_match_predictions(preds):
            errors.append(f"{match_id}: {err}")
    return errors


def write_submission_zip(
    by_match: Mapping[str, Sequence[InternalPrediction]],
    out_path: Path,
) -> dict[str, list[list]]:
    """Write the final submission zip and return the payload it contains.

    The returned dict can be used by tests to assert payload shape
    without re-reading the zip. The function validates the payload
    before writing and raises ``ValueError`` if any errors are found.
    """
    payload: dict[str, list[list]] = {
        str(match_id): build_match_document(by_match[match_id])
        for match_id in sorted(by_match)
    }

    errors = validate_submission_payload(payload)
    if errors:
        raise ValueError(
            "Refusing to write invalid submission: " + "; ".join(errors)
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SUBMISSION_FILE_NAME, json.dumps(payload, indent=2, sort_keys=True))
    return payload


# --------------------------------------------------------------------- loaders


def load_internal_predictions(
    path: Path,
    *,
    match_id: str | None = None,
    half: int | None = None,
) -> list[InternalPrediction]:
    """Load ``scripts/infer.py``-formatted predictions into the internal form.

    ``infer.py`` writes a list of ``[frame, team, jersey_number,
    class_id, score]`` rows. The internal type also needs a ``match_id``
    and ``half``; both can be supplied explicitly or inferred from the
    filename pattern ``<match_id>__<half_id>.json`` (where ``half_id``
    looks like ``H1`` / ``H2`` or contains a digit). When ``half`` is
    omitted and the filename has no half segment, the loader defaults to
    1.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Predictions file {path} must contain a JSON array of rows"
        )
    inferred_match_id, inferred_half = _infer_from_filename(path)
    base_match = match_id or inferred_match_id
    final_half = int(half if half is not None else inferred_half)
    if not base_match:
        raise ValueError(
            f"Could not infer match_id for {path}; pass --match-id explicitly."
        )
    # The submission keys each match-half separately, matching the
    # tactical HDF5 keys (e.g. "game_18_H1", "game_18_H2").
    final_match = f"{base_match}_H{final_half}"
    out: list[InternalPrediction] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != ROW_LEN:
            continue
        frame, team, jersey, class_id, score = entry
        out.append(
            InternalPrediction(
                match_id=str(final_match),
                half=final_half,
                frame=int(frame),
                team=int(team),
                jersey_number=int(jersey),
                class_id=int(class_id),
                score=float(score),
            )
        )
    return out


_FILENAME_RE = re.compile(
    r"^(?P<match>.+?)(?:__(?P<half>[Hh]?(?P<half_num>\d+)))?$"
)


def _infer_from_filename(path: Path) -> tuple[str | None, int]:
    """Best-effort ``match_id`` / ``half`` extraction from a filename stem."""
    m = _FILENAME_RE.match(path.stem)
    if m is None:
        return (None, 1)
    half_num = m.group("half_num")
    return (m.group("match"), int(half_num) if half_num else 1)
