"""PCBAS / FOOTPASS Codabench submission formatting and schema validation.

The Codabench evaluator expects a zip of per-match prediction files. We
follow the SoccerNet convention (one directory per match containing a
``Labels-ball.json`` file) and add the PCBAS-specific ``player_id``
field for player-centric scoring. The layout is::

    submission.zip
    ├── game_01/Labels-ball.json
    ├── game_02/Labels-ball.json
    └── ...

Each ``Labels-ball.json`` follows::

    {
        "UrlLocal": "game_01",
        "predictions": [
            {
                "gameTime": "1 - 00:00",
                "label": "Pass",
                "position": "0",
                "half": 1,
                "confidence": "0.95",
                "player_id": 101
            },
            ...
        ]
    }

This module deliberately keeps the conversion logic out of the CLI so
tests can exercise the schema and grouping without spinning up a full
training pipeline. ``validate_submission_payload`` is the single source
of truth for the schema rules and is reused by both the writer and the
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


SUBMISSION_FILE_NAME = "Labels-ball.json"

# Required top-level fields in a per-match Labels-ball.json document.
REQUIRED_DOC_FIELDS: tuple[str, ...] = ("UrlLocal", "predictions")

# Required per-prediction fields. ``half`` and ``player_id`` are PCBAS
# specific extensions on top of the classic SoccerNet schema.
REQUIRED_PRED_FIELDS: tuple[str, ...] = (
    "gameTime",
    "label",
    "position",
    "half",
    "confidence",
    "player_id",
)


@dataclass(frozen=True)
class InternalPrediction:
    """Frame-level prediction as emitted by ``scripts/infer.py``.

    Internal representation only. ``frame`` is absolute within the
    match-half (matching the cache / training conventions). ``half`` is
    1-based; the writer uses it to pick the "first half" vs "second
    half" prefix in the ``gameTime`` field.
    """

    match_id: str
    half: int
    frame: int
    class_id: int
    player_id: int
    score: float
    fps: float = 25.0

    @property
    def time_seconds(self) -> float:
        return float(self.frame) / max(float(self.fps), 1e-6)


def _format_game_time(half: int, seconds: float) -> str:
    """Return ``"<half> - MM:SS"`` matching SoccerNet's gameTime format."""
    half = max(int(half), 1)
    total = max(int(round(seconds)), 0)
    minutes, secs = divmod(total, 60)
    return f"{half} - {minutes:02d}:{secs:02d}"


def _format_position(seconds: float) -> str:
    """SoccerNet's ``position`` is the elapsed-time milliseconds as a string."""
    ms = max(int(round(seconds * 1000.0)), 0)
    return str(ms)


def _class_label(class_id: int) -> str:
    """Return the human-readable class name for a 1-based PCBAS class id."""
    return PCBAS_CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")


def prediction_to_payload(pred: InternalPrediction) -> dict:
    """Convert an ``InternalPrediction`` to the Codabench dict shape."""
    return {
        "gameTime": _format_game_time(pred.half, pred.time_seconds),
        "label": _class_label(pred.class_id),
        "position": _format_position(pred.time_seconds),
        "half": int(pred.half),
        "confidence": f"{float(pred.score):.6f}",
        "player_id": int(pred.player_id),
    }


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
                int(p.class_id),
                int(p.player_id),
            )
        )
    return by_match


def build_match_document(match_id: str, preds: Sequence[InternalPrediction]) -> dict:
    """Build the ``Labels-ball.json`` payload for one match."""
    return {
        "UrlLocal": str(match_id),
        "predictions": [prediction_to_payload(p) for p in preds],
    }


def validate_submission_payload(payload: Mapping) -> list[str]:
    """Return a list of validation errors (empty == valid).

    Used by tests and by the writer itself before flushing the zip so a
    malformed document is caught locally rather than by the Codabench
    server. The checks intentionally mirror the public PCBAS rules and
    do not require importing torch.
    """
    errors: list[str] = []
    for field in REQUIRED_DOC_FIELDS:
        if field not in payload:
            errors.append(f"missing top-level field {field!r}")
    if errors:
        return errors

    if not isinstance(payload["UrlLocal"], str) or not payload["UrlLocal"]:
        errors.append("UrlLocal must be a non-empty string")
    preds = payload["predictions"]
    if not isinstance(preds, list):
        errors.append("predictions must be a list")
        return errors

    valid_labels = set(PCBAS_CLASS_NAMES.values())
    game_time_re = re.compile(r"^[12] - \d{2}:\d{2}$")
    for i, pred in enumerate(preds):
        if not isinstance(pred, Mapping):
            errors.append(f"predictions[{i}] must be a dict")
            continue
        for field in REQUIRED_PRED_FIELDS:
            if field not in pred:
                errors.append(f"predictions[{i}] missing field {field!r}")
        if "gameTime" in pred and not game_time_re.match(str(pred["gameTime"])):
            errors.append(
                f"predictions[{i}].gameTime {pred['gameTime']!r} does not "
                "match '<half> - MM:SS'"
            )
        if "label" in pred and pred["label"] not in valid_labels:
            errors.append(
                f"predictions[{i}].label {pred['label']!r} is not a known PCBAS class"
            )
        if "position" in pred:
            try:
                int(pred["position"])
            except (TypeError, ValueError):
                errors.append(
                    f"predictions[{i}].position must be a stringified int"
                )
        if "half" in pred and int(pred["half"]) not in (1, 2):
            errors.append(
                f"predictions[{i}].half {pred['half']!r} must be 1 or 2"
            )
        if "confidence" in pred:
            try:
                conf = float(pred["confidence"])
            except (TypeError, ValueError):
                errors.append(
                    f"predictions[{i}].confidence must parse as a float"
                )
            else:
                if not 0.0 <= conf <= 1.0:
                    errors.append(
                        f"predictions[{i}].confidence {conf} outside [0, 1]"
                    )
        if "player_id" in pred:
            try:
                int(pred["player_id"])
            except (TypeError, ValueError):
                errors.append(
                    f"predictions[{i}].player_id must be an integer id"
                )
    return errors


def write_submission_zip(
    by_match: Mapping[str, Sequence[InternalPrediction]],
    out_path: Path,
) -> dict[str, dict]:
    """Write the final submission zip and return the per-match documents.

    The returned dict can be used by tests to assert payload shape
    without re-reading the zip. The function validates each document
    before writing and raises ``ValueError`` if any errors are found.
    """
    docs: dict[str, dict] = {}
    for match_id in sorted(by_match):
        doc = build_match_document(match_id, by_match[match_id])
        errors = validate_submission_payload(doc)
        if errors:
            raise ValueError(
                f"Refusing to write invalid submission for {match_id!r}: "
                + "; ".join(errors)
            )
        docs[match_id] = doc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for match_id, doc in docs.items():
            arcname = f"{match_id}/{SUBMISSION_FILE_NAME}"
            zf.writestr(arcname, json.dumps(doc, indent=2, sort_keys=False))
    return docs


# --------------------------------------------------------------------- loaders


def load_internal_predictions(
    path: Path,
    *,
    match_id: str | None = None,
    half: int | None = None,
    fps: float = 25.0,
) -> list[InternalPrediction]:
    """Load ``scripts/infer.py``-formatted predictions into the internal form.

    ``infer.py`` writes a list of ``{frame, class_id, player_id, score, ...}``
    dicts. The internal type needs a ``match_id`` and ``half``; both
    can be supplied explicitly or inferred from the filename pattern
    ``<match_id>__<half_id>.json`` (where ``half_id`` looks like ``H1`` /
    ``H2`` or contains a digit). When ``half`` is omitted and the
    filename has no half segment, the loader defaults to 1.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Predictions file {path} must contain a JSON array of dicts"
        )
    inferred_match_id, inferred_half = _infer_from_filename(path)
    final_match = match_id or inferred_match_id
    final_half = int(half if half is not None else inferred_half)
    if not final_match:
        raise ValueError(
            f"Could not infer match_id for {path}; pass --match-id explicitly."
        )
    out: list[InternalPrediction] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        out.append(
            InternalPrediction(
                match_id=str(final_match),
                half=final_half,
                frame=int(entry["frame"]),
                class_id=int(entry["class_id"]),
                player_id=int(entry["player_id"]),
                score=float(entry["score"]),
                fps=float(fps),
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
