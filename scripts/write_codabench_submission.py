"""Package per-match prediction JSON files into a Codabench submission zip.

Reads the prediction-file layout produced by ``scripts/infer.py``
(one JSON file per match-half with absolute-frame entries) and writes
a ``submission.zip`` whose layout matches the SoccerNet / PCBAS
Codabench expectations::

    submission.zip
    ├── game_01/Labels-ball.json
    ├── game_02/Labels-ball.json
    └── ...

Typical usage::

    # 1. Run inference on each CHALLENGE match-half.
    python scripts/infer.py offline \\
        --checkpoint checkpoints/run1/best.pt \\
        --config config.toml \\
        --match-id game_xx --half H1 \\
        --visual-cache data/pcbas/visual_features \\
        --out predictions/game_xx__H1.json

    # 2. Bundle every match into the Codabench-shaped zip.
    python scripts/write_codabench_submission.py \\
        --predictions predictions/ \\
        --fps 25 \\
        --out submission.zip

Pass ``--explicit-match <id> --explicit-half <n>`` when the filename
does not encode the match-half (e.g. when running on a single file
named arbitrarily).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.eval.submission import (  # noqa: E402
    InternalPrediction,
    group_predictions_by_match,
    load_internal_predictions,
    validate_submission_payload,
    write_submission_zip,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Either a directory of prediction JSON files (one per "
            "match-half, named like ``<match_id>__H<n>.json``) or a "
            "single prediction file."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("submission.zip"),
        help="Output zip path. Default: submission.zip.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate used to translate frame indices into seconds.",
    )
    p.add_argument(
        "--explicit-match",
        type=str,
        default=None,
        help="Override the match-id (only used when --predictions is a file).",
    )
    p.add_argument(
        "--explicit-half",
        type=int,
        default=None,
        help="Override the half (only used when --predictions is a file).",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Build the documents and validate the schema but do not "
            "write the zip. Useful in CI."
        ),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional path for a JSON report containing per-match counts "
            "and validation outcomes."
        ),
    )
    return p.parse_args(argv)


def _iter_prediction_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise SystemExit(f"--predictions {root} is neither a file nor a directory")
    for p in sorted(root.rglob("*.json")):
        yield p


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = list(_iter_prediction_files(args.predictions))
    if not files:
        print(f"No prediction files found under {args.predictions}", file=sys.stderr)
        return 2

    preds: list[InternalPrediction] = []
    for path in files:
        try:
            chunk = load_internal_predictions(
                path,
                match_id=args.explicit_match if path.is_file() and args.predictions.is_file() else None,
                half=args.explicit_half if path.is_file() and args.predictions.is_file() else None,
                fps=float(args.fps),
            )
        except Exception as exc:
            print(f"warning: skipping {path}: {exc!r}", file=sys.stderr)
            continue
        preds.extend(chunk)
        print(f"  loaded {len(chunk):>6} predictions from {path.name}")

    if not preds:
        print("No predictions loaded; refusing to write an empty submission.", file=sys.stderr)
        return 2

    by_match = group_predictions_by_match(preds)
    print(f"Matches: {len(by_match)}; total predictions: {len(preds)}")

    # Validate all documents before deciding to write anything; mirror
    # this in --report so CI can fail before any artifact is touched.
    docs: dict[str, dict] = {}
    report: dict[str, dict] = {}
    any_errors = False
    for match_id, preds_for_match in by_match.items():
        from pcspot.eval.submission import build_match_document
        doc = build_match_document(match_id, preds_for_match)
        errors = validate_submission_payload(doc)
        docs[match_id] = doc
        report[match_id] = {
            "predictions": len(preds_for_match),
            "errors": list(errors),
        }
        if errors:
            any_errors = True
            print(f"  ! {match_id}: {len(errors)} schema error(s)")
            for err in errors[:5]:
                print(f"      - {err}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote validation report -> {args.report}")

    if any_errors:
        print("Validation failures detected; not writing zip.", file=sys.stderr)
        return 1

    if args.validate_only:
        print("--validate-only set; skipping zip write.")
        return 0

    write_submission_zip(by_match, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
