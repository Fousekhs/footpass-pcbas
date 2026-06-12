"""Tests for ``pcspot.eval.submission`` and the Codabench CLI writer."""

from __future__ import annotations

import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pcspot.eval.submission import (  # noqa: E402
    InternalPrediction,
    SUBMISSION_FILE_NAME,
    build_match_document,
    group_predictions_by_match,
    load_internal_predictions,
    prediction_to_payload,
    validate_match_predictions,
    validate_submission_payload,
    write_submission_zip,
)


def _pred(match: str = "game_01", half: int = 1, frame: int = 25,
          team: int = 0, jersey_number: int = 10, class_id: int = 2,
          score: float = 0.9) -> InternalPrediction:
    return InternalPrediction(
        match_id=match,
        half=half,
        frame=frame,
        team=team,
        jersey_number=jersey_number,
        class_id=class_id,
        score=score,
    )


class FormattingTests(unittest.TestCase):
    def test_prediction_to_payload_is_positional_row(self) -> None:
        payload = prediction_to_payload(
            _pred(frame=1425, team=0, jersey_number=10, class_id=2, score=0.87)
        )
        # [frame, team, jersey_number, class_id, score]
        self.assertEqual(payload, [1425, 0, 10, 2, 0.87])


class GroupingTests(unittest.TestCase):
    def test_predictions_sorted_within_match(self) -> None:
        preds = [
            _pred(match="game_01", half=2, frame=50, team=1, jersey_number=4),
            _pred(match="game_01", half=1, frame=100, team=0, jersey_number=10),
            _pred(match="game_01", half=1, frame=50, team=0, jersey_number=7),
        ]
        by_match = group_predictions_by_match(preds)
        self.assertEqual(list(by_match.keys()), ["game_01"])
        sequence = [(p.half, p.frame, p.team, p.jersey_number) for p in by_match["game_01"]]
        # Sorted: (half, frame, team, jersey_number).
        self.assertEqual(
            sequence, [(1, 50, 0, 7), (1, 100, 0, 10), (2, 50, 1, 4)]
        )


class ValidationTests(unittest.TestCase):
    def _valid_doc(self) -> list[list]:
        return build_match_document([_pred()])

    def test_valid_doc_has_no_errors(self) -> None:
        self.assertEqual(validate_match_predictions(self._valid_doc()), [])
        self.assertEqual(
            validate_submission_payload({"game_01": self._valid_doc()}), []
        )

    def test_predictions_must_be_a_list(self) -> None:
        errs = validate_match_predictions({"not": "a list"})
        self.assertTrue(any("must be a list" in e for e in errs))

    def test_wrong_row_length_flagged(self) -> None:
        errs = validate_match_predictions([[25, 0, 10, 2]])  # missing score
        self.assertTrue(any("5 elements" in e for e in errs))

    def test_unknown_class_id_flagged(self) -> None:
        doc = self._valid_doc()
        doc[0][3] = 99
        errs = validate_match_predictions(doc)
        self.assertTrue(any("class_id" in e for e in errs))

    def test_background_class_id_flagged(self) -> None:
        doc = self._valid_doc()
        doc[0][3] = 0
        errs = validate_match_predictions(doc)
        self.assertTrue(any("class_id" in e for e in errs))

    def test_score_out_of_range_flagged(self) -> None:
        doc = self._valid_doc()
        doc[0][4] = 1.5
        errs = validate_match_predictions(doc)
        self.assertTrue(any("score" in e for e in errs))

    def test_team_must_be_zero_or_one(self) -> None:
        doc = self._valid_doc()
        doc[0][1] = 2
        errs = validate_match_predictions(doc)
        self.assertTrue(any("team" in e for e in errs))

    def test_jersey_number_must_be_integer(self) -> None:
        doc = self._valid_doc()
        doc[0][2] = "ten"
        errs = validate_match_predictions(doc)
        self.assertTrue(any("jersey_number" in e for e in errs))

    def test_submission_payload_must_be_a_mapping(self) -> None:
        errs = validate_submission_payload(["not", "a", "dict"])
        self.assertTrue(any("must be a dict" in e for e in errs))

    def test_submission_payload_prefixes_errors_with_match_id(self) -> None:
        doc = self._valid_doc()
        doc[0][1] = 2
        errs = validate_submission_payload({"game_01": doc})
        self.assertTrue(any(e.startswith("game_01: ") for e in errs))


class ZipWriterTests(unittest.TestCase):
    def test_zip_contains_predictions_json(self) -> None:
        preds = [
            _pred(match="game_01", half=1, frame=10, team=0, jersey_number=10),
            _pred(match="game_02", half=2, frame=50, team=1, jersey_number=4, class_id=7),
        ]
        by_match = group_predictions_by_match(preds)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.zip"
            payload = write_submission_zip(by_match, out)
            self.assertTrue(out.exists())
            self.assertEqual(sorted(payload.keys()), ["game_01", "game_02"])
            with zipfile.ZipFile(out, "r") as zf:
                self.assertEqual(zf.namelist(), [SUBMISSION_FILE_NAME])
                doc = json.loads(zf.read(SUBMISSION_FILE_NAME))
            self.assertEqual(sorted(doc.keys()), ["game_01", "game_02"])
            # Row: [frame, team, jersey_number, class_id, score]
            self.assertEqual(doc["game_01"][0][2], 10)  # jersey_number
            self.assertEqual(doc["game_02"][0][3], 7)   # class_id (Tackle)

    def test_writer_refuses_invalid_payload(self) -> None:
        # Build a manifestly invalid prediction (out-of-range team) and
        # ensure the writer rejects before producing a zip.
        bad = InternalPrediction(
            match_id="game_01", half=1, frame=10, team=9,
            jersey_number=10, class_id=2, score=0.5,
        )
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_submission_zip(
                    {"game_01": [bad]}, Path(tmp) / "submission.zip"
                )


class LoadInternalPredictionsTests(unittest.TestCase):
    def _write_infer_file(self, tmp: Path, name: str, entries: list[dict]) -> Path:
        path = tmp / name
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_loads_filename_inferred_match_and_half(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_infer_file(
                Path(tmp),
                "game_18__H2.json",
                [[12, 0, 10, 2, 0.7]],  # [frame, team, jersey, class_id, score]
            )
            preds = load_internal_predictions(path)
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].match_id, "game_18")
            self.assertEqual(preds[0].half, 2)
            self.assertEqual(preds[0].team, 0)
            self.assertEqual(preds[0].jersey_number, 10)
            self.assertEqual(preds[0].class_id, 2)

    def test_explicit_overrides_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_infer_file(
                Path(tmp),
                "weird_name.json",
                [[0, 1, 4, 7, 0.5]],
            )
            preds = load_internal_predictions(path, match_id="game_99", half=1)
            self.assertEqual(preds[0].match_id, "game_99")
            self.assertEqual(preds[0].half, 1)
            self.assertEqual(preds[0].class_id, 7)


class WriteCodabenchSubmissionCliTests(unittest.TestCase):
    def test_main_writes_zip_from_directory(self) -> None:
        import write_codabench_submission as wcs

        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp) / "preds"
            preds_dir.mkdir()
            for name, entries in [
                ("game_01__H1.json", [[10, 0, 10, 2, 0.9]]),   # H1: Pass
                ("game_01__H2.json", [[25, 1, 4, 3, 0.6]]),    # H2: Cross
            ]:
                (preds_dir / name).write_text(json.dumps(entries), encoding="utf-8")
            out = Path(tmp) / "submission.zip"
            report = Path(tmp) / "report.json"
            rc = wcs.main(
                [
                    "--predictions", str(preds_dir),
                    "--out", str(out),
                    "--report", str(report),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            self.assertTrue(report.exists())
            with zipfile.ZipFile(out, "r") as zf:
                doc = json.loads(zf.read(SUBMISSION_FILE_NAME))
            # Both halves should end up in the same match's prediction
            # list, ordered by (half, frame, team, jersey_number).
            class_ids = [row[3] for row in doc["game_01"]]
            self.assertEqual(class_ids, [2, 3])  # Pass (H1) before Cross (H2)

    def test_validate_only_does_not_write_zip(self) -> None:
        import write_codabench_submission as wcs

        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp) / "preds"
            preds_dir.mkdir()
            (preds_dir / "game_01__H1.json").write_text(
                json.dumps([[10, 0, 10, 2, 0.5]]),
                encoding="utf-8",
            )
            out = Path(tmp) / "submission.zip"
            rc = wcs.main(
                [
                    "--predictions", str(preds_dir),
                    "--out", str(out),
                    "--validate-only",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
