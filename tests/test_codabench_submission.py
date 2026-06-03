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
    _format_game_time,
    _format_position,
    build_match_document,
    group_predictions_by_match,
    load_internal_predictions,
    prediction_to_payload,
    validate_submission_payload,
    write_submission_zip,
)


def _pred(match: str = "game_01", half: int = 1, frame: int = 25,
          class_id: int = 2, player_id: int = 101, score: float = 0.9,
          fps: float = 25.0) -> InternalPrediction:
    return InternalPrediction(
        match_id=match,
        half=half,
        frame=frame,
        class_id=class_id,
        player_id=player_id,
        score=score,
        fps=fps,
    )


class FormattingTests(unittest.TestCase):
    def test_game_time_rounds_to_seconds(self) -> None:
        self.assertEqual(_format_game_time(1, 0.0), "1 - 00:00")
        self.assertEqual(_format_game_time(1, 62.4), "1 - 01:02")
        self.assertEqual(_format_game_time(2, 1530.7), "2 - 25:31")

    def test_position_is_milliseconds(self) -> None:
        self.assertEqual(_format_position(0.0), "0")
        self.assertEqual(_format_position(1.234), "1234")

    def test_prediction_to_payload_has_required_fields(self) -> None:
        payload = prediction_to_payload(_pred(frame=25, class_id=2, score=0.5))
        self.assertEqual(payload["label"], "Pass")
        self.assertEqual(payload["half"], 1)
        self.assertEqual(payload["gameTime"], "1 - 00:01")
        # 25 frames @ 25 fps = 1.0 s -> 1000 ms.
        self.assertEqual(payload["position"], "1000")
        self.assertEqual(payload["player_id"], 101)


class GroupingTests(unittest.TestCase):
    def test_predictions_sorted_within_match(self) -> None:
        preds = [
            _pred(match="game_01", half=2, frame=50, class_id=4, player_id=200),
            _pred(match="game_01", half=1, frame=100, class_id=2, player_id=101),
            _pred(match="game_01", half=1, frame=50, class_id=3, player_id=101),
        ]
        by_match = group_predictions_by_match(preds)
        self.assertEqual(list(by_match.keys()), ["game_01"])
        sequence = [(p.half, p.frame, p.class_id) for p in by_match["game_01"]]
        # Sorted: (half, frame, class_id, player_id).
        self.assertEqual(
            sequence, [(1, 50, 3), (1, 100, 2), (2, 50, 4)]
        )


class ValidationTests(unittest.TestCase):
    def _valid_doc(self) -> dict:
        return build_match_document("game_01", [_pred()])

    def test_valid_doc_has_no_errors(self) -> None:
        self.assertEqual(validate_submission_payload(self._valid_doc()), [])

    def test_missing_top_level_field_flagged(self) -> None:
        doc = self._valid_doc()
        del doc["UrlLocal"]
        self.assertIn("missing top-level field 'UrlLocal'", validate_submission_payload(doc))

    def test_bad_game_time_flagged(self) -> None:
        doc = self._valid_doc()
        doc["predictions"][0]["gameTime"] = "3 - 99:99"
        errs = validate_submission_payload(doc)
        self.assertTrue(any("gameTime" in e for e in errs))

    def test_unknown_label_flagged(self) -> None:
        doc = self._valid_doc()
        doc["predictions"][0]["label"] = "Bicycle Kick"
        errs = validate_submission_payload(doc)
        self.assertTrue(any("label" in e for e in errs))

    def test_confidence_out_of_range_flagged(self) -> None:
        doc = self._valid_doc()
        doc["predictions"][0]["confidence"] = "1.5"
        errs = validate_submission_payload(doc)
        self.assertTrue(any("confidence" in e for e in errs))

    def test_half_must_be_one_or_two(self) -> None:
        doc = self._valid_doc()
        doc["predictions"][0]["half"] = 3
        errs = validate_submission_payload(doc)
        self.assertTrue(any("half" in e for e in errs))


class ZipWriterTests(unittest.TestCase):
    def test_zip_contains_per_match_documents(self) -> None:
        preds = [
            _pred(match="game_01", half=1, frame=10),
            _pred(match="game_02", half=2, frame=50, class_id=3, player_id=205),
        ]
        by_match = group_predictions_by_match(preds)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.zip"
            docs = write_submission_zip(by_match, out)
            self.assertTrue(out.exists())
            self.assertEqual(sorted(docs.keys()), ["game_01", "game_02"])
            with zipfile.ZipFile(out, "r") as zf:
                names = sorted(zf.namelist())
                self.assertEqual(
                    names,
                    [f"game_01/{SUBMISSION_FILE_NAME}", f"game_02/{SUBMISSION_FILE_NAME}"],
                )
                payload = json.loads(zf.read(f"game_01/{SUBMISSION_FILE_NAME}"))
                self.assertEqual(payload["UrlLocal"], "game_01")
                self.assertEqual(payload["predictions"][0]["player_id"], 101)

    def test_writer_refuses_invalid_document(self) -> None:
        # Build a manifestly invalid prediction (out-of-range half) and
        # ensure the writer rejects before producing a zip.
        bad = InternalPrediction(
            match_id="game_01", half=9, frame=10, class_id=2,
            player_id=101, score=0.5,
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
                [
                    {"frame": 12, "class_id": 2, "player_id": 101, "score": 0.7,
                     "time_seconds": 0.48, "class_name": "Pass"},
                ],
            )
            preds = load_internal_predictions(path, fps=25.0)
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].match_id, "game_18")
            self.assertEqual(preds[0].half, 2)
            self.assertEqual(preds[0].fps, 25.0)

    def test_explicit_overrides_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_infer_file(
                Path(tmp),
                "weird_name.json",
                [{"frame": 0, "class_id": 1, "player_id": 100, "score": 0.5}],
            )
            preds = load_internal_predictions(
                path, match_id="game_99", half=1, fps=30.0
            )
            self.assertEqual(preds[0].match_id, "game_99")
            self.assertEqual(preds[0].half, 1)
            self.assertEqual(preds[0].fps, 30.0)


class WriteCodabenchSubmissionCliTests(unittest.TestCase):
    def test_main_writes_zip_from_directory(self) -> None:
        import write_codabench_submission as wcs

        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp) / "preds"
            preds_dir.mkdir()
            for name, entries in [
                ("game_01__H1.json",
                 [{"frame": 10, "class_id": 2, "player_id": 101, "score": 0.9}]),
                ("game_01__H2.json",
                 [{"frame": 25, "class_id": 3, "player_id": 202, "score": 0.6}]),
            ]:
                (preds_dir / name).write_text(json.dumps(entries), encoding="utf-8")
            out = Path(tmp) / "submission.zip"
            report = Path(tmp) / "report.json"
            rc = wcs.main(
                [
                    "--predictions", str(preds_dir),
                    "--out", str(out),
                    "--report", str(report),
                    "--fps", "25",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            self.assertTrue(report.exists())
            with zipfile.ZipFile(out, "r") as zf:
                payload = json.loads(zf.read(f"game_01/{SUBMISSION_FILE_NAME}"))
            # Both halves should end up in the same Labels-ball.json
            # ordered by (half, frame, class_id, player_id).
            halves = [p["half"] for p in payload["predictions"]]
            self.assertEqual(halves, [1, 2])

    def test_validate_only_does_not_write_zip(self) -> None:
        import write_codabench_submission as wcs

        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp) / "preds"
            preds_dir.mkdir()
            (preds_dir / "game_01__H1.json").write_text(
                json.dumps(
                    [{"frame": 10, "class_id": 2, "player_id": 101, "score": 0.5}]
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "submission.zip"
            rc = wcs.main(
                [
                    "--predictions", str(preds_dir),
                    "--out", str(out),
                    "--fps", "25",
                    "--validate-only",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
