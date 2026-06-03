"""Tests for the W&B logging and validation helpers in ``scripts/train.py``.

These tests purposely avoid any real network or W&B SDK dependency: a
``_FakeWandbRun`` captures the call arguments that ``make_log_fn``
would otherwise send to the W&B client.

The validation helper is exercised end-to-end against a tiny
``PlayerCentricSpottingModel`` and a synthetic single-half dataset so
the prediction-decoding, NMS, and metric pipeline is also covered.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_cli  # noqa: E402  (script imported as a module)

from pcspot.data.dataset import PCBASDataset  # noqa: E402
from pcspot.data.loader import HalfArray  # noqa: E402
from pcspot.data.schema import EventLabel  # noqa: E402
from pcspot.data.splits import SplitManifest  # noqa: E402
from pcspot.data.targets import CalfConfig  # noqa: E402
from pcspot.models.pipeline import PlayerCentricSpottingModel  # noqa: E402
from pcspot.train.trainer import EpochLog, Trainer, TrainStepLog  # noqa: E402


class _FakeWandbRun:
    """Minimal stand-in for a wandb run that records ``log`` calls."""

    def __init__(self) -> None:
        self.id = "fake-run-id"
        self.calls: list[tuple[dict, int | None]] = []
        self.finished = False

    def log(self, payload, step=None):
        self.calls.append((dict(payload), step))

    def finish(self):
        self.finished = True


def _synthetic_half(
    *,
    match_id: str = "m",
    half_id: str = "H1",
    num_frames: int = 24,
    num_players: int = 4,
    events: list[tuple[int, int, int]] | None = None,
) -> HalfArray:
    """Build a small (N, 14) tactical array with optional events.

    ``events`` is a list of ``(frame, player_id, class_id)`` triples.
    """
    events = events or []
    rows = []
    for frame in range(num_frames):
        for p in range(num_players):
            pid = 101 + p if p < num_players // 2 else 201 + (p - num_players // 2)
            cls = 0
            for ef, ep, ec in events:
                if frame == ef and pid == ep:
                    cls = ec
            rows.append(
                [
                    frame,
                    pid,
                    1.0,
                    p % 11 + 1,
                    1,
                    0.5 + 0.01 * p,
                    0.5,
                    0.0,
                    0.0,
                    100.0,
                    100.0,
                    50.0,
                    50.0,
                    cls,
                ]
            )
    arr = np.asarray(rows, dtype=np.float32)
    return HalfArray(match_id=match_id, half_id=half_id, array=arr)


class ParseTolerancesTests(unittest.TestCase):
    def test_parses_csv(self) -> None:
        self.assertEqual(train_cli._parse_tolerances("1, 2,3"), [1, 2, 3])

    def test_empty_raises(self) -> None:
        with self.assertRaises(SystemExit):
            train_cli._parse_tolerances("")
        with self.assertRaises(SystemExit):
            train_cli._parse_tolerances(", , ")

    def test_non_positive_raises(self) -> None:
        with self.assertRaises(SystemExit):
            train_cli._parse_tolerances("0,3")
        with self.assertRaises(SystemExit):
            train_cli._parse_tolerances("-1,2")


class GtEventsFromHalfTests(unittest.TestCase):
    def test_extracts_events(self) -> None:
        half = _synthetic_half(
            num_frames=20,
            num_players=4,
            events=[(5, 101, 2), (12, 202, 3)],
        )
        events = train_cli._gt_events_for_half(half)
        self.assertEqual(len(events), 2)
        kinds = sorted((e.frame, e.player_id, e.class_id) for e in events)
        self.assertEqual(kinds, [(5, 101, 2), (12, 202, 3)])

    def test_empty_half(self) -> None:
        empty = HalfArray(match_id="m", half_id="H1", array=np.zeros((0, 14), dtype=np.float32))
        self.assertEqual(train_cli._gt_events_for_half(empty), [])


class FilterHalvesTests(unittest.TestCase):
    def test_filter_picks_split_halves_and_reports_missing(self) -> None:
        h1 = _synthetic_half(match_id="m", half_id="H1")
        h2 = _synthetic_half(match_id="m", half_id="H2")
        manifest = SplitManifest.from_dict(
            {
                "train": [("m", "H1"), ("m", "Hmissing")],
                "val": [("m", "H2")],
            }
        )
        train_halves, missing = train_cli._filter_halves([h1, h2], manifest, "train")
        self.assertEqual({(h.match_id, h.half_id) for h in train_halves}, {("m", "H1")})
        self.assertEqual(missing, {("m", "Hmissing")})

        val_halves, missing_val = train_cli._filter_halves([h1, h2], manifest, "val")
        self.assertEqual({(h.match_id, h.half_id) for h in val_halves}, {("m", "H2")})
        self.assertEqual(missing_val, set())


class MakeLogFnTests(unittest.TestCase):
    def test_stdout_only_when_no_wandb(self) -> None:
        lines: list[str] = []
        log = train_cli.make_log_fn(wandb_run=None, print_fn=lines.append)
        log(TrainStepLog(
            step=3, total_loss=0.4, bce_loss=0.3, tmse_loss=0.05,
            objectness_loss=0.05, learning_rate=1e-3,
        ))
        log(EpochLog(
            epoch=0, num_steps=3,
            avg_total_loss=0.4, avg_bce_loss=0.3,
            avg_tmse_loss=0.05, avg_objectness_loss=0.05,
            last_learning_rate=1e-3, validation={"val/map": 0.6},
        ))
        self.assertEqual(len(lines), 2)
        self.assertIn("step=", lines[0])
        self.assertIn("epoch", lines[1])

    def test_wandb_step_log_keys(self) -> None:
        run = _FakeWandbRun()
        log = train_cli.make_log_fn(wandb_run=run, print_fn=lambda _s: None)
        log(TrainStepLog(
            step=7, total_loss=0.4, bce_loss=0.3, tmse_loss=0.05,
            objectness_loss=0.05, learning_rate=1e-3,
        ))
        self.assertEqual(len(run.calls), 1)
        payload, step = run.calls[0]
        self.assertEqual(step, 7)
        self.assertEqual(
            set(payload.keys()),
            {
                "train/loss_total",
                "train/loss_bce",
                "train/loss_tmse",
                "train/loss_objectness",
                "train/learning_rate",
                "train/step",
            },
        )
        self.assertAlmostEqual(payload["train/loss_total"], 0.4)

    def test_wandb_epoch_log_uses_last_step_and_merges_validation(self) -> None:
        run = _FakeWandbRun()
        log = train_cli.make_log_fn(wandb_run=run, print_fn=lambda _s: None)
        log(TrainStepLog(
            step=10, total_loss=1.0, bce_loss=0.5, tmse_loss=0.3,
            objectness_loss=0.2, learning_rate=5e-4,
        ))
        log(EpochLog(
            epoch=2, num_steps=11,
            avg_total_loss=0.9, avg_bce_loss=0.45,
            avg_tmse_loss=0.25, avg_objectness_loss=0.2,
            last_learning_rate=5e-4,
            validation={"val/map": 0.75, "val/map_joint": 0.55},
        ))
        self.assertEqual(len(run.calls), 2)
        epoch_payload, step = run.calls[1]
        self.assertEqual(step, 10, "epoch log must reuse last train step for monotonicity")
        self.assertEqual(epoch_payload["epoch"], 2)
        self.assertAlmostEqual(epoch_payload["train/avg_loss_total"], 0.9)
        self.assertAlmostEqual(epoch_payload["val/map"], 0.75)
        self.assertAlmostEqual(epoch_payload["val/map_joint"], 0.55)

    def test_wandb_epoch_log_drops_non_numeric_validation(self) -> None:
        run = _FakeWandbRun()
        log = train_cli.make_log_fn(wandb_run=run, print_fn=lambda _s: None)
        log(EpochLog(
            epoch=0, num_steps=1,
            avg_total_loss=0.1, avg_bce_loss=0.1,
            avg_tmse_loss=0.0, avg_objectness_loss=0.0,
            last_learning_rate=1e-3,
            validation={"val/map": 0.5, "bad": [1, 2, 3]},  # type: ignore[dict-item]
        ))
        self.assertEqual(len(run.calls), 1)
        payload, _ = run.calls[0]
        self.assertIn("val/map", payload)
        self.assertNotIn("bad", payload)


class MakeValidationFnTests(unittest.TestCase):
    def _build_setup(self):
        torch.manual_seed(0)
        np.random.seed(0)
        half = _synthetic_half(
            num_frames=24,
            num_players=4,
            events=[(8, 101, 2), (15, 202, 3)],
        )
        manifest = SplitManifest.from_dict({"val": [("m", "H1")]})
        ds_val = PCBASDataset(
            manifest=manifest,
            split="val",
            window_size=8,
            stride=8,
            halves=[half],
            calf_config=CalfConfig(),
            compute_targets=False,
        )
        model = PlayerCentricSpottingModel(
            hidden_dim=8,
            num_hgt_layers=1,
            num_heads=2,
            num_mstcn_stages=1,
            num_mstcn_layers=2,
            knn=2,
        )
        trainer = Trainer(model, learning_rate=1e-3)
        gt = {("m", "H1"): train_cli._gt_events_for_half(half)}
        return ds_val, trainer, gt, model.num_classes

    def test_returns_expected_metric_keys(self) -> None:
        ds_val, trainer, gt, num_classes = self._build_setup()
        validate = train_cli.make_validation_fn(
            val_dataset=ds_val,
            val_gt_per_half=gt,
            num_classes=num_classes,
            decode_threshold=0.0,  # accept everything so we always get preds
            nms_radius=2,
            nms_mode="per_player_class",
            tolerances=[3, 12],
            device="cpu",
        )
        out = validate(trainer)
        for tol in (3, 12):
            self.assertIn(f"val/map_at_t{tol}", out)
            self.assertIn(f"val/map_joint_at_t{tol}", out)
            self.assertIn(f"val/player_id_acc_at_t{tol}", out)
        self.assertIn("val/map", out)
        self.assertIn("val/map_joint", out)
        self.assertIn("val/player_identity_accuracy", out)
        self.assertIn("val/num_predictions", out)
        self.assertIn("val/num_ground_truth", out)
        self.assertEqual(out["val/num_ground_truth"], 2.0)
        self.assertGreater(out["val/num_predictions"], 0.0)

    def test_rejects_empty_tolerances(self) -> None:
        ds_val, _, gt, num_classes = self._build_setup()
        with self.assertRaises(ValueError):
            train_cli.make_validation_fn(
                val_dataset=ds_val,
                val_gt_per_half=gt,
                num_classes=num_classes,
                decode_threshold=0.5,
                nms_radius=2,
                nms_mode="per_player_class",
                tolerances=[],
                device="cpu",
            )

    def test_returns_zero_metrics_with_no_predictions(self) -> None:
        ds_val, trainer, gt, num_classes = self._build_setup()
        validate = train_cli.make_validation_fn(
            val_dataset=ds_val,
            val_gt_per_half=gt,
            num_classes=num_classes,
            decode_threshold=10.0,  # impossibly high -> no predictions survive
            nms_radius=2,
            nms_mode="per_player_class",
            tolerances=[3],
            device="cpu",
        )
        out = validate(trainer)
        self.assertEqual(out["val/num_predictions"], 0.0)
        self.assertEqual(out["val/map"], 0.0)
        self.assertEqual(out["val/map_joint"], 0.0)


class ArgParserSmokeTests(unittest.TestCase):
    def test_wandb_and_validation_flags_present(self) -> None:
        parser = train_cli._build_argparser()
        args = parser.parse_args(
            [
                "--splits", "s.json",
                "--output-dir", "o",
                "--validation-split", "val",
                "--wandb",
                "--wandb-project", "pcspot",
                "--wandb-tags", "a,b",
                "--metric-tolerances", "3,12",
            ]
        )
        self.assertTrue(args.wandb)
        self.assertEqual(args.wandb_project, "pcspot")
        self.assertEqual(args.wandb_tags, "a,b")
        self.assertEqual(args.validation_split, "val")
        self.assertEqual(args.metric_tolerances, "3,12")

    def test_objectness_default_is_true_and_no_objectness_flag_disables(self) -> None:
        parser = train_cli._build_argparser()
        args_default = parser.parse_args(["--splits", "s.json", "--output-dir", "o"])
        self.assertTrue(args_default.objectness)

        args_off = parser.parse_args(
            ["--splits", "s.json", "--output-dir", "o", "--no-objectness"]
        )
        self.assertFalse(args_off.objectness)

    def test_sampler_defaults_to_sequential(self) -> None:
        parser = train_cli._build_argparser()
        args = parser.parse_args(["--splits", "s.json", "--output-dir", "o"])
        self.assertEqual(args.sampler, "sequential")
        self.assertEqual(args.positive_ratio, 0.7)
        self.assertEqual(args.event_weighting, "uniform")
        self.assertIsNone(args.samples_per_epoch)
        self.assertEqual(args.grad_accum_steps, 1)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bf16")

    def test_sampler_flags_round_trip(self) -> None:
        parser = train_cli._build_argparser()
        args = parser.parse_args(
            [
                "--splits", "s.json", "--output-dir", "o",
                "--sampler", "mixed",
                "--positive-ratio", "0.5",
                "--event-weighting", "linear",
                "--samples-per-epoch", "200",
                "--sampler-seed", "13",
                "--grad-accum-steps", "4",
                "--amp",
                "--amp-dtype", "fp16",
            ]
        )
        self.assertEqual(args.sampler, "mixed")
        self.assertAlmostEqual(args.positive_ratio, 0.5)
        self.assertEqual(args.event_weighting, "linear")
        self.assertEqual(args.samples_per_epoch, 200)
        self.assertEqual(args.sampler_seed, 13)
        self.assertEqual(args.grad_accum_steps, 4)
        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "fp16")

    def test_resume_flag_parses_path(self) -> None:
        parser = train_cli._build_argparser()
        args = parser.parse_args(
            [
                "--splits", "s.json", "--output-dir", "o",
                "--resume", "checkpoints/run1/latest.pt",
            ]
        )
        self.assertEqual(args.resume, Path("checkpoints/run1/latest.pt"))


def _write_train_config(tmp_dir: Path, body: str) -> Path:
    path = tmp_dir / "train.toml"
    path.write_text(body, encoding="utf-8")
    return path


class LoadTrainConfigTests(unittest.TestCase):
    def test_maps_all_three_sections_to_argparse_dests(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_train_config(
                Path(tmp),
                """
                [train]
                epochs = 7
                batch_size = 8
                learning_rate = 5e-4
                window_size = 96
                no_objectness = true
                visual_backbone = "dinov2_vitb14"

                [validation]
                validation_split = "val"
                decode_threshold = 0.25
                nms_radius = 8
                metric_tolerances = [3, 12, 25]

                [wandb]
                enabled = true
                project = "pcspot"
                entity = "team-a"
                tags = ["pcbas", "baseline"]
                mode = "offline"
                log_artifacts = true
                """,
            )
            out = train_cli._load_train_config(path)

        self.assertEqual(out["epochs"], 7)
        self.assertEqual(out["batch_size"], 8)
        self.assertAlmostEqual(out["learning_rate"], 5e-4)
        self.assertEqual(out["window_size"], 96)
        # no_objectness=true means objectness=False after inversion.
        self.assertFalse(out["objectness"])
        self.assertNotIn("no_objectness", out)
        self.assertEqual(out["visual_backbone"], "dinov2_vitb14")

        self.assertEqual(out["validation_split"], "val")
        self.assertEqual(out["decode_threshold"], 0.25)
        self.assertEqual(out["nms_radius"], 8)
        self.assertEqual(out["metric_tolerances"], "3,12,25")

        self.assertTrue(out["wandb"])
        self.assertEqual(out["wandb_project"], "pcspot")
        self.assertEqual(out["wandb_entity"], "team-a")
        self.assertEqual(out["wandb_tags"], "pcbas,baseline")
        self.assertEqual(out["wandb_mode"], "offline")
        self.assertTrue(out["wandb_log_artifacts"])

    def test_unknown_section_aborts(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_train_config(
                Path(tmp),
                """
                [bogus]
                key = 1
                """,
            )
            with self.assertRaises(SystemExit):
                train_cli._load_train_config(path)

    def test_unknown_key_aborts(self) -> None:
        import tempfile
        for section, key in (
            ("train", "epoxs"),
            ("validation", "tolerance"),
            ("wandb", "api_key"),
        ):
            with self.subTest(section=section, key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    path = _write_train_config(
                        Path(tmp), f"[{section}]\n{key} = 1\n"
                    )
                    with self.assertRaises(SystemExit):
                        train_cli._load_train_config(path)

    def test_missing_file_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            train_cli._load_train_config(Path("does/not/exist.toml"))

    def test_invalid_toml_aborts(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_train_config(Path(tmp), "this is not = valid [toml")
            with self.assertRaises(SystemExit):
                train_cli._load_train_config(path)

    def test_sampler_and_amp_keys_round_trip(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_train_config(
                Path(tmp),
                """
                [train]
                sampler = "mixed"
                positive_ratio = 0.6
                event_weighting = "linear"
                samples_per_epoch = 400
                sampler_seed = 21
                grad_accum_steps = 8
                amp = true
                amp_dtype = "bf16"
                """,
            )
            out = train_cli._load_train_config(path)
        self.assertEqual(out["sampler"], "mixed")
        self.assertAlmostEqual(out["positive_ratio"], 0.6)
        self.assertEqual(out["event_weighting"], "linear")
        self.assertEqual(out["samples_per_epoch"], 400)
        self.assertEqual(out["sampler_seed"], 21)
        self.assertEqual(out["grad_accum_steps"], 8)
        self.assertTrue(out["amp"])
        self.assertEqual(out["amp_dtype"], "bf16")


class TrainConfigPrecedenceTests(unittest.TestCase):
    """Built-in defaults < --train-config TOML < explicit CLI flags."""

    def _parse(self, defaults, extra_argv):
        parser = train_cli._build_argparser(defaults=defaults)
        argv = ["--splits", "s.json", "--output-dir", "o"] + list(extra_argv)
        return parser.parse_args(argv)

    def test_builtin_default_when_neither_file_nor_cli(self) -> None:
        args = self._parse(defaults={}, extra_argv=[])
        self.assertEqual(args.epochs, 5)
        self.assertFalse(args.wandb)
        self.assertTrue(args.objectness)

    def test_file_default_used_when_cli_absent(self) -> None:
        defaults = {
            "epochs": 11,
            "wandb": True,
            "wandb_project": "from-file",
            "objectness": False,
            "metric_tolerances": "3,12",
        }
        args = self._parse(defaults=defaults, extra_argv=[])
        self.assertEqual(args.epochs, 11)
        self.assertTrue(args.wandb)
        self.assertEqual(args.wandb_project, "from-file")
        self.assertFalse(args.objectness)
        self.assertEqual(args.metric_tolerances, "3,12")

    def test_cli_overrides_file_default(self) -> None:
        defaults = {
            "epochs": 11,
            "wandb": True,
            "wandb_project": "from-file",
            "objectness": False,
        }
        args = self._parse(
            defaults=defaults,
            extra_argv=[
                "--epochs", "3",
                "--no-wandb",
                "--wandb-project", "from-cli",
                "--objectness",
            ],
        )
        self.assertEqual(args.epochs, 3)
        self.assertFalse(args.wandb)
        self.assertEqual(args.wandb_project, "from-cli")
        self.assertTrue(args.objectness)


class PreparseTrainConfigTests(unittest.TestCase):
    def test_returns_none_when_absent(self) -> None:
        self.assertIsNone(
            train_cli._preparse_train_config(
                ["--splits", "s.json", "--output-dir", "o"]
            )
        )

    def test_picks_up_path_anywhere_in_argv(self) -> None:
        out = train_cli._preparse_train_config(
            [
                "--splits", "s.json",
                "--train-config", "configs/train/baseline.toml",
                "--output-dir", "o",
            ]
        )
        self.assertEqual(out, Path("configs/train/baseline.toml"))


if __name__ == "__main__":
    unittest.main()
