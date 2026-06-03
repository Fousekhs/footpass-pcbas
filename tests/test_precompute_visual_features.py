"""Tests for ``scripts/precompute_visual_features.py``.

Focuses on the new split/match-list filtering and resumability hooks.
The torch-heavy DINOv2 forward path is exercised separately in
``tests/test_visual_dinov2.py``; here we stay in pure-Python land by
using --dry-run and synthetic fixtures.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import precompute_visual_features as pvf  # noqa: E402
from pcbas_data import MatchAssets  # noqa: E402


def _write_h5_with_keys(path: Path, keys: list[str]) -> None:
    """Write a minimal HDF5 file containing one dataset per key."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        for k in keys:
            f.create_dataset(k, data=np.zeros((1, 14), dtype=np.float32))


def _make_split_layout(root: Path, split: str, matches: list[str]) -> None:
    """Lay out an extracted/ tree mimicking what the full mirror produces."""
    split_l = split.lower()
    tactical = root / "extracted" / f"tactical_data_{split_l}"
    videos = root / "extracted" / f"videos_fullhd_{split_l}"
    tactical.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    keys = [f"{m}_H1" for m in matches]
    _write_h5_with_keys(tactical / f"{split_l}_tactical_data.h5", keys)
    for m in matches:
        (videos / f"{m}.mp4").write_bytes(b"\0")


def _write_config(tmp: Path, output_dir: Path) -> Path:
    cfg = tmp / "config.toml"
    cfg.write_text(
        f"""
[pcbas]
huggingface_repo = "SoccerNet/SN-PCBAS-2026"
output_dir = {json.dumps(str(output_dir))}
""".strip(),
        encoding="utf-8",
    )
    return cfg


class ParseSplitsTests(unittest.TestCase):
    def test_none_passes_through(self) -> None:
        self.assertIsNone(pvf.parse_splits(None))

    def test_csv_normalises_to_uppercase_unique(self) -> None:
        self.assertEqual(
            pvf.parse_splits("train, val,train"), ["TRAIN", "VAL"]
        )

    def test_invalid_split_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            pvf.parse_splits("TRAIN,FOO")

    def test_empty_csv_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            pvf.parse_splits(", ,")


class LoadMatchListTests(unittest.TestCase):
    def test_ignores_blanks_and_comments(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.txt"
            p.write_text("# comment\n\ngame_01\ngame_02\n\n", encoding="utf-8")
            self.assertEqual(pvf.load_match_list(p), {"game_01", "game_02"})

    def test_empty_file_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.txt"
            p.write_text("# nothing here\n\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                pvf.load_match_list(p)


class DiscoverMatchesInSplitTests(unittest.TestCase):
    def test_discovers_matches_for_split(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_split_layout(root, "TRAIN", ["game_01", "game_02"])
            _make_split_layout(root, "VAL", ["game_18"])
            train_matches = pvf.discover_matches_in_split(root, "TRAIN")
            self.assertEqual(
                sorted(m.match_id for m in train_matches),
                ["game_01", "game_02"],
            )
            val_matches = pvf.discover_matches_in_split(root, "VAL")
            self.assertEqual([m.match_id for m in val_matches], ["game_18"])

    def test_missing_split_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_split_layout(root, "TRAIN", ["game_01"])
            self.assertEqual(
                pvf.discover_matches_in_split(root, "CHALLENGE"), []
            )


class SelectMatchesTests(unittest.TestCase):
    def test_splits_label_is_attached(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_split_layout(root, "TRAIN", ["game_01", "game_02"])
            _make_split_layout(root, "VAL", ["game_18"])
            pairs = pvf._select_matches(
                root, match_id=None, splits=["TRAIN", "VAL"]
            )
            labels = {m.match_id: s for s, m in pairs}
            self.assertEqual(labels["game_01"], "TRAIN")
            self.assertEqual(labels["game_18"], "VAL")

    def test_match_list_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_split_layout(root, "TRAIN", ["game_01", "game_02"])
            pairs = pvf._select_matches(
                root,
                match_id=None,
                splits=["TRAIN"],
                match_list={"game_02"},
            )
            self.assertEqual([m.match_id for _, m in pairs], ["game_02"])

    def test_unmatched_match_list_aborts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_split_layout(root, "TRAIN", ["game_01"])
            with self.assertRaises(SystemExit):
                pvf._select_matches(
                    root,
                    match_id=None,
                    splits=["TRAIN"],
                    match_list={"game_99"},
                )


class ShardExistsTests(unittest.TestCase):
    def test_detects_existing_shard(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "game_01__game_01_H1.npz").write_bytes(b"\0")
            self.assertTrue(pvf._shard_exists(root, "game_01", "game_01_H1"))
            self.assertFalse(pvf._shard_exists(root, "game_01", "game_01_H2"))


class ParseArgsNewFlagsTests(unittest.TestCase):
    def test_fp16_flag_accepted(self) -> None:
        args = pvf.parse_args(["--config", "config.toml", "--fp16"])
        self.assertTrue(args.fp16)

    def test_compile_flag_accepted(self) -> None:
        args = pvf.parse_args(["--config", "config.toml", "--compile"])
        self.assertTrue(args.compile)

    def test_prefetch_queue_size_accepted(self) -> None:
        args = pvf.parse_args(["--config", "config.toml", "--prefetch-queue-size", "4"])
        self.assertEqual(args.prefetch_queue_size, 4)

    def test_num_workers_accepted(self) -> None:
        args = pvf.parse_args(["--config", "config.toml", "--num-workers", "4"])
        self.assertEqual(args.num_workers, 4)

    def test_defaults(self) -> None:
        args = pvf.parse_args(["--config", "config.toml"])
        self.assertFalse(args.fp16)
        self.assertFalse(args.compile)
        self.assertEqual(args.prefetch_queue_size, 8)
        self.assertEqual(args.num_workers, 1)


class PrefetchDecodeWorkerTests(unittest.TestCase):
    def _make_fake_cap(self, frames: list):
        """Minimal cv2.VideoCapture stand-in."""
        class _FakeCap:
            def __init__(self, frames):
                self._frames = list(frames)
                self._idx = 0
            def read(self):
                if self._idx >= len(self._frames):
                    return False, None
                f = self._frames[self._idx]
                self._idx += 1
                return True, f
            def release(self):
                pass
        return _FakeCap(frames)

    def test_worker_emits_sentinel_after_all_frames(self) -> None:
        import queue as q_mod
        import cv2
        # Build 3 tiny RGB frames (fake BGR→RGB is a no-op for this test).
        frames_bgr = [
            np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)
        ]
        cap = self._make_fake_cap(frames_bgr)

        # Patch _read_frame_bgr_to_rgb to just return the "frame" directly.
        with mock.patch.object(pvf, "_read_frame_bgr_to_rgb", side_effect=lambda c: c.read()[1]):
            from pcspot.features.cropper import CropperConfig, PaddedPlayerCropper
            cropper = PaddedPlayerCropper(CropperConfig(crop_size=8, pad_factor=1.0))
            out_q = q_mod.Queue()
            by_frame: dict = {}
            pvf._prefetch_decode_crop(cap, by_frame, cropper, 0, 2, out_q)

        items = []
        while True:
            item = out_q.get_nowait()
            if item is None:
                break
            items.append(item)

        self.assertEqual(len(items), 3)
        self.assertEqual([it["frame"] for it in items], [0, 1, 2])
        self.assertTrue(all(it["n_valid"] == 0 for it in items))


class ProcessHalfTaskTests(unittest.TestCase):
    """Exercise _process_half_task (the parallel worker) directly."""

    def test_worker_produces_shard(self) -> None:
        import argparse
        from tempfile import TemporaryDirectory
        import numpy as np

        from pcspot.features.cache import VisualFeatureMetadata

        with TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "cache"

            # Tiny synthetic tactical array: 4 frames × 2 players, 14 columns.
            # Columns: frame, player_id, l2r, shirt, role, x, y, sx, sy,
            #           roi_x, roi_y, roi_w, roi_h, class
            rows = []
            for frame in range(4):
                for pid in [101, 102]:
                    rows.append([
                        frame, pid, 1.0, 1, 1,
                        0.5, 0.5, 0.0, 0.0,
                        200.0, 200.0, 80.0, 160.0,
                        0.0,
                    ])
            arr = np.array(rows, dtype=np.float32)

            metadata = VisualFeatureMetadata(
                backbone_name="dinov2_vits14_stub",
                feature_dim=8,
                crop_size=32,
                pad_factor=1.6,
                fullhd_width=1920,
                fullhd_height=1080,
            )

            payload = {
                "match_id": "game_test",
                "half_id": "game_test_H1",
                "split": "TRAIN",
                "arr": arr,
                "video_path": "irrelevant.mp4",
                "backbone": "dinov2_vits14",
                "crop_size": 32,
                "pad_factor": 1.6,
                "min_box_size": 8,
                "batch_size": 8,
                "device": "cpu",
                "use_stub": True,
                "fp16": False,
                "compile_model": False,
                "prefetch_queue_size": 2,
                "start_frame": None,
                "end_frame": None,
                "max_frames": None,
                "out_root": str(out_root),
                "metadata": metadata.to_dict(),
            }

            # Patch video reading so the worker doesn't need a real file.
            fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            with mock.patch.object(pvf, "_open_video", return_value=mock.MagicMock()), \
                 mock.patch.object(pvf, "_seek"), \
                 mock.patch.object(pvf, "_read_frame_bgr_to_rgb",
                                   side_effect=[fake_frame] * 4 + [None]):
                result = pvf._process_half_task(payload)

            self.assertEqual(result["status"], "written")
            self.assertEqual(result["frames"], 4)
            self.assertGreater(len(result["shards"]), 0)


class MainDryRunTests(unittest.TestCase):
    def test_dry_run_records_planned_matches_without_torch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            _make_split_layout(data_dir, "TRAIN", ["game_01", "game_02"])
            _make_split_layout(data_dir, "VAL", ["game_18"])
            cfg = _write_config(root, data_dir)
            out_dir = root / "cache"

            # No torch shouldn't be required in dry-run mode; assert by
            # patching DinoV2Extractor to a sentinel that would explode
            # if instantiated.
            with mock.patch(
                "pcspot.features.dinov2.DinoV2Extractor",
                side_effect=AssertionError("DINOv2 should not be loaded in --dry-run"),
            ):
                rc = pvf.main(
                    [
                        "--config", str(cfg),
                        "--splits", "TRAIN,VAL",
                        "--out-dir", str(out_dir),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 0)
            backbone_root = out_dir / "dinov2_vits14"
            manifest_path = backbone_root / "manifest.json"
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            statuses = [r["status"] for r in payload["shards"]]
            self.assertTrue(all(s == "would-process" for s in statuses))
            match_ids = sorted(r["match_id"] for r in payload["shards"])
            self.assertEqual(match_ids, ["game_01", "game_02", "game_18"])


if __name__ == "__main__":
    unittest.main()
