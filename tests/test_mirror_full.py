"""Tests for ``scripts/mirror_pcbas_full.py``.

The tests stub out the Hugging Face network calls and exercise:

- CLI argument parsing (splits, resolutions, mutually exclusive flags).
- File selection (fullHD default, --include-lowres union).
- Skip-existing behavior (same-size local file is skipped).
- Dry-run does not download and produces a manifest with no
  ``downloaded`` files.
- Per-split and top-level manifests are written and contain the expected
  structure (without ever logging credentials).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mirror_pcbas_full as mpf  # noqa: E402
from mirror_pcbas_one_match import Config, FileEntry  # noqa: E402


def _write_config(tmp: Path, output_dir: Path) -> Path:
    cfg = tmp / "config.toml"
    cfg.write_text(
        f"""
[pcbas]
huggingface_repo = "SoccerNet/SN-PCBAS-2026"
huggingface_token = "TEST_TOKEN_DO_NOT_LOG"
soccernet_password = "TEST_PASSWORD_DO_NOT_LOG"
output_dir = {json.dumps(str(output_dir))}
""".strip(),
        encoding="utf-8",
    )
    return cfg


def _fake_entries() -> list[FileEntry]:
    """Realistic-shaped subset of the SN-PCBAS-2026 repo listing."""
    return [
        FileEntry(repo_path="tactical_data_format.txt", size=2_000),
        FileEntry(repo_path="README.md", size=1_500),
        FileEntry(repo_path="tactical_data_TRAIN.zip", size=10_000),
        FileEntry(repo_path="tactical_data_VAL.zip", size=4_000),
        FileEntry(repo_path="tactical_data_CHALLENGE.zip", size=4_000),
        FileEntry(repo_path="videos_fullHD_TRAIN_01.zip", size=1_000_000),
        FileEntry(repo_path="videos_fullHD_TRAIN_02.zip", size=1_000_000),
        FileEntry(repo_path="videos_fullHD_VAL.zip", size=500_000),
        FileEntry(repo_path="videos_fullHD_CHALLENGE.zip", size=500_000),
        FileEntry(repo_path="videos_352x640_TRAIN.zip", size=200_000),
        FileEntry(repo_path="videos_352x640_VAL.zip", size=100_000),
        FileEntry(repo_path="videos_352x640_CHALLENGE.zip", size=100_000),
    ]


class ParseSplitsTests(unittest.TestCase):
    def test_default_is_all_three(self) -> None:
        ns = mpf.parse_args(["--config", "x"])
        self.assertEqual(mpf.parse_splits(ns.splits), ["TRAIN", "VAL", "CHALLENGE"])

    def test_csv_is_uppercased_and_deduped(self) -> None:
        self.assertEqual(
            mpf.parse_splits("train, val, val,train"),
            ["TRAIN", "VAL"],
        )

    def test_invalid_split_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            mpf.parse_splits("TRAIN,BOGUS")

    def test_empty_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            mpf.parse_splits(", ,")


class ResolutionsForTests(unittest.TestCase):
    def test_default_is_fullhd_only(self) -> None:
        ns = mpf.parse_args(["--config", "x"])
        self.assertEqual(mpf.resolutions_for(ns), ["fullHD"])

    def test_include_lowres_appends(self) -> None:
        ns = mpf.parse_args(["--config", "x", "--include-lowres"])
        self.assertEqual(mpf.resolutions_for(ns), ["fullHD", "352x640"])

    def test_lowres_primary_with_include_lowres_is_idempotent(self) -> None:
        ns = mpf.parse_args(
            ["--config", "x", "--resolution", "352x640", "--include-lowres"]
        )
        self.assertEqual(mpf.resolutions_for(ns), ["352x640"])


class SelectSplitFilesTests(unittest.TestCase):
    def test_fullhd_train_selects_tactical_and_train_videos(self) -> None:
        entries = _fake_entries()
        picked = mpf.select_split_files(entries, "TRAIN", ["fullHD"])
        names = sorted(e.repo_path for e in picked)
        self.assertEqual(
            names,
            [
                "tactical_data_TRAIN.zip",
                "videos_fullHD_TRAIN_01.zip",
                "videos_fullHD_TRAIN_02.zip",
            ],
        )

    def test_include_lowres_union_dedupes_by_repo_path(self) -> None:
        entries = _fake_entries()
        picked = mpf.select_split_files(entries, "VAL", ["fullHD", "352x640"])
        names = sorted(e.repo_path for e in picked)
        self.assertEqual(
            names,
            [
                "tactical_data_VAL.zip",
                "videos_352x640_VAL.zip",
                "videos_fullHD_VAL.zip",
            ],
        )


class SkipExistingTests(unittest.TestCase):
    def test_same_size_local_file_is_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.zip"
            p.write_bytes(b"x" * 10)
            self.assertTrue(mpf._should_skip_existing(p, 10))
            self.assertFalse(mpf._should_skip_existing(p, 11))

    def test_missing_file_is_not_skipped(self) -> None:
        self.assertFalse(mpf._should_skip_existing(Path("nope"), 10))

    def test_no_remote_size_is_not_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.zip"
            p.write_bytes(b"x")
            self.assertFalse(mpf._should_skip_existing(p, None))


class MainDryRunTests(unittest.TestCase):
    """End-to-end main() invocation in dry-run mode."""

    def _run_main(self, tmp_root: Path, extra_argv: list[str]) -> mpf.FullMirrorResult:
        cfg_path = _write_config(tmp_root, tmp_root / "out")
        argv = ["--config", str(cfg_path), "--dry-run"] + extra_argv
        with mock.patch.object(mpf, "list_repo_files", return_value=_fake_entries()):
            rc = mpf.main(argv)
        self.assertEqual(rc, 0)
        top = (tmp_root / "out" / "manifest_full.json").read_text(encoding="utf-8")
        return json.loads(top)

    def test_default_dry_run_lists_all_three_splits(self) -> None:
        with TemporaryDirectory() as tmp:
            top = self._run_main(Path(tmp), extra_argv=[])
            splits = [s["split"] for s in top["per_split"]]
            self.assertEqual(splits, ["TRAIN", "VAL", "CHALLENGE"])
            for s in top["per_split"]:
                self.assertEqual(s["resolution"], "fullHD")
                # Dry run: nothing should be downloaded or extracted.
                for f in s["files"]:
                    self.assertFalse(f["downloaded"])
                self.assertEqual(s["extractions"], [])
                self.assertEqual(s["errors"], [])

    def test_subset_splits_only_writes_their_manifests(self) -> None:
        with TemporaryDirectory() as tmp:
            top = self._run_main(Path(tmp), extra_argv=["--splits", "VAL"])
            self.assertEqual([s["split"] for s in top["per_split"]], ["VAL"])
            self.assertTrue((Path(tmp) / "out" / "manifest_VAL.json").exists())
            self.assertFalse((Path(tmp) / "out" / "manifest_TRAIN.json").exists())

    def test_dry_run_manifest_does_not_leak_credentials(self) -> None:
        with TemporaryDirectory() as tmp:
            top = self._run_main(Path(tmp), extra_argv=[])
            blob = json.dumps(top)
            self.assertNotIn("TEST_TOKEN_DO_NOT_LOG", blob)
            self.assertNotIn("TEST_PASSWORD_DO_NOT_LOG", blob)


class MainSkipExistingTests(unittest.TestCase):
    """When archives are already present locally, they are not re-downloaded."""

    def test_existing_archives_are_recorded_as_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            (out / "raw").mkdir(parents=True)
            # Pre-populate the on-disk files at the same sizes as the
            # fake remote entries; everything should be skipped.
            for e in _fake_entries():
                p = out / "raw" / e.repo_path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\0" * (e.size or 0))
            cfg_path = _write_config(Path(tmp), out)
            # --no-extract keeps the test free of py7zr/pyzipper dependencies.
            argv = ["--config", str(cfg_path), "--splits", "VAL", "--no-extract"]
            with mock.patch.object(
                mpf, "list_repo_files", return_value=_fake_entries()
            ), mock.patch.object(mpf, "_download_one") as dl:
                rc = mpf.main(argv)
            self.assertEqual(rc, 0)
            dl.assert_not_called()
            split = json.loads(
                (out / "manifest_VAL.json").read_text(encoding="utf-8")
            )
            # Every selected VAL file should land in the skipped list.
            self.assertGreaterEqual(len(split["skipped"]), 2)
            self.assertEqual(split["files"], [])


if __name__ == "__main__":
    unittest.main()
