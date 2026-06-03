"""End-to-end smoke test for the full PCBAS training stack.

The test exercises the whole pipeline using a synthetic dataset:

    synthetic half  -> PCBASDataset
                    -> BatchProvider (MixedEventSampler-style)
                    -> Trainer.fit (1 epoch, latest.pt + epoch_0000.pt)
                    -> reload checkpoint into a fresh model
                    -> decode_predictions + player_centric_nms
                    -> Codabench submission writer + schema validation

It is deliberately small (tiny hidden dim, 2 windows, 1 epoch) so it
runs in a few seconds on CPU. The point is to catch interface drift
between the trainer, dataset, inference helpers, and the submission
writer -- not to exercise the math.
"""

from __future__ import annotations

import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.data.dataset import PCBASDataset
from pcspot.data.loader import HalfArray
from pcspot.data.sampling import (
    build_dataset_batch_provider,
    reseeded_mixed_sampler_factory,
)
from pcspot.data.splits import SplitManifest
from pcspot.data.targets import CalfConfig
from pcspot.eval.nms import decode_predictions, player_centric_nms
from pcspot.eval.submission import (
    InternalPrediction,
    SUBMISSION_FILE_NAME,
    group_predictions_by_match,
    validate_submission_payload,
    write_submission_zip,
)
from pcspot.models.pipeline import (
    PlayerCentricSpottingModel,
    stacked_to_batch,
)
from pcspot.train.trainer import Trainer


def _make_half(match: str = "game_e2e", num_frames: int = 32,
               num_players: int = 4,
               events: list[tuple[int, int, int]] | None = None) -> HalfArray:
    """Build a synthetic tactical HalfArray with optional per-frame events."""
    events = events or [(8, 101, 2), (20, 202, 3)]
    rows: list[list[float]] = []
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
                    0.5 - 0.01 * p,
                    0.0,
                    0.0,
                    100.0,
                    100.0,
                    50.0,
                    50.0,
                    cls,
                ]
            )
    return HalfArray(match_id=match, half_id="H1", array=np.asarray(rows, dtype=np.float32))


def _tiny_model() -> PlayerCentricSpottingModel:
    return PlayerCentricSpottingModel(
        hidden_dim=8,
        num_hgt_layers=1,
        num_heads=2,
        num_mstcn_stages=1,
        num_mstcn_layers=2,
        knn=2,
    )


class EndToEndSmokeTests(unittest.TestCase):
    def test_train_reload_infer_submit(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

        half = _make_half(num_frames=32)
        manifest = SplitManifest.from_dict({"train": [(half.match_id, half.half_id)]})

        ds = PCBASDataset(
            manifest=manifest,
            split="train",
            window_size=8,
            stride=8,
            halves=[half],
            calf_config=CalfConfig(),
            compute_targets=False,
        )
        self.assertGreater(len(ds), 0)

        model = _tiny_model()
        trainer = Trainer(model, learning_rate=1e-3)
        sampler_factory = reseeded_mixed_sampler_factory(
            ds.event_counts,
            positive_ratio=0.5,
            num_samples=len(ds),
            base_seed=42,
        )
        provider = build_dataset_batch_provider(
            ds, sampler_factory=sampler_factory, batch_size=2
        )

        with TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            logs = trainer.fit(
                provider, epochs=1, batch_size=2, checkpoint_dir=ckpt_dir,
            )
            self.assertEqual(len(logs), 1)
            latest = ckpt_dir / "latest.pt"
            epoch0 = ckpt_dir / "epoch_0000.pt"
            self.assertTrue(latest.exists())
            self.assertTrue(epoch0.exists())

            # Build a fresh model + trainer and reload the checkpoint to
            # confirm the on-disk state survives a round-trip.
            torch.manual_seed(1)
            fresh = _tiny_model()
            fresh_trainer = Trainer(fresh, learning_rate=1e-3)
            fresh_trainer.load_checkpoint(latest)
            self.assertGreaterEqual(fresh_trainer.global_step, 1)

            # Sanity: trained and reloaded params should agree.
            for p_train, p_reload in zip(
                model.parameters(), fresh.parameters()
            ):
                self.assertTrue(
                    torch.allclose(p_train.detach().cpu(), p_reload.detach().cpu())
                )

            # Run inference on the first window and decode predictions.
            stacked, _ = ds[0]
            batch = stacked_to_batch([stacked])
            fresh.eval()
            with torch.no_grad():
                outputs = fresh(batch)
            preds = decode_predictions(
                outputs["logits"][0],  # (T, P, C) for batch index 0
                confidence=outputs["confidence"][0] if "confidence" in outputs else None,
                valid_mask=stacked.valid_mask,
                player_ids=stacked.player_ids,
                score_threshold=0.0,
            )
            kept = player_centric_nms(preds, window_radius=2)
            # decode_predictions can legitimately return zero predictions
            # for an untrained tiny model; the smoke test asserts the
            # pipeline runs without raising rather than asserting on a
            # specific count.
            self.assertIsInstance(kept, list)

            # Translate at least one synthetic prediction into the
            # internal Codabench shape and exercise the writer + schema
            # validator end-to-end. Use a deterministic placeholder so
            # the assertion holds even when the model produces zero
            # decoded predictions.
            internal = [
                InternalPrediction(
                    match_id=half.match_id,
                    half=1,
                    frame=8,
                    class_id=2,
                    player_id=101,
                    score=0.42,
                    fps=25.0,
                )
            ]
            by_match = group_predictions_by_match(internal)
            from pcspot.eval.submission import build_match_document
            self.assertEqual(
                validate_submission_payload(
                    build_match_document(half.match_id, by_match[half.match_id])
                ),
                [],
            )
            zip_path = ckpt_dir / "submission.zip"
            write_submission_zip(by_match, zip_path)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn(
                    f"{half.match_id}/{SUBMISSION_FILE_NAME}", zf.namelist()
                )
                payload = json.loads(zf.read(f"{half.match_id}/{SUBMISSION_FILE_NAME}"))
            self.assertEqual(payload["UrlLocal"], half.match_id)
            self.assertEqual(len(payload["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
