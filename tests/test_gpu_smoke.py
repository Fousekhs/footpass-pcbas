"""GPU smoke test for the PCBAS training + inference stack.

This is the CUDA counterpart to ``test_e2e_smoke.py``. It runs the same
tiny synthetic pipeline -- train a few steps, reload, run inference -- but
forces everything onto the GPU so we catch device-placement drift (tensors
left on CPU, ``.to(device)`` calls dropped, AMP autocast regressions) that a
CPU-only suite cannot see.

The whole module is skipped when no CUDA device is visible, so it is a
no-op on CPU CI and only exercises real hardware when present. Run it
explicitly with::

    python -m pytest tests/test_gpu_smoke.py -v

It is deliberately small (tiny hidden dim, 2 windows) so it finishes in a
couple of seconds on any GPU.
"""

from __future__ import annotations

import sys
import unittest
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
from pcspot.models.pipeline import PlayerCentricSpottingModel, stacked_to_batch
from pcspot.train.trainer import Trainer


def _make_half(match: str = "game_gpu", num_frames: int = 32,
               num_players: int = 4) -> HalfArray:
    """Build a synthetic tactical HalfArray with a couple of events."""
    events = [(8, 101, 2), (20, 202, 3)]
    rows: list[list[float]] = []
    for frame in range(num_frames):
        for p in range(num_players):
            pid = 101 + p if p < num_players // 2 else 201 + (p - num_players // 2)
            cls = 0
            for ef, ep, ec in events:
                if frame == ef and pid == ep:
                    cls = ec
            rows.append(
                [frame, pid, 1.0, p % 11 + 1, 1, 0.5 + 0.01 * p, 0.5 - 0.01 * p,
                 0.0, 0.0, 100.0, 100.0, 50.0, 50.0, cls]
            )
    return HalfArray(match_id=match, half_id="H1",
                     array=np.asarray(rows, dtype=np.float32))


def _tiny_model() -> PlayerCentricSpottingModel:
    return PlayerCentricSpottingModel(
        hidden_dim=8,
        num_hgt_layers=1,
        num_heads=2,
        num_mstcn_stages=1,
        num_mstcn_layers=2,
        knn=2,
    )


def _dataset(num_frames: int = 32, window_size: int = 8) -> PCBASDataset:
    half = _make_half(num_frames=num_frames)
    manifest = SplitManifest.from_dict({"train": [(half.match_id, half.half_id)]})
    return PCBASDataset(
        manifest=manifest,
        split="train",
        window_size=window_size,
        stride=window_size,
        halves=[half],
        calf_config=CalfConfig(),
        compute_targets=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class GpuSmokeTests(unittest.TestCase):
    def test_train_reload_infer_on_gpu(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

        ds = _dataset(num_frames=32, window_size=8)
        self.assertGreater(len(ds), 0)

        model = _tiny_model()
        trainer = Trainer(model, learning_rate=1e-3, device="cuda")

        # The model must actually live on the GPU after construction.
        param = next(model.parameters())
        self.assertEqual(param.device.type, "cuda")

        sampler_factory = reseeded_mixed_sampler_factory(
            ds.event_counts, positive_ratio=0.5, num_samples=len(ds), base_seed=42,
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
            # A real optimisation step produces a finite loss.
            self.assertTrue(np.isfinite(logs[0].avg_total_loss))
            self.assertGreaterEqual(trainer.global_step, 1)

            latest = ckpt_dir / "latest.pt"
            self.assertTrue(latest.exists())

            # Reload into a fresh GPU trainer and confirm the round-trip lands
            # weights back on CUDA, matching the trained params.
            torch.manual_seed(1)
            fresh = _tiny_model()
            fresh_trainer = Trainer(fresh, learning_rate=1e-3, device="cuda")
            fresh_trainer.load_checkpoint(latest)
            self.assertEqual(next(fresh.parameters()).device.type, "cuda")
            for p_train, p_reload in zip(model.parameters(), fresh.parameters()):
                self.assertTrue(
                    torch.allclose(p_train.detach().cpu(), p_reload.detach().cpu())
                )

            # Inference: move the batch onto the GPU and confirm the model
            # produces CUDA-resident logits without raising.
            stacked, _ = ds[0]
            batch = stacked_to_batch([stacked])
            gpu_batch = fresh_trainer._move_batch(batch)
            fresh.eval()
            with torch.no_grad():
                outputs = fresh(gpu_batch)
            self.assertEqual(outputs["logits"].device.type, "cuda")
            self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_amp_train_step_on_gpu(self) -> None:
        """Mixed-precision (fp16 + GradScaler) takes one step cleanly."""
        torch.manual_seed(0)
        ds = _dataset(num_frames=16, window_size=8)
        samples = [ds[i][0] for i in range(2)]

        trainer = Trainer(
            _tiny_model(),
            learning_rate=1e-3,
            device="cuda",
            amp_enabled=True,
            amp_dtype="fp16",
        )
        log = trainer.train_step(samples)
        self.assertTrue(np.isfinite(log.total_loss))
        self.assertEqual(trainer.global_step, 1)


if __name__ == "__main__":
    unittest.main()
