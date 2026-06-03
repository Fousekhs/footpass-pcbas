"""Frozen DINOv2 ViT-S/14 feature extractor.

This module owns the visual backbone used by both the offline
precompute script and the online inference wrapper. Design rules:

- The backbone is **always frozen**: ``eval()`` mode, no gradients,
  ``torch.inference_mode()`` for forward passes, and parameters are
  detached from any optimiser.
- Preprocessing is deterministic: standard ImageNet mean/std on
  ``[0, 1]`` float tensors. Crop resizing is the cropper's job; this
  module only handles normalisation and channel layout.
- The default backbone is **DINOv2 ViT-S/14** loaded via
  ``torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")``,
  matching the architecture the design doc calls for. Other
  ``"dinov2_vits14"``-compatible callables can be plugged in via
  ``backbone`` for tests / fine-tuning experiments.
- Output is the CLS-token embedding (``F_visual = 384`` for ViT-S/14).

The wrapper falls back to a small, deterministic CPU-only "stub"
extractor when ``torch.hub`` is unavailable or the user explicitly
asks for it. The stub produces a hashed embedding from the crop
tensor so the rest of the pipeline (cache, fusion, online inference)
can be exercised without GPU access. The stub is **not** a substitute
for the real backbone in terms of accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


try:  # Soft dependency: torch is required for the real backbone.
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - keeps the module importable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# DINOv2 ViT-S/14 produces a 384-dim CLS token. We expose this as a
# constant so the cache writer / dataset can validate cached arrays.
DINOV2_VITS14_DIM: int = 384


@dataclass
class DinoV2Config:
    """Configuration for :class:`DinoV2Extractor`."""

    backbone_name: str = "dinov2_vits14"
    crop_size: int = 224
    feature_dim: int = DINOV2_VITS14_DIM
    device: str = "cpu"
    use_stub: bool = False
    batch_size: int = 32


class _SupportsForward(Protocol):
    def __call__(self, x: "torch.Tensor") -> "torch.Tensor": ...  # type: ignore[name-defined]


class _StubBackbone:
    """Deterministic fallback when the real DINOv2 weights are unavailable.

    Computes a fixed-dimensional projection of the (mean, std,
    quadrant means) statistics of each crop. The output is **not**
    suitable for evaluation — it exists only to keep the rest of the
    pipeline runnable on machines without ``torch.hub`` access or
    network connectivity. All "real" runs should use the loaded
    DINOv2 weights.
    """

    def __init__(self, feature_dim: int) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("StubBackbone requires torch")
        self.feature_dim = int(feature_dim)
        # Deterministic projection independent of process state.
        gen = torch.Generator(device="cpu").manual_seed(20240601)
        self._proj = torch.randn(15, self.feature_dim, generator=gen)

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # x: (N, 3, H, W) float32 in [0,1] (post-normalisation).
        N = x.shape[0]
        h2 = x.shape[2] // 2
        w2 = x.shape[3] // 2
        stats = torch.stack(
            [
                x.mean(dim=(2, 3))[:, 0],
                x.mean(dim=(2, 3))[:, 1],
                x.mean(dim=(2, 3))[:, 2],
                x.std(dim=(2, 3))[:, 0],
                x.std(dim=(2, 3))[:, 1],
                x.std(dim=(2, 3))[:, 2],
                x[:, :, :h2, :w2].mean(dim=(2, 3)).mean(-1),
                x[:, :, :h2, w2:].mean(dim=(2, 3)).mean(-1),
                x[:, :, h2:, :w2].mean(dim=(2, 3)).mean(-1),
                x[:, :, h2:, w2:].mean(dim=(2, 3)).mean(-1),
                x[:, :, :h2, :w2].std(dim=(2, 3)).mean(-1),
                x[:, :, :h2, w2:].std(dim=(2, 3)).mean(-1),
                x[:, :, h2:, :w2].std(dim=(2, 3)).mean(-1),
                x[:, :, h2:, w2:].std(dim=(2, 3)).mean(-1),
                x.amax(dim=(2, 3)).mean(-1),
            ],
            dim=-1,
        )  # (N, 15)
        proj = self._proj.to(stats.device, stats.dtype)
        return stats @ proj  # (N, F)


def _load_dinov2_backbone(config: DinoV2Config) -> _SupportsForward:
    """Load a DINOv2 backbone, with stub fallback when ``use_stub`` is set."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("DinoV2Extractor requires torch to be installed")
    if config.use_stub:
        return _StubBackbone(config.feature_dim)
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            config.backbone_name,
            trust_repo=True,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on network/cache
        raise RuntimeError(
            "Failed to load DINOv2 via torch.hub. Pass use_stub=True for "
            "tests, or pre-download the weights manually."
        ) from exc
    model = model.to(config.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DinoV2Extractor:
    """Frozen DINOv2 ViT-S/14 feature extractor.

    The extractor is the only component that owns the backbone tensor
    and its preprocessing tensors. ``extract_features`` accepts a
    NumPy crop tensor (``N, H, W, 3`` uint8 or float32) and returns a
    NumPy ``(N, F_visual)`` tensor of CLS-token embeddings. The
    backbone is held in eval mode and forward passes run inside
    ``torch.inference_mode()`` so no autograd tape is built.
    """

    def __init__(
        self,
        config: DinoV2Config | None = None,
        *,
        backbone: Optional[Callable] = None,
    ) -> None:
        self.config = config or DinoV2Config()
        self._backbone = backbone if backbone is not None else _load_dinov2_backbone(self.config)
        if _TORCH_AVAILABLE:
            self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(self.config.device)
            self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(self.config.device)

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    @property
    def device(self) -> str:
        return self.config.device

    def _preprocess(self, crops: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required")
        if crops.ndim != 4 or crops.shape[3] != 3:
            raise ValueError("crops must be (N, H, W, 3)")
        S = self.config.crop_size
        if crops.shape[1] != S or crops.shape[2] != S:
            raise ValueError(
                f"crops must be {S}x{S}, got {crops.shape[1]}x{crops.shape[2]}"
            )
        if crops.dtype == np.uint8:
            arr = crops.astype(np.float32) / 255.0
        else:
            arr = crops.astype(np.float32)
        # (N, H, W, 3) -> (N, 3, H, W)
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
        x = x.to(self.config.device, non_blocking=False)
        x = (x - self._mean) / self._std
        return x

    def extract_features(self, crops: np.ndarray) -> np.ndarray:
        """Run the frozen backbone on ``(N, H, W, 3)`` crops.

        Returns ``(N, F_visual)`` float32. Invisible / blank crops
        should still be passed in (their features are simply ignored
        by the cache writer); the cropper's mask is what governs which
        rows are zeroed in the cached store.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required")
        if crops.size == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        out_chunks: list[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        with torch.inference_mode():
            for i in range(0, crops.shape[0], bs):
                batch = self._preprocess(crops[i : i + bs])
                feats = self._backbone(batch)
                if feats.ndim != 2:
                    raise ValueError(
                        f"backbone returned shape {tuple(feats.shape)}; "
                        "expected (N, F)"
                    )
                if feats.shape[1] != self.feature_dim:
                    raise ValueError(
                        f"backbone returned F={feats.shape[1]}; configured "
                        f"feature_dim={self.feature_dim}"
                    )
                out_chunks.append(feats.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32, copy=False)
