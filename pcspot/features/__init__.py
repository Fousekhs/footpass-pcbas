"""Visual feature extraction for player-centric action spotting.

This package implements the visual side of the architecture described
in ``docs/player_centric_hgt_mstcn_calf_design.md`` and the
``hgt_visual_features`` plan:

- :mod:`pcspot.features.cropper` extracts padded player crops from a
  video frame, handling fullHD-to-video coordinate scaling, padding,
  clamping, and missing/invisible players.
- :mod:`pcspot.features.dinov2` wraps a frozen DINOv2 ViT-S/14 backbone
  with deterministic preprocessing and ``torch.inference_mode``-based
  feature extraction.
- :mod:`pcspot.features.cache` reads and writes per-half visual feature
  caches keyed by ``(match_id, half_id, frame, player_id)`` and emits
  ``(T, P, F_visual)`` tensors aligned to a :class:`StackedSample`.

Public exports keep the surface area small and importable without
torch/PIL when only the cache reader is needed.
"""

from pcspot.features.cache import (
    VisualFeatureCache,
    VisualFeatureStore,
    align_features_to_stacked,
)
from pcspot.features.cropper import (
    CropperConfig,
    PaddedPlayerCropper,
    pad_and_clamp_box,
    scale_roi_box,
)

__all__ = [
    "CropperConfig",
    "PaddedPlayerCropper",
    "VisualFeatureCache",
    "VisualFeatureStore",
    "align_features_to_stacked",
    "pad_and_clamp_box",
    "scale_roi_box",
]
