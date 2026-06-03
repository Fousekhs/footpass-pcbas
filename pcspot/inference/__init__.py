"""Online / streaming inference for player-centric action spotting.

Strict single-pass evaluation environments (e.g. Codabench) typically
require predictions to be produced from raw video frames without
random access to the full match. The :mod:`pcspot.inference.online`
module implements a memory-bounded sliding-window wrapper that:

- Ingests one tactical row group + one video frame at a time.
- Extracts padded player crops via :class:`PaddedPlayerCropper`.
- Runs the frozen DINOv2 backbone under ``torch.inference_mode()``.
- Maintains a ring buffer of size ``window_size`` with strict
  eviction, so visual / kinematic tensors for old frames are freed
  immediately after the window slides.
- Builds a :class:`StackedSample` for the current window, feeds it
  through :class:`PlayerCentricSpottingModel`, and emits decoded
  predictions on a configurable cadence.
"""

from pcspot.inference.online import (
    OnlineInferenceConfig,
    OnlineSpotter,
    SlidingWindowBuffer,
)

__all__ = [
    "OnlineInferenceConfig",
    "OnlineSpotter",
    "SlidingWindowBuffer",
]
