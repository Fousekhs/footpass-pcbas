"""Padded tracked-player crop extraction.

The PCBAS tactical schema stores ROI bounding boxes in fullHD pixel
coordinates (``1920 x 1080``). Broadcast videos are usually delivered
at a different resolution (e.g. ``640 x 352``), so the ROI must be
scaled to the actual video frame before cropping. This module owns
that scaling, the padding strategy that captures the player's
immediate environment (and implicitly the ball), bound clamping, and
the policy for invisible / NaN ROI rows.

The cropper is intentionally numpy-only so it can be exercised in
unit tests without OpenCV / PyAV, and so cache writers can run on
machines that only have a CPU + ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# Default reference resolution for the FOOTPASS / PCBAS ROI columns
# (``roi_x``, ``roi_y``, ``roi_width``, ``roi_height``). Centralised
# here so callers don't have to import the loader to pass them in.
DEFAULT_FULLHD_W: int = 1920
DEFAULT_FULLHD_H: int = 1080


@dataclass
class CropperConfig:
    """Configuration for :class:`PaddedPlayerCropper`.

    Attributes:
        crop_size: edge length of the square crop in pixels (after the
            cropper resizes the padded ROI). Must match the input
            resolution expected by the downstream backbone (e.g. 224
            for DINOv2 ViT-S/14 at the default patch size of 14).
        pad_factor: multiplicative padding applied to the ROI's larger
            edge before clamping. ``pad_factor=1.6`` expands a
            ``50x100`` ROI into a square of side ``100 * 1.6 = 160``.
        min_box_size: minimum crop side length in fullHD pixels. ROIs
            smaller than this (e.g. very distant players) are inflated
            to ``min_box_size`` before the pad factor is applied so
            the backbone always sees enough pixels to be useful.
        fullhd_width / fullhd_height: reference resolution for the
            ROI columns (defaults match the PCBAS layout).
        zero_for_invisible: when ``True``, invisible players (NaN ROI
            or ``visible=False``) yield an all-zero crop. When
            ``False``, a black square is still produced but a flag is
            returned so the caller can use it for masking.
    """

    crop_size: int = 224
    pad_factor: float = 1.6
    min_box_size: int = 32
    fullhd_width: int = DEFAULT_FULLHD_W
    fullhd_height: int = DEFAULT_FULLHD_H
    zero_for_invisible: bool = True

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError("crop_size must be > 0")
        if self.pad_factor <= 0:
            raise ValueError("pad_factor must be > 0")
        if self.min_box_size <= 0:
            raise ValueError("min_box_size must be > 0")
        if self.fullhd_width <= 0 or self.fullhd_height <= 0:
            raise ValueError("fullhd dimensions must be > 0")


def scale_roi_box(
    roi_xywh: Sequence[float],
    *,
    video_width: int,
    video_height: int,
    fullhd_width: int = DEFAULT_FULLHD_W,
    fullhd_height: int = DEFAULT_FULLHD_H,
) -> tuple[float, float, float, float] | None:
    """Convert a fullHD ROI (x, y, w, h) into video-frame coordinates.

    Returns ``None`` when any component is non-finite (NaN / inf), so
    callers can branch on visibility cleanly.
    """
    if any(not np.isfinite(v) for v in roi_xywh):
        return None
    sx = video_width / float(fullhd_width)
    sy = video_height / float(fullhd_height)
    x = float(roi_xywh[0]) * sx
    y = float(roi_xywh[1]) * sy
    w = float(roi_xywh[2]) * sx
    h = float(roi_xywh[3]) * sy
    return (x, y, w, h)


def pad_and_clamp_box(
    box_xywh: tuple[float, float, float, float],
    *,
    pad_factor: float,
    min_box_size: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Pad an ``(x, y, w, h)`` box, square it up, and clamp to frame.

    Returns ``(x1, y1, x2, y2)`` integer pixel coordinates. The output
    box is guaranteed to lie inside ``[0, frame_width) x [0, frame_height)``
    and to have positive width and height.
    """
    x, y, w, h = box_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    # Square up around the larger edge so player aspect is preserved.
    side = max(w, h)
    side = max(side, float(min_box_size))
    side *= float(pad_factor)
    half = side / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))
    # Clamp to the actual video frame.
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    return (x1, y1, x2, y2)


def _resize_crop(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize ``crop`` (H, W, 3) uint8 to ``size x size`` via bilinear.

    Uses cv2.resize when OpenCV is available (~0.2 ms/call). Falls back
    to a pure-NumPy implementation for environments without OpenCV (tests,
    machines without the binary wheel).
    """
    src_h, src_w = crop.shape[:2]
    if src_h == size and src_w == size:
        return crop
    if src_h == 0 or src_w == 0:
        return np.zeros((size, size, crop.shape[2]), dtype=crop.dtype)
    try:
        import cv2 as _cv2
        return _cv2.resize(crop, (size, size), interpolation=_cv2.INTER_LINEAR)
    except ImportError:
        pass
    # NumPy fallback — ~11 ms/call; only for environments without OpenCV.
    # Generate sample coordinates centered in each output pixel.
    ys = (np.arange(size) + 0.5) * (src_h / size) - 0.5
    xs = (np.arange(size) + 0.5) * (src_w / size) - 0.5
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (ys - y0).reshape(-1, 1)
    wx = (xs - x0).reshape(1, -1)
    Iy0x0 = crop[y0[:, None], x0[None, :]].astype(np.float32)
    Iy0x1 = crop[y0[:, None], x1[None, :]].astype(np.float32)
    Iy1x0 = crop[y1[:, None], x0[None, :]].astype(np.float32)
    Iy1x1 = crop[y1[:, None], x1[None, :]].astype(np.float32)
    top = Iy0x0 * (1 - wx[..., None]) + Iy0x1 * wx[..., None]
    bot = Iy1x0 * (1 - wx[..., None]) + Iy1x1 * wx[..., None]
    out = top * (1 - wy[..., None]) + bot * wy[..., None]
    return out.astype(crop.dtype)


@dataclass
class PaddedPlayerCropper:
    """Extract padded player crops from a video frame.

    Usage::

        cropper = PaddedPlayerCropper(CropperConfig())
        crops, mask = cropper.extract(
            frame=image,                      # (H, W, 3) uint8
            roi_xywh=rois_for_visible_players  # (N, 4) float32
        )

    where ``crops`` is shaped ``(N, crop_size, crop_size, 3)`` and
    ``mask[i]`` is True when the player produced a non-empty crop. The
    cropper does not implement any backbone-specific normalisation;
    that lives in :mod:`pcspot.features.dinov2`.
    """

    config: CropperConfig

    def extract_one(
        self,
        frame: np.ndarray,
        roi_xywh: Sequence[float] | None,
    ) -> tuple[np.ndarray, bool]:
        """Crop a single player from ``frame``.

        Returns ``(crop, valid)`` where ``valid`` is True when the
        player was visible and the resulting crop is non-empty.
        Invisible / NaN players yield a zero-filled crop when
        ``zero_for_invisible`` is True.
        """
        cfg = self.config
        H, W = frame.shape[:2]
        if roi_xywh is None or any(not np.isfinite(v) for v in roi_xywh):
            return (
                np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=frame.dtype),
                False,
            )
        scaled = scale_roi_box(
            roi_xywh,
            video_width=W,
            video_height=H,
            fullhd_width=cfg.fullhd_width,
            fullhd_height=cfg.fullhd_height,
        )
        if scaled is None:
            return (
                np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=frame.dtype),
                False,
            )
        x1, y1, x2, y2 = pad_and_clamp_box(
            scaled,
            pad_factor=cfg.pad_factor,
            min_box_size=cfg.min_box_size,
            frame_width=W,
            frame_height=H,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return (
                np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=frame.dtype),
                False,
            )
        crop = _resize_crop(crop, cfg.crop_size)
        return crop, True

    def extract(
        self,
        frame: np.ndarray,
        roi_xywh: np.ndarray | Iterable[Sequence[float] | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop a list of players from a single frame.

        Vectorises the bounding-box math (scale / pad / clamp) with NumPy
        so only the final per-crop ``cv2.resize`` call runs in a Python
        loop, eliminating ~9 ms of per-player function-call overhead on
        frames with many visible players.

        Args:
            frame: ``(H, W, 3)`` uint8 image.
            roi_xywh: ``(N, 4)`` array of fullHD ROI boxes, or an
                iterable of ``(x, y, w, h)`` tuples (``None`` allowed
                for invisible players).

        Returns:
            ``(crops, mask)`` where ``crops`` is ``(N, S, S, 3)`` of
            the same dtype as ``frame`` and ``mask`` is ``(N,)`` bool.
        """
        cfg = self.config
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be (H, W, 3)")
        H, W = frame.shape[:2]
        S = cfg.crop_size

        # Normalise to a float64 array; None rows become all-NaN.
        if isinstance(roi_xywh, np.ndarray):
            if roi_xywh.ndim != 2 or roi_xywh.shape[1] != 4:
                raise ValueError("roi_xywh must be (N, 4)")
            rois_arr = roi_xywh.astype(np.float64, copy=False)
        else:
            raw = list(roi_xywh)
            rois_arr = np.full((len(raw), 4), np.nan, dtype=np.float64)
            for i, r in enumerate(raw):
                if r is not None:
                    rois_arr[i] = r

        N = len(rois_arr)
        crops = np.zeros((N, S, S, 3), dtype=frame.dtype)
        mask = np.zeros(N, dtype=bool)
        if N == 0:
            return crops, mask

        # --- Vectorised box math (replaces per-player Python function calls) ---

        # 1. Visibility: any non-finite component → invisible
        finite = np.all(np.isfinite(rois_arr), axis=1)  # (N,) bool

        # 2. Scale fullHD → video-frame coordinates
        sx = W / float(cfg.fullhd_width)
        sy = H / float(cfg.fullhd_height)
        vx = rois_arr[:, 0] * sx
        vy = rois_arr[:, 1] * sy
        vw = rois_arr[:, 2] * sx
        vh = rois_arr[:, 3] * sy

        # 3. Square, pad, clamp  (mirrors pad_and_clamp_box exactly)
        cx = vx + vw * 0.5
        cy = vy + vh * 0.5
        side = (
            np.maximum(np.maximum(vw, vh), float(cfg.min_box_size))
            * float(cfg.pad_factor)
        )
        half = side * 0.5
        # nan_to_num converts invisible-row NaNs to 0 before the int cast so
        # NumPy doesn't emit "invalid value in cast" warnings; those rows are
        # skipped by the ``finite[i]`` guard in the loop below anyway.
        x1 = np.clip(np.nan_to_num(np.round(cx - half)).astype(np.int64), 0, W - 1)
        y1 = np.clip(np.nan_to_num(np.round(cy - half)).astype(np.int64), 0, H - 1)
        x2 = np.maximum(x1 + 1, np.clip(np.nan_to_num(np.round(cx + half)).astype(np.int64), 0, W))
        y2 = np.maximum(y1 + 1, np.clip(np.nan_to_num(np.round(cy + half)).astype(np.int64), 0, H))

        # --- Per-crop resize (variable source sizes → loop is unavoidable) ---
        _cv2_resize = None
        try:
            import cv2 as _cv2
            _cv2_resize = lambda src: _cv2.resize(  # noqa: E731
                src, (S, S), interpolation=_cv2.INTER_LINEAR
            )
        except ImportError:
            pass

        for i in range(N):
            if not finite[i]:
                continue
            raw_crop = frame[y1[i] : y2[i], x1[i] : x2[i]]
            if raw_crop.size == 0:
                continue
            crops[i] = (
                _cv2_resize(raw_crop)
                if _cv2_resize is not None
                else _resize_crop(raw_crop, S)
            )
            mask[i] = True

        return crops, mask
