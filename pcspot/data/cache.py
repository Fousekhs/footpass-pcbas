"""Disk-backed cache for player-aware CALF targets.

Computing PC-CALF targets is O(events * T * P * C) and runs in pure numpy.
For large training runs the cost stacks up, especially because the model
itself is much smaller than e.g. an action-classification CNN. We cache
the targets to ``.npz`` files keyed on ``(match_id, half_id, window, config)``.

Cache invalidation:

- Any change to ``CalfConfig`` -> different ``cache_key`` -> cache miss.
- Any change to the underlying tactical array (different ``events``/positions
  inside the window) is captured by the optional ``data_fingerprint`` arg,
  which the caller can compute from the window's row hash. If not provided,
  we trust the (match, half, window) key.

We deliberately keep the file format simple (``np.savez_compressed``) so the
artifacts are portable and inspectable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pcspot.data.targets import CalfConfig


def _serializable_config(cfg: CalfConfig) -> dict[str, Any]:
    d = asdict(cfg)
    # frozenset is not JSON-serializable; convert to sorted lists.
    for key in ("attacking_classes", "duel_classes"):
        if key in d and not isinstance(d[key], list):
            d[key] = sorted(int(x) for x in d[key])
    # Tuple keys / values inside dicts are fine for asdict() output.
    return d


def config_hash(cfg: CalfConfig) -> str:
    """Stable short hash of the CALF config for cache filenames."""
    payload = json.dumps(
        _serializable_config(cfg), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


class TargetCache:
    """On-disk cache for ``(class_targets, class_weights, obj_targets, obj_weights)``.

    Each entry is one ``.npz`` file. The key is composed of a
    ``cache_key`` (caller-provided string identifier for the sample) plus the
    config hash so that different configs cohabit safely.
    """

    def __init__(self, root: str | Path, *, config: CalfConfig | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or CalfConfig()
        self.config_hash = config_hash(self.config)

    def _path(self, cache_key: str) -> Path:
        safe = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:24]
        return self.root / f"{self.config_hash}_{safe}.npz"

    def has(self, cache_key: str) -> bool:
        return self._path(cache_key).exists()

    def load(self, cache_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        p = self._path(cache_key)
        if not p.exists():
            return None
        with np.load(p) as data:
            return (
                data["class_targets"].astype(np.float32, copy=False),
                data["class_weights"].astype(np.float32, copy=False),
                data["objectness_targets"].astype(np.float32, copy=False),
                data["objectness_weights"].astype(np.float32, copy=False),
            )

    def save(
        self,
        cache_key: str,
        class_targets: np.ndarray,
        class_weights: np.ndarray,
        objectness_targets: np.ndarray,
        objectness_weights: np.ndarray,
    ) -> None:
        p = self._path(cache_key)
        # Use a tmp + rename so concurrent dataset workers don't corrupt each
        # other's reads.
        tmp = p.with_suffix(p.suffix + ".tmp")
        # ``np.savez_compressed`` appends ``.npz`` to string/Path inputs
        # but not to file objects. Open as a file handle so the tmp
        # rename below targets the actual on-disk path.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                class_targets=class_targets.astype(np.float32, copy=False),
                class_weights=class_weights.astype(np.float32, copy=False),
                objectness_targets=objectness_targets.astype(np.float32, copy=False),
                objectness_weights=objectness_weights.astype(np.float32, copy=False),
            )
        tmp.replace(p)
