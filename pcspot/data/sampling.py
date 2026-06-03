"""Event-centered sampling for PCBAS spotting.

Real-world action spotting datasets are extremely sparse: most windows do
not contain any event, so a uniform sampler spends nearly the entire
budget on negatives. This module implements a mixed sampler that
oversamples event-bearing windows by a configurable ratio while still
seeing some pure-background windows so the model learns the "nothing
happening" prior.

Two policies are supported:

- ``MixedEventSampler`` draws each minibatch index with probability
  ``positive_ratio`` from event-bearing windows and ``1 - positive_ratio``
  from background windows. The probability of an individual window inside
  each pool is uniform unless ``event_weighting="linear"``, in which case
  windows with more events are proportionally more likely (capped at
  ``cap_events_per_window``).

- ``UniformWindowSampler`` is a thin wrapper over ``range(len(dataset))``
  for back-compat (mostly useful for evaluation, where we want every
  window deterministically once).

Both samplers operate on the ``event_counts`` attribute of
``PCBASDataset``, so they are dataset-agnostic apart from the contract
``dataset.event_counts[i] >= 0``.

Documentation note (rationale):

- Why oversample events? PCBAS labels ~10-30 events per match-half over
  ~67k frames, so under a 256-frame window with stride 128 only ~5% of
  windows hold an event. Without oversampling, gradient updates from
  events drown in negatives.
- Why still keep background? The objectness head and the negative regions
  of CALF still need real "nothing here" examples; full event-only
  sampling causes the model to underestimate the negative class.
- Default ratio = 0.7 was chosen so each minibatch carries ~7 of 10
  positive windows on average, which is a typical CALF-style mix.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Iterator, Sequence, TYPE_CHECKING

import numpy as np

try:
    from torch.utils.data import Sampler
except Exception:  # pragma: no cover
    Sampler = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pcspot.data.dataset import PCBASDataset
    from pcspot.data.schema import StackedSample


class MixedEventSampler(Sampler):  # type: ignore[misc]
    """Yields a fixed number of indices per epoch, mixing positives & negatives."""

    def __init__(
        self,
        event_counts: Sequence[int],
        *,
        num_samples: int | None = None,
        positive_ratio: float = 0.7,
        event_weighting: str = "uniform",
        cap_events_per_window: int = 5,
        seed: int | None = None,
        replacement: bool = True,
    ) -> None:
        if not 0.0 <= positive_ratio <= 1.0:
            raise ValueError("positive_ratio must be in [0, 1]")
        if event_weighting not in {"uniform", "linear"}:
            raise ValueError("event_weighting must be 'uniform' or 'linear'")
        counts = np.asarray(event_counts, dtype=np.int64)
        if counts.ndim != 1:
            raise ValueError("event_counts must be 1D")
        self._counts = counts
        self._pos_idx = np.where(counts > 0)[0]
        self._neg_idx = np.where(counts == 0)[0]
        self._num_samples = (
            int(num_samples) if num_samples is not None else int(counts.size)
        )
        self._positive_ratio = float(positive_ratio)
        self._event_weighting = event_weighting
        self._cap = int(cap_events_per_window)
        self._replacement = bool(replacement)
        self._rng = np.random.default_rng(seed)

        if self._event_weighting == "linear" and self._pos_idx.size > 0:
            w = np.minimum(counts[self._pos_idx], self._cap).astype(np.float64)
            self._pos_probs = w / w.sum()
        else:
            self._pos_probs = None

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n_total = self._num_samples
        n_pos = (
            int(round(n_total * self._positive_ratio)) if self._pos_idx.size else 0
        )
        n_neg = n_total - n_pos
        if self._neg_idx.size == 0:
            n_pos += n_neg
            n_neg = 0
        if self._pos_idx.size == 0:
            n_neg += n_pos
            n_pos = 0

        pos_choices: np.ndarray
        if n_pos > 0:
            pos_choices = self._rng.choice(
                self._pos_idx,
                size=n_pos,
                replace=self._replacement,
                p=self._pos_probs,
            )
        else:
            pos_choices = np.empty(0, dtype=np.int64)
        if n_neg > 0:
            neg_choices = self._rng.choice(
                self._neg_idx, size=n_neg, replace=self._replacement
            )
        else:
            neg_choices = np.empty(0, dtype=np.int64)

        mix = np.concatenate([pos_choices, neg_choices])
        self._rng.shuffle(mix)
        for idx in mix:
            yield int(idx)


class UniformWindowSampler(Sampler):  # type: ignore[misc]
    """Deterministic iteration over every window once (for evaluation)."""

    def __init__(self, num_items: int) -> None:
        self._n = int(num_items)

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._n))


# --------------------------------------------------------------------- providers


BatchProvider = Callable[[int], Iterable[Sequence["StackedSample"]]]
"""A function ``epoch -> iterable of StackedSample batches``.

Trainers can iterate over the batches without ever materialising the
full sample list. The trainer is expected to call the provider once per
epoch with the integer epoch number so the provider can reseed
samplers, log epoch-specific statistics, etc.
"""


def build_dataset_batch_provider(
    dataset: "PCBASDataset",
    *,
    sampler_factory: Callable[[int], Iterable[int]],
    batch_size: int,
    drop_last: bool = False,
) -> BatchProvider:
    """Wrap a dataset+sampler combo into a ``BatchProvider``.

    The returned callable receives an integer ``epoch`` and yields lists
    of ``StackedSample`` of length up to ``batch_size``. Each call to the
    provider invokes ``sampler_factory(epoch)`` afresh so callers can
    reseed samplers per epoch (e.g. to vary positive/negative draws).

    The returned batches contain only the ``StackedSample`` (no target
    tensors); ``Trainer.train_step`` rebuilds CALF / objectness targets
    on the fly, which keeps this provider compatible with both
    ``PCBASDataset(compute_targets=True)`` and
    ``compute_targets=False`` configurations.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    def _provider(epoch: int) -> Iterator[Sequence["StackedSample"]]:
        batch: list = []
        for idx in sampler_factory(epoch):
            stacked, _ = dataset[int(idx)]
            batch.append(stacked)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    return _provider


def reseeded_mixed_sampler_factory(
    event_counts: Sequence[int],
    *,
    positive_ratio: float = 0.7,
    event_weighting: str = "uniform",
    cap_events_per_window: int = 5,
    num_samples: int | None = None,
    base_seed: int | None = None,
    replacement: bool = True,
) -> Callable[[int], Iterable[int]]:
    """Return a callable that builds a fresh ``MixedEventSampler`` per epoch.

    Each call seeds the sampler with ``base_seed + epoch`` (when
    ``base_seed`` is set) so different epochs see different positive /
    negative draws while remaining reproducible.
    """

    def _factory(epoch: int) -> MixedEventSampler:
        seed = None if base_seed is None else int(base_seed) + int(epoch)
        return MixedEventSampler(
            event_counts,
            num_samples=num_samples,
            positive_ratio=positive_ratio,
            event_weighting=event_weighting,
            cap_events_per_window=cap_events_per_window,
            seed=seed,
            replacement=replacement,
        )

    return _factory
