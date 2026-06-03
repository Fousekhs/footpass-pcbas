"""Train/val/test split discipline for the PCBAS dataset.

The dataset is partitioned at the (match, half) level: a single match-half
should not contribute frames to two different splits, otherwise nearby
windows can leak across train/val/test through overlapping context.

Two helpers are provided:

- ``SplitManifest`` is the canonical container used by ``PCBASDataset``.
  Callers can build it from a JSON file, a dict, or a list of explicit
  ``(match_id, half_id, split)`` tuples.
- ``ensure_no_overlap`` validates that the manifest is consistent: every
  ``(match_id, half_id)`` appears in at most one split.

The split string is opaque ("train" / "val" / "test" are conventional but
not enforced). When no manifest is available, callers can fall back to
``SplitManifest.single("train")`` to use everything for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class SplitManifest:
    """Maps ``(match_id, half_id)`` -> split name.

    Halves missing from the manifest are excluded by default so train/val
    leakage cannot happen silently.
    """

    assignments: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SplitManifest":
        out: dict[tuple[str, str], str] = {}
        for split, halves in d.items():
            for entry in halves:
                if isinstance(entry, dict):
                    key = (str(entry["match_id"]), str(entry["half_id"]))
                else:
                    key = (str(entry[0]), str(entry[1]))
                split_str = str(split)
                if key in out and out[key] != split_str:
                    raise ValueError(
                        f"Half {key} is assigned to multiple splits "
                        f"({out[key]!r} and {split_str!r}); split discipline "
                        "requires uniqueness."
                    )
                out[key] = split_str
        return cls(assignments=out)

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def single(cls, split: str, halves: Iterable[tuple[str, str]]) -> "SplitManifest":
        return cls(assignments={(str(m), str(h)): str(split) for m, h in halves})

    def ensure_no_overlap(self) -> None:
        seen: dict[tuple[str, str], str] = {}
        for key, split in self.assignments.items():
            if key in seen and seen[key] != split:
                raise ValueError(
                    f"Half {key} is assigned to multiple splits "
                    f"({seen[key]!r} and {split!r}); split discipline requires uniqueness."
                )
            seen[key] = split

    def split_of(self, match_id: str, half_id: str) -> str | None:
        return self.assignments.get((str(match_id), str(half_id)))

    def halves_for(self, split: str) -> list[tuple[str, str]]:
        return sorted(k for k, v in self.assignments.items() if v == split)

    def __iter__(self) -> Iterator[tuple[tuple[str, str], str]]:
        return iter(self.assignments.items())

    def __len__(self) -> int:
        return len(self.assignments)


def ensure_no_overlap(*manifests: SplitManifest) -> None:
    """Verify that several split manifests refer to disjoint halves."""
    seen: dict[tuple[str, str], str] = {}
    for m in manifests:
        for key, split in m.assignments.items():
            if key in seen and seen[key] != split:
                raise ValueError(
                    f"Half {key} is assigned to multiple splits across manifests "
                    f"({seen[key]!r} vs {split!r})."
                )
            seen[key] = split
