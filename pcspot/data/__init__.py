"""Data contract, dataset adapters, target generators, and caching."""

from pcspot.data.cache import TargetCache, config_hash
from pcspot.data.dataset import (
    DatasetItemKey,
    PCBASDataset,
    SampleTargets,
)
from pcspot.data.sampling import MixedEventSampler, UniformWindowSampler
from pcspot.data.schema import (
    NUM_PCBAS_CLASSES,
    PCBAS_CLASS_NAMES,
    PCBAS_ROLE_NAMES,
    EventLabel,
    PlayerSnapshot,
    Sample,
    SampleMeta,
    StackedSample,
    attach_visual_features,
    stack_sample,
)
from pcspot.data.splits import SplitManifest, ensure_no_overlap
from pcspot.data.targets import (
    ATTACKING_CLASSES_DEFAULT,
    DUEL_CLASSES_DEFAULT,
    CalfConfig,
    build_objectness_targets,
    build_pc_calf_targets,
    stack_objectness_targets,
    stack_pc_calf_targets,
)

__all__ = [
    "NUM_PCBAS_CLASSES",
    "PCBAS_CLASS_NAMES",
    "PCBAS_ROLE_NAMES",
    "EventLabel",
    "PlayerSnapshot",
    "Sample",
    "SampleMeta",
    "StackedSample",
    "attach_visual_features",
    "stack_sample",
    "ATTACKING_CLASSES_DEFAULT",
    "DUEL_CLASSES_DEFAULT",
    "CalfConfig",
    "build_objectness_targets",
    "build_pc_calf_targets",
    "stack_objectness_targets",
    "stack_pc_calf_targets",
    "TargetCache",
    "config_hash",
    "SplitManifest",
    "ensure_no_overlap",
    "PCBASDataset",
    "DatasetItemKey",
    "SampleTargets",
    "MixedEventSampler",
    "UniformWindowSampler",
]
