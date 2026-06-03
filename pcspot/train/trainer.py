"""Training loop scaffold for the player-centric model.

This module wires together:

- ``PCBASDataset`` / ``DataLoader`` for lazy windowed sampling.
- ``PlayerCentricSpottingModel`` forward.
- ``PlayerAwareCalfLoss`` with optional objectness supervision.
- A small learning-rate schedule (linear warmup -> cosine decay) and
  gradient clipping for stability on long runs.
- Save/load checkpoint helpers and an evaluation hook callback.

It is intentionally framework-light: no DDP, no AMP. Those are easy to
layer on top once we know the model trains, and the design doc tracks
where to add them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, Union

import numpy as np
import torch

from pcspot.data.schema import StackedSample
from pcspot.data.targets import (
    CalfConfig,
    build_objectness_targets,
    build_pc_calf_targets,
    stack_objectness_targets,
    stack_pc_calf_targets,
)
from pcspot.losses.pc_calf import PlayerAwareCalfLoss
from pcspot.models.pipeline import (
    PlayerCentricSpottingModel,
    StackedSampleBatch,
    stacked_to_batch,
)


# A "batch provider" is a callable that receives the current epoch and
# yields an iterable of ``Sequence[StackedSample]`` (one entry per
# batch). It is used by ``Trainer.fit`` and ``Trainer.train_epoch`` so a
# trainer can be driven either by a pre-materialised list of samples or
# by a sampler that re-draws windows each epoch (see
# ``pcspot.data.sampling.build_dataset_batch_provider``).
BatchProvider = Callable[[int], Iterable[Sequence[StackedSample]]]
TrainInput = Union[Sequence[StackedSample], BatchProvider]


@dataclass
class TrainStepLog:
    step: int
    total_loss: float
    bce_loss: float
    tmse_loss: float
    objectness_loss: float
    learning_rate: float = 0.0


@dataclass
class EpochLog:
    epoch: int
    num_steps: int
    avg_total_loss: float
    avg_bce_loss: float
    avg_tmse_loss: float
    avg_objectness_loss: float
    last_learning_rate: float
    validation: dict[str, float] = field(default_factory=dict)


@dataclass
class BatchTargets:
    class_targets: torch.Tensor
    class_weights: torch.Tensor
    objectness_targets: torch.Tensor
    objectness_weights: torch.Tensor


def _chunk(samples: Sequence[StackedSample], batch_size: int) -> Iterator[list[StackedSample]]:
    for i in range(0, len(samples), batch_size):
        yield list(samples[i : i + batch_size])


class _NullContext:
    """Tiny context-manager stand-in used when AMP is disabled.

    Lets ``train_step`` always wrap the forward in ``with ctx:`` without
    paying for ``torch.autocast`` when AMP is off.
    """

    def __enter__(self):  # noqa: D401 - trivial passthrough
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def make_targets_for_batch(
    samples: Sequence[StackedSample],
    *,
    config: CalfConfig | None = None,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_t: list[np.ndarray] = []
    per_w: list[np.ndarray] = []
    for s in samples:
        t, w = build_pc_calf_targets(s, config=config, num_classes=num_classes)
        per_t.append(t)
        per_w.append(w)
    t_b, w_b = stack_pc_calf_targets(per_t, per_w)
    return torch.from_numpy(t_b), torch.from_numpy(w_b)


def make_full_targets_for_batch(
    samples: Sequence[StackedSample],
    *,
    config: CalfConfig | None = None,
    num_classes: int,
) -> BatchTargets:
    per_t: list[np.ndarray] = []
    per_w: list[np.ndarray] = []
    per_ot: list[np.ndarray] = []
    per_ow: list[np.ndarray] = []
    for s in samples:
        t, w = build_pc_calf_targets(s, config=config, num_classes=num_classes)
        ot, ow = build_objectness_targets(t, w)
        per_t.append(t)
        per_w.append(w)
        per_ot.append(ot)
        per_ow.append(ow)
    t_b, w_b = stack_pc_calf_targets(per_t, per_w)
    ot_b, ow_b = stack_objectness_targets(per_ot, per_ow)
    return BatchTargets(
        class_targets=torch.from_numpy(t_b),
        class_weights=torch.from_numpy(w_b),
        objectness_targets=torch.from_numpy(ot_b),
        objectness_weights=torch.from_numpy(ow_b),
    )


class WarmupCosineSchedule:
    """Linear warmup -> cosine decay, returns a multiplier on ``base_lr``.

    Self-contained so it does not require constructing a torch
    ``LRScheduler`` and can be used by the manual loop or by passing
    ``optim.param_groups[i]['lr'] = base_lr * sched(step)``.
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)

    def __call__(self, step: int) -> float:
        step = max(0, int(step))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


class Trainer:
    """Single-process trainer for ``PlayerCentricSpottingModel``."""

    def __init__(
        self,
        model: PlayerCentricSpottingModel,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        tmse_lambda: float = 0.15,
        objectness_lambda: float = 1.0,
        stage_weights: object = None,
        calf_config: CalfConfig | None = None,
        device: str | torch.device = "cpu",
        grad_clip_norm: float | None = 1.0,
        schedule: WarmupCosineSchedule | None = None,
        grad_accum_steps: int = 1,
        amp_enabled: bool = False,
        amp_dtype: str = "bf16",
    ) -> None:
        self.model = model.to(device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = PlayerAwareCalfLoss(
            tmse_lambda=tmse_lambda,
            objectness_lambda=objectness_lambda,
            stage_weights=stage_weights,  # type: ignore[arg-type]
        )
        self.calf_config = calf_config or CalfConfig()
        self.device = torch.device(device)
        self.base_lr = float(learning_rate)
        self.grad_clip_norm = grad_clip_norm
        self.schedule = schedule
        self._global_step = 0
        self._epoch = 0
        # Gradient accumulation: divide the loss by ``grad_accum_steps``
        # and only call optim.step() every Nth micro-batch. ``train_step``
        # uses the counter ``self._accum_counter`` to decide whether to
        # zero gradients and step the optimizer.
        if int(grad_accum_steps) < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        self.grad_accum_steps = int(grad_accum_steps)
        self._accum_counter = 0
        # Mixed precision: ``torch.autocast`` wraps the forward+loss
        # call when ``amp_enabled=True``. On CPU autocast is a no-op
        # for fp16, and bf16 is only well-supported on Ampere+ GPUs;
        # the trainer just falls back to fp32 when CUDA is absent.
        self.amp_enabled = bool(amp_enabled)
        if amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        else:
            raise ValueError(
                f"amp_dtype must be 'bf16' or 'fp16', got {amp_dtype!r}"
            )
        # GradScaler is only needed for fp16 (bf16 has fp32-equivalent
        # exponent range). We keep one around and enable it on demand.
        use_scaler = (
            self.amp_enabled
            and self.amp_dtype == torch.float16
            and self.device.type == "cuda"
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Fallback for older torch where torch.cuda.amp.GradScaler is
            # the canonical entry point.
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # type: ignore[attr-defined]

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def epoch(self) -> int:
        return self._epoch

    def _move_batch(self, batch: StackedSampleBatch) -> StackedSampleBatch:
        def _opt(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(self.device) if t is not None else None

        return StackedSampleBatch(
            pitch_xy=batch.pitch_xy.to(self.device),
            velocity=batch.velocity.to(self.device),
            bbox=batch.bbox.to(self.device),
            roles=batch.roles.to(self.device),
            teams=batch.teams.to(self.device),
            valid_mask=batch.valid_mask.to(self.device),
            targets_class=batch.targets_class.to(self.device),
            global_features=_opt(batch.global_features),
            acceleration=_opt(batch.acceleration),
            time_features=_opt(batch.time_features),
            frames=_opt(batch.frames),
            visual_features=_opt(batch.visual_features),
            left_to_right=_opt(batch.left_to_right),
            shirt_numbers=_opt(batch.shirt_numbers),
        )

    def _apply_schedule(self) -> float:
        if self.schedule is None:
            return float(self.base_lr)
        mult = self.schedule(self._global_step)
        lr = self.base_lr * mult
        for pg in self.optim.param_groups:
            pg["lr"] = lr
        return float(lr)

    def train_epoch(
        self,
        samples: Sequence[StackedSample],
        *,
        batch_size: int = 4,
    ) -> Iterable[TrainStepLog]:
        self.model.train()
        for chunk in _chunk(samples, batch_size):
            yield self.train_step(chunk)

    def train_epoch_from_batches(
        self,
        batches: Iterable[Sequence[StackedSample]],
    ) -> Iterable[TrainStepLog]:
        """Run one training epoch from a pre-batched iterable.

        Use this when batches come from a sampler (e.g.
        ``MixedEventSampler``) so the trainer never needs to materialise
        the full sample list. Each batch is a sequence of
        ``StackedSample`` and is consumed verbatim by ``train_step``.
        """
        self.model.train()
        for chunk in batches:
            if not chunk:
                continue
            yield self.train_step(chunk)

    def _autocast_ctx(self):
        if not self.amp_enabled:
            return _NullContext()
        # torch.autocast accepts a device_type string. CUDA gets the
        # selected dtype; CPU bf16 autocast also works on recent torch
        # builds and is harmless to enable.
        try:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            )
        except (RuntimeError, ValueError):
            return _NullContext()

    def train_step(self, chunk: Sequence[StackedSample]) -> TrainStepLog:
        """Run one micro-batch.

        With ``grad_accum_steps == 1`` (default) this matches the legacy
        behavior. With ``grad_accum_steps > 1`` the loss is divided by
        ``grad_accum_steps`` and the optimizer only steps every Nth
        call, so the effective batch is ``batch_size * grad_accum_steps``
        without the memory cost.
        """
        self.model.train()
        batch = stacked_to_batch(list(chunk))
        tgt = make_full_targets_for_batch(
            chunk,
            config=self.calf_config,
            num_classes=self.model.num_classes,
        )
        batch = self._move_batch(batch)
        class_targets = tgt.class_targets.to(self.device)
        class_weights = tgt.class_weights.to(self.device)
        obj_targets = tgt.objectness_targets.to(self.device)
        obj_weights = tgt.objectness_weights.to(self.device)

        lr = self._apply_schedule()
        if self._accum_counter == 0:
            self.optim.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            outputs = self.model(batch)
            loss_out = self.loss_fn(
                outputs["stage_logits"],
                class_targets,
                class_weights,
                batch.valid_mask,
                stage_objectness_logits=outputs.get("stage_confidence"),
                objectness_targets=obj_targets,
                objectness_weights=obj_weights,
            )
        loss_for_backward = loss_out.total / float(self.grad_accum_steps)
        # ``scaler`` is a no-op when GradScaler is disabled (bf16 / cpu).
        self.scaler.scale(loss_for_backward).backward()
        self._accum_counter += 1
        if self._accum_counter >= self.grad_accum_steps:
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                # Unscale before clipping so the threshold is comparable
                # across AMP / no-AMP runs.
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
            self.scaler.step(self.optim)
            self.scaler.update()
            self._accum_counter = 0
        log = TrainStepLog(
            step=self._global_step,
            total_loss=float(loss_out.total.detach().cpu()),
            bce_loss=float(loss_out.bce.detach().cpu()),
            tmse_loss=float(loss_out.tmse.detach().cpu()),
            objectness_loss=float(loss_out.objectness.detach().cpu()),
            learning_rate=lr,
        )
        self._global_step += 1
        return log

    def fit(
        self,
        samples: TrainInput,
        *,
        epochs: int,
        batch_size: int = 4,
        validation_fn: Callable[["Trainer"], dict[str, float]] | None = None,
        checkpoint_dir: str | Path | None = None,
        keep_best_metric: str | None = None,
        log_fn: Callable[[TrainStepLog | EpochLog], None] | None = None,
        save_latest: bool = True,
        start_epoch: int | None = None,
    ) -> list[EpochLog]:
        """Run ``epochs`` of training and return per-epoch summaries.

        ``samples`` is either:

        - a ``Sequence[StackedSample]`` -- the trainer chunks it into
          ``batch_size`` slices, identical to the legacy behavior.
        - a ``BatchProvider`` callable -- the trainer calls
          ``samples(epoch)`` once per epoch and iterates the returned
          batches. This is what
          ``pcspot.data.sampling.build_dataset_batch_provider`` returns
          and is the preferred path for full-dataset training where the
          window count makes materialisation expensive.

        ``validation_fn`` (optional) receives ``self`` and returns a
        ``dict[str, float]`` of metric name -> value. If provided and
        ``keep_best_metric`` is set, the best epoch's checkpoint is
        copied to ``best.pt`` in ``checkpoint_dir``.

        When ``save_latest`` is True, ``latest.pt`` in
        ``checkpoint_dir`` is overwritten after every epoch so that
        ``--resume <output_dir>/latest.pt`` can recover from interruptions.

        ``start_epoch`` overrides the first epoch number used for
        bookkeeping and checkpoint filenames. Useful when resuming a
        previous run.
        """
        epoch_logs: list[EpochLog] = []
        best_value: float | None = None
        provider: BatchProvider | None = None
        if callable(samples) and not isinstance(samples, (list, tuple)):
            provider = samples  # type: ignore[assignment]

        first_epoch = (
            int(start_epoch) if start_epoch is not None else int(self._epoch)
        )
        for offset in range(epochs):
            ep = first_epoch + offset
            self._epoch = ep
            totals = {"total": 0.0, "bce": 0.0, "tmse": 0.0, "obj": 0.0}
            last_lr = 0.0
            step_count = 0
            if provider is not None:
                step_iter: Iterable[TrainStepLog] = self.train_epoch_from_batches(
                    provider(ep)
                )
            else:
                step_iter = self.train_epoch(
                    samples,  # type: ignore[arg-type]
                    batch_size=batch_size,
                )
            for log in step_iter:
                totals["total"] += log.total_loss
                totals["bce"] += log.bce_loss
                totals["tmse"] += log.tmse_loss
                totals["obj"] += log.objectness_loss
                last_lr = log.learning_rate
                step_count += 1
                if log_fn is not None:
                    log_fn(log)
            denom = max(step_count, 1)
            validation: dict[str, float] = {}
            if validation_fn is not None:
                validation = dict(validation_fn(self))
            epoch_log = EpochLog(
                epoch=ep,
                num_steps=step_count,
                avg_total_loss=totals["total"] / denom,
                avg_bce_loss=totals["bce"] / denom,
                avg_tmse_loss=totals["tmse"] / denom,
                avg_objectness_loss=totals["obj"] / denom,
                last_learning_rate=last_lr,
                validation=validation,
            )
            epoch_logs.append(epoch_log)
            if log_fn is not None:
                log_fn(epoch_log)

            if checkpoint_dir is not None:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(ckpt_dir / f"epoch_{ep:04d}.pt")
                if save_latest:
                    self.save_checkpoint(ckpt_dir / "latest.pt")
                if (
                    keep_best_metric is not None
                    and keep_best_metric in validation
                ):
                    val = float(validation[keep_best_metric])
                    if best_value is None or val > best_value:
                        best_value = val
                        self.save_checkpoint(ckpt_dir / "best.pt")
        return epoch_logs

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optim.state_dict(),
            "global_step": self._global_step,
            "epoch": self._epoch,
            "base_lr": self.base_lr,
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path: str | Path, *, load_optim: bool = True) -> None:
        # weights_only=True is the safe default on torch >= 2.4. Fall back
        # for older versions which do not accept the kwarg.
        try:
            payload: dict[str, Any] = torch.load(str(path), map_location=self.device, weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        if load_optim and "optim_state" in payload:
            self.optim.load_state_dict(payload["optim_state"])
        self._global_step = int(payload.get("global_step", 0))
        self._epoch = int(payload.get("epoch", 0))
        self.base_lr = float(payload.get("base_lr", self.base_lr))
