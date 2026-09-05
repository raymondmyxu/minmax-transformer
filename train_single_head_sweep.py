"""Adaptively train and evaluate one-head min/max models across value dimensions.

Training uses distinct triangular value features and cosine warm restarts.  The
controller stops only after two complete restart cycles fail to improve both
smoothed maximum loss and smoothed maximum accuracy (or a success/safety
limit), and never touches the independent test draw until training has stopped.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from minmax_transformer import (
    ClassificationMetrics,
    MinMaxTransformer,
    ModelConfig,
    ProblemConfig,
    SyntheticBatchGenerator,
    TrainingConfig,
    ValidationConfig,
    evaluate,
    resolve_device,
    seed_everything,
    train_one_epoch,
)


class VectorizedTensorBatches:
    """Iterate tensor batches without TensorDataset's per-row Python fetches."""

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        batch_size: int,
        shuffle_seed: int | None,
    ) -> None:
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.generator = None
        if shuffle_seed is not None:
            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(shuffle_seed)

    def __len__(self) -> int:
        return (self.inputs.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        sample_count = self.inputs.shape[0]
        if self.generator is None:
            for start in range(0, sample_count, self.batch_size):
                end = start + self.batch_size
                yield self.inputs[start:end], self.labels[start:end]
            return

        # Match DataLoader and RandomSampler generator advances exactly so this
        # faster loader preserves the established mini-batch sequence.
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        indices = torch.randperm(sample_count, generator=self.generator)
        for start in range(0, sample_count, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield self.inputs[batch_indices], self.labels[batch_indices]
        torch.randperm(sample_count, generator=self.generator)


def make_vectorized_iid_batches(
    problem: ProblemConfig,
    *,
    sample_count: int,
    batch_size: int,
    data_seed: int,
    shuffle_seed: int | None,
) -> VectorizedTensorBatches:
    """Draw one ascending-sorted IID data set and expose vectorized batches."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data_seed < 0:
        raise ValueError("data_seed must be non-negative")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ValueError("shuffle_seed must be non-negative or None")
    inputs, labels = SyntheticBatchGenerator(problem, seed=data_seed).sample_iid(sample_count)
    inputs = inputs.sort(dim=-1).values
    return VectorizedTensorBatches(
        inputs,
        labels,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
    )


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Cosine warm-restart schedule and cycle-level stopping policy."""

    max_epochs: int = 500_000
    diagnostic_interval: int = 5_000
    report_interval: int = 10_000
    maximum_learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-4
    restart_cycle_epochs: int = 50_000
    exponential_average_decay: float = 0.8
    cycle_accuracy_improvement: float = 2.5e-3
    cycle_loss_improvement: float = 5e-3
    non_improving_cycles_to_stop: int = 2
    success_accuracy: float = 0.99
    success_required_checks: int = 2

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive")
        if self.report_interval <= 0:
            raise ValueError("report_interval must be positive")
        if self.report_interval % self.diagnostic_interval != 0:
            raise ValueError("report_interval must be a multiple of diagnostic_interval")
        if self.maximum_learning_rate <= 0 or self.minimum_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.minimum_learning_rate > self.maximum_learning_rate:
            raise ValueError("minimum_learning_rate must not exceed maximum_learning_rate")
        if self.restart_cycle_epochs <= 1:
            raise ValueError("restart_cycle_epochs must be greater than one")
        if self.restart_cycle_epochs % self.diagnostic_interval != 0:
            raise ValueError("restart_cycle_epochs must be a multiple of diagnostic_interval")
        if not 0.0 <= self.exponential_average_decay < 1.0:
            raise ValueError("exponential_average_decay must be in [0, 1)")
        if self.cycle_accuracy_improvement <= 0:
            raise ValueError("cycle_accuracy_improvement must be positive")
        if self.cycle_loss_improvement <= 0:
            raise ValueError("cycle_loss_improvement must be positive")
        if self.non_improving_cycles_to_stop < 2:
            raise ValueError("non_improving_cycles_to_stop must be at least two")
        if not 0.0 < self.success_accuracy <= 1.0:
            raise ValueError("success_accuracy must be in (0, 1]")
        if self.success_required_checks <= 0:
            raise ValueError("success_required_checks must be positive")

    def learning_rate_at(self, epoch: int) -> float:
        """Return the cosine learning rate for a one-based training epoch."""

        if epoch <= 0:
            raise ValueError("epoch must be positive")
        position = (epoch - 1) % self.restart_cycle_epochs
        fraction = position / (self.restart_cycle_epochs - 1)
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.minimum_learning_rate + (
            self.maximum_learning_rate - self.minimum_learning_rate
        ) * cosine_weight


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """Action selected after one full training-set diagnostic."""

    event: str
    completed_cycles: int
    non_improving_cycles: int
    smoothed_maximum_loss: float
    smoothed_maximum_accuracy: float
    stop_reason: str | None


class ConvergenceController:
    """Stop after two full restart cycles fail to improve smoothed max metrics."""

    def __init__(self, config: ConvergenceConfig) -> None:
        self.config = config
        self.success_checks = 0
        self.completed_cycles = 0
        self.non_improving_cycles = 0
        self.smoothed_maximum_loss: float | None = None
        self.smoothed_maximum_accuracy: float | None = None
        self.best_cycle_maximum_loss: float | None = None
        self.best_cycle_maximum_accuracy: float | None = None

    def state_dict(self) -> dict[str, int | float | None]:
        """Return plain controller state suitable for a resumable checkpoint."""

        return {
            "success_checks": self.success_checks,
            "completed_cycles": self.completed_cycles,
            "non_improving_cycles": self.non_improving_cycles,
            "smoothed_maximum_loss": self.smoothed_maximum_loss,
            "smoothed_maximum_accuracy": self.smoothed_maximum_accuracy,
            "best_cycle_maximum_loss": self.best_cycle_maximum_loss,
            "best_cycle_maximum_accuracy": self.best_cycle_maximum_accuracy,
        }

    def load_state_dict(
        self,
        state: dict[str, int | float | None],
    ) -> None:
        """Restore state produced by :meth:`state_dict`."""

        self.success_checks = int(state["success_checks"])
        self.completed_cycles = int(state["completed_cycles"])
        self.non_improving_cycles = int(state["non_improving_cycles"])
        for name in (
            "smoothed_maximum_loss",
            "smoothed_maximum_accuracy",
            "best_cycle_maximum_loss",
            "best_cycle_maximum_accuracy",
        ):
            value = state[name]
            setattr(self, name, None if value is None else float(value))

    def _update_smoothed_metrics(self, metrics: ClassificationMetrics) -> None:
        if not math.isfinite(metrics.maximum_loss):
            raise ValueError("maximum-coordinate loss must be finite")
        decay = self.config.exponential_average_decay
        if self.smoothed_maximum_loss is None:
            self.smoothed_maximum_loss = metrics.maximum_loss
            self.smoothed_maximum_accuracy = metrics.maximum_accuracy
            return
        assert self.smoothed_maximum_accuracy is not None
        self.smoothed_maximum_loss = (
            decay * self.smoothed_maximum_loss + (1.0 - decay) * metrics.maximum_loss
        )
        self.smoothed_maximum_accuracy = (
            decay * self.smoothed_maximum_accuracy
            + (1.0 - decay) * metrics.maximum_accuracy
        )

    def observe(self, epoch: int, metrics: ClassificationMetrics) -> ControllerDecision:
        """Consume one deterministic training diagnostic and select the next action."""

        self._update_smoothed_metrics(metrics)
        assert self.smoothed_maximum_loss is not None
        assert self.smoothed_maximum_accuracy is not None
        if metrics.exact_accuracy >= self.config.success_accuracy:
            self.success_checks += 1
        else:
            self.success_checks = 0

        if self.success_checks >= self.config.success_required_checks:
            reason = (
                f"joint training accuracy >= {self.config.success_accuracy:.2%} "
                f"for {self.config.success_required_checks} checks"
            )
            return ControllerDecision(
                event="accuracy_target_reached",
                completed_cycles=self.completed_cycles,
                non_improving_cycles=self.non_improving_cycles,
                smoothed_maximum_loss=self.smoothed_maximum_loss,
                smoothed_maximum_accuracy=self.smoothed_maximum_accuracy,
                stop_reason=reason,
            )

        if epoch % self.config.restart_cycle_epochs != 0:
            return ControllerDecision(
                event="diagnostic",
                completed_cycles=self.completed_cycles,
                non_improving_cycles=self.non_improving_cycles,
                smoothed_maximum_loss=self.smoothed_maximum_loss,
                smoothed_maximum_accuracy=self.smoothed_maximum_accuracy,
                stop_reason=None,
            )

        self.completed_cycles += 1
        if self.best_cycle_maximum_loss is None:
            self.best_cycle_maximum_loss = self.smoothed_maximum_loss
            self.best_cycle_maximum_accuracy = self.smoothed_maximum_accuracy
            event = "restart_cycle_baseline"
        else:
            assert self.best_cycle_maximum_accuracy is not None
            accuracy_improved = self.smoothed_maximum_accuracy >= (
                self.best_cycle_maximum_accuracy + self.config.cycle_accuracy_improvement
            )
            loss_improved = self.smoothed_maximum_loss <= (
                self.best_cycle_maximum_loss - self.config.cycle_loss_improvement
            )
            if accuracy_improved:
                self.best_cycle_maximum_accuracy = self.smoothed_maximum_accuracy
            if loss_improved:
                self.best_cycle_maximum_loss = self.smoothed_maximum_loss
            if accuracy_improved or loss_improved:
                self.non_improving_cycles = 0
                event = "restart_cycle_improved"
            else:
                self.non_improving_cycles += 1
                event = "restart_cycle_no_improvement"

        if self.non_improving_cycles >= self.config.non_improving_cycles_to_stop:
            reason = (
                f"smoothed maximum loss and accuracy failed to improve for "
                f"{self.non_improving_cycles} complete restart cycles"
            )
            return ControllerDecision(
                event="restart_convergence_reached",
                completed_cycles=self.completed_cycles,
                non_improving_cycles=self.non_improving_cycles,
                smoothed_maximum_loss=self.smoothed_maximum_loss,
                smoothed_maximum_accuracy=self.smoothed_maximum_accuracy,
                stop_reason=reason,
            )
        return ControllerDecision(
            event=event,
            completed_cycles=self.completed_cycles,
            non_improving_cycles=self.non_improving_cycles,
            smoothed_maximum_loss=self.smoothed_maximum_loss,
            smoothed_maximum_accuracy=self.smoothed_maximum_accuracy,
            stop_reason=None,
        )


@dataclass(frozen=True, slots=True)
class SweepSettings:
    """Fully validated common settings for the adaptive sweep."""

    problem: ProblemConfig
    model_configs: tuple[ModelConfig, ...]
    training: TrainingConfig
    test: ValidationConfig
    convergence: ConvergenceConfig
    model_seed: int


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """One periodic full-training-set measurement."""

    epoch: int
    optimizer_steps: int
    learning_rate: float
    event: str
    completed_cycles: int
    non_improving_cycles: int
    smoothed_maximum_loss: float
    smoothed_maximum_accuracy: float
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    """Final metrics and convergence metadata for one value dimension."""

    value_dim: int
    trainable_parameters: int
    epochs_completed: int
    optimizer_steps: int
    best_epoch: int
    stop_reason: str
    training_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    diagnostic_history: tuple[DiagnosticRecord, ...]
    history_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptively train H=1 models across value dimensions."
    )
    parser.add_argument(
        "--value-dims",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 11, 13],
        metavar="D",
    )
    parser.add_argument("--samples", type=int, default=4_000)
    parser.add_argument("--max-epochs", type=int, default=500_000)
    parser.add_argument("--diagnostic-every", type=int, default=5_000)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--maximum-learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-4)
    parser.add_argument("--restart-cycle-epochs", type=int, default=50_000)
    parser.add_argument("--ema-decay", type=float, default=0.8)
    parser.add_argument("--cycle-accuracy-improvement", type=float, default=2.5e-3)
    parser.add_argument("--cycle-loss-improvement", type=float, default=5e-3)
    parser.add_argument("--non-improving-cycles", type=int, default=2)
    parser.add_argument("--success-accuracy", type=float, default=0.99)
    parser.add_argument("--success-required-checks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--data-seed", type=int, default=1_001)
    parser.add_argument("--shuffle-seed", type=int, default=1_002)
    parser.add_argument("--test-samples", type=int, default=5_000)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--test-data-seed", type=int, default=3_001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of concurrent CPU dimension runs (default: 1).",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing latest checkpoint and start the requested dimensions afresh.",
    )
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/single_head_adaptive_sweep"),
    )

    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--key-query-dim", type=int, default=1)
    parser.add_argument("--precision-bits", type=int, default=3)
    parser.add_argument("--max-value-norm", type=float, default=16.0)
    parser.add_argument("--initial-key-slope", type=float, default=0.25)
    parser.add_argument("--initial-value-amplitude", type=float, default=8.0)
    parser.add_argument("--initial-auxiliary-amplitude", type=float, default=4.0)
    return parser


def validate_value_dims(value_dims: list[int]) -> tuple[int, ...]:
    if not value_dims:
        raise ValueError("value_dims must contain at least one dimension")
    if any(value_dim <= 0 for value_dim in value_dims):
        raise ValueError("every value dimension must be positive")
    if len(set(value_dims)) != len(value_dims):
        raise ValueError("value dimensions must be distinct")
    return tuple(value_dims)


def build_settings(args: argparse.Namespace) -> SweepSettings:
    value_dims = validate_value_dims(args.value_dims)
    if args.model_seed < 0:
        raise ValueError("model_seed must be non-negative")
    convergence = ConvergenceConfig(
        max_epochs=args.max_epochs,
        diagnostic_interval=args.diagnostic_every,
        report_interval=args.report_every,
        maximum_learning_rate=args.maximum_learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        restart_cycle_epochs=args.restart_cycle_epochs,
        exponential_average_decay=args.ema_decay,
        cycle_accuracy_improvement=args.cycle_accuracy_improvement,
        cycle_loss_improvement=args.cycle_loss_improvement,
        non_improving_cycles_to_stop=args.non_improving_cycles,
        success_accuracy=args.success_accuracy,
        success_required_checks=args.success_required_checks,
    )
    problem = ProblemConfig(sequence_length=args.sequence_length, max_value=args.max_value)
    model_configs = tuple(
        ModelConfig(
            num_heads=1,
            key_query_dim=args.key_query_dim,
            value_dim=value_dim,
            precision_bits=args.precision_bits,
            max_value_norm=args.max_value_norm,
            initial_key_slope=args.initial_key_slope,
            initial_value_amplitude=args.initial_value_amplitude,
            initial_auxiliary_amplitude=args.initial_auxiliary_amplitude,
        )
        for value_dim in value_dims
    )
    training = TrainingConfig(
        sample_count=args.samples,
        batch_size=args.batch_size,
        max_epochs=convergence.max_epochs,
        evaluation_interval=convergence.diagnostic_interval,
        target_train_accuracy=convergence.success_accuracy,
        learning_rate=convergence.maximum_learning_rate,
        balance_regularization_strength=0.0,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        data_seed=args.data_seed,
        shuffle_seed=args.shuffle_seed,
    )
    test = ValidationConfig(
        sample_count=args.test_samples,
        batch_size=args.test_batch_size,
        data_seed=args.test_data_seed,
    )
    return SweepSettings(problem, model_configs, training, test, convergence, args.model_seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initialize_distinct_value_features(model: MinMaxTransformer) -> None:
    """Initialize one-head value coordinates as distinct triangular features.

    The evenly spaced hat functions form a low-norm feature bank over the token
    vocabulary.  At most two neighboring coordinates are nonzero for any token,
    so the initialization leaves substantial room below the value-vector bound.
    """

    if model.config.num_heads != 1:
        raise ValueError("distinct value-feature initialization requires one head")
    head = model.attention.heads[0]
    original = head.value_embedding.parametrizations.weight.original
    vocabulary_size, value_dim = original.shape
    positions = torch.linspace(0.0, 1.0, vocabulary_size, device=original.device)
    if value_dim == 1:
        features = positions.unsqueeze(-1)
    else:
        centers = torch.linspace(0.0, 1.0, value_dim, device=original.device)
        spacing = 1.0 / (value_dim - 1)
        features = (1.0 - (positions[:, None] - centers[None, :]).abs() / spacing).clamp_min(0.0)
    with torch.no_grad():
        original.copy_(features * model.config.initial_value_amplitude)

    effective_values = head.value_embedding.weight
    largest_norm = torch.linalg.vector_norm(effective_values, dim=-1).max().item()
    if largest_norm > model.config.max_value_norm + 1e-6:
        raise RuntimeError("distinct value-feature initialization exceeded the norm bound")


def _format_metrics(metrics: ClassificationMetrics) -> str:
    return (
        f"loss={metrics.loss:.6f}, max_loss={metrics.maximum_loss:.6f}, "
        f"joint={metrics.exact_accuracy:.2%}, "
        f"minimum={metrics.minimum_accuracy:.2%}, maximum={metrics.maximum_accuracy:.2%}"
    )


def save_diagnostic_history(path: str | Path, history: Sequence[DiagnosticRecord]) -> Path:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "optimizer_steps",
        "learning_rate",
        "event",
        "completed_restart_cycles",
        "non_improving_restart_cycles",
        "smoothed_maximum_loss",
        "smoothed_maximum_accuracy",
        "training_loss",
        "training_minimum_loss",
        "training_maximum_loss",
        "training_exact_accuracy",
        "training_minimum_accuracy",
        "training_maximum_accuracy",
        "training_mean_absolute_error",
        "training_sample_count",
    ]
    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "epoch": record.epoch,
                    "optimizer_steps": record.optimizer_steps,
                    "learning_rate": record.learning_rate,
                    "event": record.event,
                    "completed_restart_cycles": record.completed_cycles,
                    "non_improving_restart_cycles": record.non_improving_cycles,
                    "smoothed_maximum_loss": record.smoothed_maximum_loss,
                    "smoothed_maximum_accuracy": record.smoothed_maximum_accuracy,
                    "training_loss": record.metrics.loss,
                    "training_minimum_loss": record.metrics.minimum_loss,
                    "training_maximum_loss": record.metrics.maximum_loss,
                    "training_exact_accuracy": record.metrics.exact_accuracy,
                    "training_minimum_accuracy": record.metrics.minimum_accuracy,
                    "training_maximum_accuracy": record.metrics.maximum_accuracy,
                    "training_mean_absolute_error": record.metrics.mean_absolute_error,
                    "training_sample_count": record.metrics.sample_count,
                }
            )
    return history_path


def _checkpoint_fingerprint(
    settings: SweepSettings,
    model_config: ModelConfig,
) -> dict[str, object]:
    convergence = asdict(settings.convergence)
    convergence.pop("max_epochs")
    training = asdict(settings.training)
    training.pop("max_epochs")
    return {
        "problem": asdict(settings.problem),
        "model": asdict(model_config),
        "value_initialization": "distinct_triangular_features_v1",
        "training_except_max_epochs": training,
        "test": asdict(settings.test),
        "convergence_except_max_epochs": convergence,
        "model_seed": settings.model_seed,
    }


def _diagnostic_record_to_state(record: DiagnosticRecord) -> dict[str, object]:
    return {
        "epoch": record.epoch,
        "optimizer_steps": record.optimizer_steps,
        "learning_rate": record.learning_rate,
        "event": record.event,
        "completed_cycles": record.completed_cycles,
        "non_improving_cycles": record.non_improving_cycles,
        "smoothed_maximum_loss": record.smoothed_maximum_loss,
        "smoothed_maximum_accuracy": record.smoothed_maximum_accuracy,
        "metrics": asdict(record.metrics),
    }


def _diagnostic_record_from_state(state: dict[str, object]) -> DiagnosticRecord:
    metrics_state = state["metrics"]
    if not isinstance(metrics_state, dict):
        raise ValueError("checkpoint diagnostic metrics are invalid")
    return DiagnosticRecord(
        epoch=int(state["epoch"]),
        optimizer_steps=int(state["optimizer_steps"]),
        learning_rate=float(state["learning_rate"]),
        event=str(state["event"]),
        completed_cycles=int(state["completed_cycles"]),
        non_improving_cycles=int(state["non_improving_cycles"]),
        smoothed_maximum_loss=float(state["smoothed_maximum_loss"]),
        smoothed_maximum_accuracy=float(state["smoothed_maximum_accuracy"]),
        metrics=ClassificationMetrics(**metrics_state),
    )


def _atomic_torch_save(state: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)
    return path


def save_training_checkpoints(
    *,
    latest_path: Path,
    best_path: Path,
    settings: SweepSettings,
    model_config: ModelConfig,
    epoch: int,
    model: MinMaxTransformer,
    optimizer: Adam,
    controller: ConvergenceController,
    training_loader: VectorizedTensorBatches,
    history: Sequence[DiagnosticRecord],
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_model_joint: float,
    best_model_loss: float,
) -> tuple[Path, Path]:
    """Atomically save current and best state for an exact training resume."""

    generator_state = None
    if training_loader.generator is not None:
        generator_state = training_loader.generator.get_state()
    fingerprint = _checkpoint_fingerprint(settings, model_config)
    _atomic_torch_save(
        {
            "format_version": 1,
            "fingerprint": fingerprint,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "controller_state_dict": controller.state_dict(),
            "training_generator_state": generator_state,
            "torch_rng_state": torch.get_rng_state(),
            "history": [_diagnostic_record_to_state(record) for record in history],
            "best_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_model_joint": best_model_joint,
            "best_model_loss": best_model_loss,
        },
        latest_path,
    )
    _atomic_torch_save(
        {
            "format_version": 1,
            "fingerprint": fingerprint,
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "training_joint_accuracy": best_model_joint,
            "training_loss": best_model_loss,
        },
        best_path,
    )
    return latest_path, best_path


def run_dimension(
    *,
    settings: SweepSettings,
    model_config: ModelConfig,
    device: torch.device,
    output_dir: Path,
    no_progress: bool,
    resume_existing: bool,
) -> SweepRunResult:
    training = settings.training
    seed_everything(settings.model_seed)
    training_loader = make_vectorized_iid_batches(
        settings.problem,
        sample_count=training.sample_count,
        batch_size=training.batch_size,
        data_seed=training.data_seed,
        shuffle_seed=training.shuffle_seed,
    )
    training_diagnostic_loader = make_vectorized_iid_batches(
        settings.problem,
        sample_count=training.sample_count,
        batch_size=training.batch_size,
        data_seed=training.data_seed,
        shuffle_seed=None,
    )
    test_loader = make_vectorized_iid_batches(
        settings.problem,
        sample_count=settings.test.sample_count,
        batch_size=settings.test.batch_size,
        data_seed=settings.test.data_seed,
        shuffle_seed=None,
    )
    model = MinMaxTransformer(settings.problem, model_config).to(device)
    initialize_distinct_value_features(model)
    controller = ConvergenceController(settings.convergence)
    optimizer = Adam(
        model.parameters(),
        lr=settings.convergence.learning_rate_at(1),
        weight_decay=training.weight_decay,
    )
    steps_per_epoch = len(training_loader)
    trainable_parameters = count_trainable_parameters(model)
    history_path = output_dir / f"d{model_config.value_dim}_training_diagnostics.csv"
    latest_checkpoint_path = output_dir / f"d{model_config.value_dim}_latest.pt"
    best_checkpoint_path = output_dir / f"d{model_config.value_dim}_best.pt"
    print(
        f"\nstarting H=1, d={model_config.value_dim}: {trainable_parameters:,} parameters",
        flush=True,
    )
    initial_value_norm = torch.linalg.vector_norm(
        model.attention.heads[0].value_embedding.weight,
        dim=-1,
    ).max()
    print(
        "value initialization: distinct triangular features; "
        f"largest row norm={initial_value_norm.item():.4f}",
        flush=True,
    )

    history: list[DiagnosticRecord] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_model_joint = -math.inf
    best_model_loss = math.inf
    epochs_completed = 0
    stop_reason = f"maximum epoch cap ({settings.convergence.max_epochs}) reached"
    start_epoch = 1

    if resume_existing and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != _checkpoint_fingerprint(settings, model_config):
            raise ValueError(
                f"checkpoint settings mismatch for d={model_config.value_dim}; "
                "use --no-resume to start afresh"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        controller.load_state_dict(checkpoint["controller_state_dict"])
        generator_state = checkpoint["training_generator_state"]
        if training_loader.generator is not None and generator_state is not None:
            training_loader.generator.set_state(generator_state)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        history = [
            _diagnostic_record_from_state(record_state)
            for record_state in checkpoint["history"]
        ]
        best_state = checkpoint["best_state_dict"]
        best_epoch = int(checkpoint["best_epoch"])
        best_model_joint = float(checkpoint["best_model_joint"])
        best_model_loss = float(checkpoint["best_model_loss"])
        epochs_completed = int(checkpoint["epoch"])
        start_epoch = epochs_completed + 1
        print(
            f"d={model_config.value_dim}: resuming after epoch {epochs_completed} "
            f"at lr={optimizer.param_groups[0]['lr']:g}",
            flush=True,
        )

    progress = tqdm(
        range(start_epoch, settings.convergence.max_epochs + 1),
        desc=f"H=1, d={model_config.value_dim}",
        unit="epoch",
        dynamic_ncols=True,
        disable=no_progress,
    )

    for epoch in progress:
        learning_rate_used = settings.convergence.learning_rate_at(epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate_used
        update_metrics = train_one_epoch(
            model,
            training_loader,
            optimizer,
            device=device,
            max_grad_norm=training.max_grad_norm,
            balance_regularization_strength=0.0,
        )
        epochs_completed = epoch
        should_diagnose = (
            epoch % settings.convergence.diagnostic_interval == 0
            or epoch == settings.convergence.max_epochs
        )
        if not should_diagnose:
            continue

        metrics = evaluate(model, training_diagnostic_loader, device=device)
        is_best_model = metrics.exact_accuracy > best_model_joint or (
            metrics.exact_accuracy == best_model_joint and metrics.loss < best_model_loss
        )
        if is_best_model:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_model_joint = metrics.exact_accuracy
            best_model_loss = metrics.loss

        decision = controller.observe(epoch, metrics)
        event = decision.event
        if epoch == settings.convergence.max_epochs and decision.stop_reason is None:
            event = "maximum_epoch_cap_reached"
        history.append(
            DiagnosticRecord(
                epoch=epoch,
                optimizer_steps=epoch * steps_per_epoch,
                learning_rate=learning_rate_used,
                event=event,
                completed_cycles=decision.completed_cycles,
                non_improving_cycles=decision.non_improving_cycles,
                smoothed_maximum_loss=decision.smoothed_maximum_loss,
                smoothed_maximum_accuracy=decision.smoothed_maximum_accuracy,
                metrics=metrics,
            )
        )
        should_report = (
            epoch % settings.convergence.report_interval == 0
            or event != "diagnostic"
            or decision.stop_reason is not None
        )
        if should_report:
            print(
                f"d={model_config.value_dim} epoch={epoch} lr={learning_rate_used:g} "
                f"event={event}: {_format_metrics(metrics)}, "
                f"smoothed_max_loss={decision.smoothed_maximum_loss:.6f}, "
                f"smoothed_maximum={decision.smoothed_maximum_accuracy:.2%}, "
                f"non_improving_cycles={decision.non_improving_cycles}",
                flush=True,
            )
        progress.set_postfix(
            update_loss=f"{update_metrics.loss:.4f}",
            train_joint=f"{metrics.exact_accuracy:.2%}",
            lr=f"{learning_rate_used:g}",
            refresh=False,
        )

        if event.startswith("restart_cycle") and decision.stop_reason is None:
            print(
                f"d={model_config.value_dim}: warm restart after cycle "
                f"{decision.completed_cycles}; next epoch lr="
                f"{settings.convergence.maximum_learning_rate:g}",
                flush=True,
            )
        should_persist = (
            epoch % settings.convergence.report_interval == 0
            or event != "diagnostic"
            or decision.stop_reason is not None
        )
        if should_persist:
            if best_state is None:
                raise RuntimeError("cannot checkpoint before selecting a best model")
            save_diagnostic_history(history_path, history)
            save_training_checkpoints(
                latest_path=latest_checkpoint_path,
                best_path=best_checkpoint_path,
                settings=settings,
                model_config=model_config,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                controller=controller,
                training_loader=training_loader,
                history=history,
                best_state=best_state,
                best_epoch=best_epoch,
                best_model_joint=best_model_joint,
                best_model_loss=best_model_loss,
            )
            print(
                f"d={model_config.value_dim}: saved incremental diagnostics and checkpoints "
                f"at epoch {epoch}",
                flush=True,
            )
        if decision.stop_reason is not None:
            stop_reason = decision.stop_reason
            break
        if epoch == settings.convergence.max_epochs:
            break
    progress.close()

    if best_state is None:
        raise RuntimeError("training completed without a diagnostic")
    model.load_state_dict(best_state)
    training_metrics = evaluate(model, training_diagnostic_loader, device=device)
    test_metrics = evaluate(model, test_loader, device=device)
    history_path = save_diagnostic_history(history_path, history)
    print(f"d={model_config.value_dim} stop reason: {stop_reason}", flush=True)
    print(f"d={model_config.value_dim} restored best epoch: {best_epoch}", flush=True)
    print(
        f"d={model_config.value_dim} training: {_format_metrics(training_metrics)}",
        flush=True,
    )
    print(f"d={model_config.value_dim} test:     {_format_metrics(test_metrics)}", flush=True)
    print(f"saved diagnostics: {history_path.resolve()}", flush=True)

    return SweepRunResult(
        value_dim=model_config.value_dim,
        trainable_parameters=trainable_parameters,
        epochs_completed=epochs_completed,
        optimizer_steps=epochs_completed * steps_per_epoch,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        training_metrics=training_metrics,
        test_metrics=test_metrics,
        diagnostic_history=tuple(history),
        history_path=history_path,
    )


def initialize_dimension_worker(torch_num_threads: int) -> None:
    """Configure PyTorch exactly once when a spawned worker starts."""

    torch.set_num_threads(torch_num_threads)
    torch.set_num_interop_threads(torch_num_threads)


def run_dimension_worker(
    settings: SweepSettings,
    model_config: ModelConfig,
    device_type: str,
    output_dir: Path,
    no_progress: bool,
    resume_existing: bool,
) -> SweepRunResult:
    """Run one dimension in a reusable spawned CPU process."""

    return run_dimension(
        settings=settings,
        model_config=model_config,
        device=torch.device(device_type),
        output_dir=output_dir,
        no_progress=no_progress,
        resume_existing=resume_existing,
    )


def run_dimensions(
    *,
    settings: SweepSettings,
    device: torch.device,
    output_dir: Path,
    no_progress: bool,
    parallel_workers: int,
    torch_num_threads: int,
    resume_existing: bool,
) -> list[SweepRunResult]:
    """Run the configured dimensions sequentially or in a bounded CPU pool."""

    if parallel_workers == 1:
        return [
            run_dimension(
                settings=settings,
                model_config=model_config,
                device=device,
                output_dir=output_dir,
                no_progress=no_progress,
                resume_existing=resume_existing,
            )
            for model_config in settings.model_configs
        ]

    completed: dict[int, SweepRunResult] = {}
    with ProcessPoolExecutor(
        max_workers=parallel_workers,
        mp_context=get_context("spawn"),
        initializer=initialize_dimension_worker,
        initargs=(torch_num_threads,),
    ) as executor:
        futures = {
            executor.submit(
                run_dimension_worker,
                settings,
                model_config,
                device.type,
                output_dir,
                no_progress,
                resume_existing,
            ): model_config.value_dim
            for model_config in settings.model_configs
        }
        for future in as_completed(futures):
            value_dim = futures[future]
            result = future.result()
            completed[value_dim] = result
            print(f"completed dimension d={value_dim}", flush=True)

    return [completed[config.value_dim] for config in settings.model_configs]


_METRIC_NAMES = (
    "loss",
    "minimum_loss",
    "maximum_loss",
    "exact_accuracy",
    "minimum_accuracy",
    "maximum_accuracy",
    "mean_absolute_error",
    "sample_count",
)


def save_sweep_summary(
    path: str | Path,
    runs: Sequence[SweepRunResult],
    settings: SweepSettings,
) -> Path:
    if not runs:
        raise ValueError("cannot save an empty sweep")
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "value_dim",
        "num_heads",
        "precision_bits",
        "max_value_norm",
        "sequence_length",
        "max_input_value",
        "training_samples",
        "test_samples",
        "trainable_parameters",
        "epochs_completed",
        "optimizer_steps",
        "best_epoch",
        "stop_reason",
        "history_path",
    ]
    metric_fields = [f"{split}_{name}" for split in ("training", "test") for name in _METRIC_NAMES]
    configs = {config.value_dim: config for config in settings.model_configs}
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=base_fields + metric_fields)
        writer.writeheader()
        for run in runs:
            config = configs[run.value_dim]
            row: dict[str, str | int | float] = {
                "value_dim": run.value_dim,
                "num_heads": config.num_heads,
                "precision_bits": config.precision_bits,
                "max_value_norm": config.max_value_norm,
                "sequence_length": settings.problem.sequence_length,
                "max_input_value": settings.problem.max_value,
                "training_samples": settings.training.sample_count,
                "test_samples": settings.test.sample_count,
                "trainable_parameters": run.trainable_parameters,
                "epochs_completed": run.epochs_completed,
                "optimizer_steps": run.optimizer_steps,
                "best_epoch": run.best_epoch,
                "stop_reason": run.stop_reason,
                "history_path": str(run.history_path),
            }
            for split in ("training", "test"):
                metrics = getattr(run, f"{split}_metrics")
                row.update({f"{split}_{name}": getattr(metrics, name) for name in _METRIC_NAMES})
            writer.writerow(row)
    return summary_path


def save_dimension_sweep_plot(
    path: str | Path,
    runs: Sequence[SweepRunResult],
    settings: SweepSettings,
    *,
    show: bool,
) -> Path:
    if not runs:
        raise ValueError("cannot plot an empty sweep")
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ordered_runs = sorted(runs, key=lambda run: run.value_dim)
    value_dims = [run.value_dim for run in ordered_runs]
    figure, axis = plt.subplots(figsize=(11, 6))
    for metric_name, display_name, color in (
        ("minimum_accuracy", "Minimum", "tab:blue"),
        ("maximum_accuracy", "Maximum", "tab:orange"),
    ):
        axis.plot(
            value_dims,
            [getattr(run.training_metrics, metric_name) for run in ordered_runs],
            label=f"{display_name} — training",
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
        )
        axis.plot(
            value_dims,
            [getattr(run.test_metrics, metric_name) for run in ordered_runs],
            label=f"{display_name} — test",
            color=color,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
    config = settings.model_configs[0]
    axis.set_title(
        "Minimum/maximum accuracy vs value dimension\n"
        f"H=1, p={config.precision_bits}, L={config.max_value_norm:g}"
    )
    axis.set_xlabel("Value embedding dimension d")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(value_dims)
    axis.set_ylim(0.0, 1.01)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return plot_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.torch_num_threads <= 0:
        parser.error("torch_num_threads must be positive")
    if args.parallel_workers <= 0:
        parser.error("parallel_workers must be positive")
    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(args.torch_num_threads)
    try:
        settings = build_settings(args)
        device = resolve_device(args.device)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.parallel_workers > 1 and device.type != "cpu":
        parser.error("parallel dimension training requires --device cpu")

    dimensions = [config.value_dim for config in settings.model_configs]
    print(f"device: {device}")
    print(f"sweep: H=1 with d={dimensions}")
    print(
        f"problem: n={settings.problem.sequence_length}, M={settings.problem.max_value}, "
        f"p={settings.model_configs[0].precision_bits}, "
        f"L={settings.model_configs[0].max_value_norm:g}"
    )
    print(f"maximum epochs: {settings.convergence.max_epochs}")
    print(f"diagnostic interval: {settings.convergence.diagnostic_interval}")
    print(f"live report interval: {settings.convergence.report_interval}")
    print(
        "learning-rate schedule: cosine warm restarts from "
        f"{settings.convergence.maximum_learning_rate:g} to "
        f"{settings.convergence.minimum_learning_rate:g} every "
        f"{settings.convergence.restart_cycle_epochs} epochs"
    )
    print(
        "stopping rule: smoothed maximum loss and maximum accuracy fail to improve for "
        f"{settings.convergence.non_improving_cycles_to_stop} complete restart cycles; "
        f"EMA decay={settings.convergence.exponential_average_decay:g}, "
        f"accuracy threshold={settings.convergence.cycle_accuracy_improvement:.2%}, "
        f"loss threshold={settings.convergence.cycle_loss_improvement:g}"
    )
    print("value initialization: distinct triangular features")
    print(f"parallel workers: {args.parallel_workers}")
    print("test protocol: independent test data are used only after stopping")

    runs = run_dimensions(
        settings=settings,
        device=device,
        output_dir=args.output_dir,
        no_progress=args.no_progress,
        parallel_workers=args.parallel_workers,
        torch_num_threads=args.torch_num_threads,
        resume_existing=not args.no_resume,
    )
    summary_path = save_sweep_summary(
        args.output_dir / "dimension_sweep_summary.csv", runs, settings
    )
    plot_path = save_dimension_sweep_plot(
        args.output_dir / "accuracy_vs_value_dimension.png",
        runs,
        settings,
        show=args.show_plot,
    )
    print(f"saved aggregate summary: {summary_path.resolve()}")
    print(f"saved accuracy plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
