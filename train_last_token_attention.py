"""Train the separate H=1 last-token-query min/max attention experiment.

The final input token supplies a learned query.  That query attends to learned
keys and values at every sequence position, including the last position, and a
single linear classifier predicts the minimum and maximum from the resulting
attention output.  Running this file is the only action that starts training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm.auto import tqdm

from minmax_transformer import (
    ClassificationMetrics,
    ProblemConfig,
    SyntheticBatchGenerator,
    resolve_device,
    seed_everything,
)
from minmax_transformer.last_token_attention import (
    LastTokenAttentionConfig,
    LastTokenAttentionMinMax,
)


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Finite-data optimization and independent-test settings."""

    sample_count: int = 4_000
    batch_size: int = 128
    test_sample_count: int = 5_000
    test_batch_size: int = 256
    max_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    data_seed: int = 1_001
    shuffle_seed: int = 1_002
    test_data_seed: int = 2_001
    model_seed: int = 7

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.test_sample_count <= 0:
            raise ValueError("sample counts must be positive")
        if self.batch_size <= 0 or self.test_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if min(self.data_seed, self.shuffle_seed, self.test_data_seed, self.model_seed) < 0:
            raise ValueError("seeds must be non-negative")


@dataclass(frozen=True, slots=True)
class ConvergenceSettings:
    """Cosine warm restarts and architecture-neutral cycle convergence."""

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
    success_joint_accuracy: float = 0.99
    success_required_checks: int = 2

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.diagnostic_interval <= 0 or self.report_interval <= 0:
            raise ValueError("diagnostic and report intervals must be positive")
        if self.report_interval % self.diagnostic_interval != 0:
            raise ValueError("report_interval must be a multiple of diagnostic_interval")
        if self.maximum_learning_rate <= 0 or self.minimum_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.minimum_learning_rate > self.maximum_learning_rate:
            raise ValueError("minimum_learning_rate must not exceed maximum_learning_rate")
        if self.restart_cycle_epochs <= 1:
            raise ValueError("restart_cycle_epochs must exceed one")
        if self.restart_cycle_epochs % self.diagnostic_interval != 0:
            raise ValueError("restart_cycle_epochs must be divisible by diagnostic_interval")
        if not 0.0 <= self.exponential_average_decay < 1.0:
            raise ValueError("exponential_average_decay must be in [0, 1)")
        if self.cycle_accuracy_improvement <= 0 or self.cycle_loss_improvement <= 0:
            raise ValueError("cycle improvement thresholds must be positive")
        if self.non_improving_cycles_to_stop < 2:
            raise ValueError("at least two non-improving cycles are required")
        if not 0.0 < self.success_joint_accuracy <= 1.0:
            raise ValueError("success_joint_accuracy must be in (0, 1]")
        if self.success_required_checks <= 0:
            raise ValueError("success_required_checks must be positive")

    def learning_rate_at(self, epoch: int) -> float:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        position = (epoch - 1) % self.restart_cycle_epochs
        fraction = position / (self.restart_cycle_epochs - 1)
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.minimum_learning_rate + (
            self.maximum_learning_rate - self.minimum_learning_rate
        ) * cosine_weight


class FixedIIDBatches:
    """One fixed IID sample draw, deliberately preserving sequence order."""

    def __init__(
        self,
        inputs: Tensor,
        labels: Tensor,
        *,
        batch_size: int,
        shuffle_seed: int | None,
    ) -> None:
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed

    def __len__(self) -> int:
        return (self.inputs.shape[0] + self.batch_size - 1) // self.batch_size

    def batches(self, epoch: int | None = None) -> Iterator[tuple[Tensor, Tensor]]:
        sample_count = self.inputs.shape[0]
        if self.shuffle_seed is None:
            indices = torch.arange(sample_count)
        else:
            if epoch is None or epoch <= 0:
                raise ValueError("a positive epoch is required for shuffled batches")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.shuffle_seed + epoch - 1)
            indices = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, self.batch_size):
            selected = indices[start : start + self.batch_size]
            yield self.inputs[selected], self.labels[selected]


def make_fixed_iid_batches(
    problem: ProblemConfig,
    *,
    sample_count: int,
    batch_size: int,
    data_seed: int,
    shuffle_seed: int | None,
) -> FixedIIDBatches:
    """Draw IID sequences without sorting or otherwise changing token order."""

    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count and batch_size must be positive")
    if data_seed < 0 or (shuffle_seed is not None and shuffle_seed < 0):
        raise ValueError("seeds must be non-negative")
    inputs, labels = SyntheticBatchGenerator(problem, seed=data_seed).sample_iid(sample_count)
    return FixedIIDBatches(
        inputs,
        labels,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
    )


class _MetricAccumulator:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.minimum_loss_sum = 0.0
        self.maximum_loss_sum = 0.0
        self.joint_correct = 0
        self.minimum_correct = 0
        self.maximum_correct = 0
        self.absolute_error_sum = 0.0
        self.sample_count = 0

    def update(self, logits: Tensor, labels: Tensor, coordinate_losses: Tensor) -> None:
        batch_size = labels.shape[0]
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels)
        self.loss_sum += coordinate_losses.mean().item() * batch_size
        self.minimum_loss_sum += coordinate_losses[0].item() * batch_size
        self.maximum_loss_sum += coordinate_losses[1].item() * batch_size
        self.joint_correct += correct.all(dim=-1).sum().item()
        self.minimum_correct += correct[:, 0].sum().item()
        self.maximum_correct += correct[:, 1].sum().item()
        self.absolute_error_sum += predictions.sub(labels).abs().sum().item()
        self.sample_count += batch_size

    def compute(self) -> ClassificationMetrics:
        if self.sample_count == 0:
            raise ValueError("cannot compute metrics from an empty batch collection")
        return ClassificationMetrics(
            loss=self.loss_sum / self.sample_count,
            exact_accuracy=self.joint_correct / self.sample_count,
            minimum_accuracy=self.minimum_correct / self.sample_count,
            maximum_accuracy=self.maximum_correct / self.sample_count,
            mean_absolute_error=self.absolute_error_sum / (2 * self.sample_count),
            sample_count=self.sample_count,
            minimum_loss=self.minimum_loss_sum / self.sample_count,
            maximum_loss=self.maximum_loss_sum / self.sample_count,
        )


def coordinate_losses(logits: Tensor, labels: Tensor) -> Tensor:
    return torch.stack(
        [F.cross_entropy(logits[:, coordinate], labels[:, coordinate]) for coordinate in range(2)]
    )


def train_one_epoch(
    model: LastTokenAttentionMinMax,
    batches: FixedIIDBatches,
    optimizer: Adam,
    *,
    epoch: int,
    device: torch.device,
    max_grad_norm: float | None,
) -> None:
    model.train()
    for inputs, labels in batches.batches(epoch):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = coordinate_losses(model(inputs), labels)
        losses.mean().backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()


@torch.no_grad()
def evaluate_model(
    model: LastTokenAttentionMinMax,
    batches: FixedIIDBatches,
    *,
    device: torch.device,
) -> ClassificationMetrics:
    model.eval()
    accumulator = _MetricAccumulator()
    for inputs, labels in batches.batches():
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        accumulator.update(logits, labels, coordinate_losses(logits, labels))
    return accumulator.compute()


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    event: str
    completed_cycles: int
    non_improving_cycles: int
    smoothed_worst_loss: float
    smoothed_worst_accuracy: float
    stop_reason: str | None


class ConvergenceController:
    """Track the weaker coordinate instead of assuming min or max is harder."""

    def __init__(self, settings: ConvergenceSettings) -> None:
        self.settings = settings
        self.success_checks = 0
        self.completed_cycles = 0
        self.non_improving_cycles = 0
        self.smoothed_worst_loss: float | None = None
        self.smoothed_worst_accuracy: float | None = None
        self.best_cycle_worst_loss: float | None = None
        self.best_cycle_worst_accuracy: float | None = None

    def state_dict(self) -> dict[str, int | float | None]:
        return {
            "success_checks": self.success_checks,
            "completed_cycles": self.completed_cycles,
            "non_improving_cycles": self.non_improving_cycles,
            "smoothed_worst_loss": self.smoothed_worst_loss,
            "smoothed_worst_accuracy": self.smoothed_worst_accuracy,
            "best_cycle_worst_loss": self.best_cycle_worst_loss,
            "best_cycle_worst_accuracy": self.best_cycle_worst_accuracy,
        }

    def load_state_dict(self, state: dict[str, int | float | None]) -> None:
        self.success_checks = int(state["success_checks"])
        self.completed_cycles = int(state["completed_cycles"])
        self.non_improving_cycles = int(state["non_improving_cycles"])
        for name in (
            "smoothed_worst_loss",
            "smoothed_worst_accuracy",
            "best_cycle_worst_loss",
            "best_cycle_worst_accuracy",
        ):
            value = state[name]
            setattr(self, name, None if value is None else float(value))

    def observe(self, epoch: int, metrics: ClassificationMetrics) -> ControllerDecision:
        worst_loss = max(metrics.minimum_loss, metrics.maximum_loss)
        worst_accuracy = min(metrics.minimum_accuracy, metrics.maximum_accuracy)
        decay = self.settings.exponential_average_decay
        if self.smoothed_worst_loss is None:
            self.smoothed_worst_loss = worst_loss
            self.smoothed_worst_accuracy = worst_accuracy
        else:
            assert self.smoothed_worst_accuracy is not None
            self.smoothed_worst_loss = decay * self.smoothed_worst_loss + (1 - decay) * worst_loss
            self.smoothed_worst_accuracy = (
                decay * self.smoothed_worst_accuracy + (1 - decay) * worst_accuracy
            )
        assert self.smoothed_worst_accuracy is not None

        self.success_checks = (
            self.success_checks + 1
            if metrics.exact_accuracy >= self.settings.success_joint_accuracy
            else 0
        )
        event = "diagnostic"
        stop_reason = None
        if self.success_checks >= self.settings.success_required_checks:
            event = "accuracy_target_reached"
            stop_reason = (
                f"joint training accuracy >= {self.settings.success_joint_accuracy:.2%} "
                f"for {self.settings.success_required_checks} checks"
            )
        elif epoch % self.settings.restart_cycle_epochs == 0:
            self.completed_cycles += 1
            if self.best_cycle_worst_loss is None:
                event = "restart_cycle_baseline"
                improved = True
            else:
                improved = (
                    self.smoothed_worst_accuracy
                    >= self.best_cycle_worst_accuracy
                    + self.settings.cycle_accuracy_improvement
                    or self.smoothed_worst_loss
                    <= self.best_cycle_worst_loss - self.settings.cycle_loss_improvement
                )
                event = "restart_cycle_improved" if improved else "restart_cycle_no_improvement"
            if improved:
                self.best_cycle_worst_loss = self.smoothed_worst_loss
                self.best_cycle_worst_accuracy = self.smoothed_worst_accuracy
                self.non_improving_cycles = 0
            else:
                self.non_improving_cycles += 1
                if self.non_improving_cycles >= self.settings.non_improving_cycles_to_stop:
                    event = "restart_convergence_reached"
                    stop_reason = (
                        f"{self.non_improving_cycles} complete restart cycles without "
                        "sufficient smoothed worst-coordinate improvement"
                    )
        if stop_reason is None and epoch >= self.settings.max_epochs:
            event = "safety_cap_reached"
            stop_reason = f"safety cap of {self.settings.max_epochs} epochs reached"

        return ControllerDecision(
            event=event,
            completed_cycles=self.completed_cycles,
            non_improving_cycles=self.non_improving_cycles,
            smoothed_worst_loss=self.smoothed_worst_loss,
            smoothed_worst_accuracy=self.smoothed_worst_accuracy,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    epoch: int
    optimizer_steps: int
    learning_rate: float
    event: str
    completed_cycles: int
    non_improving_cycles: int
    smoothed_worst_loss: float
    smoothed_worst_accuracy: float
    metrics: ClassificationMetrics

    def flat_dict(self) -> dict[str, int | float | str]:
        row: dict[str, int | float | str] = {
            "epoch": self.epoch,
            "optimizer_steps": self.optimizer_steps,
            "learning_rate": self.learning_rate,
            "event": self.event,
            "completed_cycles": self.completed_cycles,
            "non_improving_cycles": self.non_improving_cycles,
            "smoothed_worst_loss": self.smoothed_worst_loss,
            "smoothed_worst_accuracy": self.smoothed_worst_accuracy,
        }
        row.update(asdict(self.metrics))
        return row

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> DiagnosticRecord:
        metric_fields = {
            name: row[name]
            for name in ClassificationMetrics.__dataclass_fields__
        }
        return cls(
            epoch=int(row["epoch"]),
            optimizer_steps=int(row["optimizer_steps"]),
            learning_rate=float(row["learning_rate"]),
            event=str(row["event"]),
            completed_cycles=int(row["completed_cycles"]),
            non_improving_cycles=int(row["non_improving_cycles"]),
            smoothed_worst_loss=float(row["smoothed_worst_loss"]),
            smoothed_worst_accuracy=float(row["smoothed_worst_accuracy"]),
            metrics=ClassificationMetrics(**metric_fields),
        )


def save_diagnostics(path: Path, history: Sequence[DiagnosticRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.flat_dict() for record in history]
    if not rows:
        raise ValueError("cannot save empty diagnostic history")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _format_metrics(metrics: ClassificationMetrics) -> str:
    return (
        f"loss={metrics.loss:.6f}, joint={metrics.exact_accuracy:.2%}, "
        f"minimum={metrics.minimum_accuracy:.2%}, maximum={metrics.maximum_accuracy:.2%}"
    )


def _checkpoint_payload(
    *,
    model: LastTokenAttentionMinMax,
    optimizer: Adam,
    problem: ProblemConfig,
    model_config: LastTokenAttentionConfig,
    training: TrainingSettings,
    convergence: ConvergenceSettings,
    controller: ConvergenceController,
    history: Sequence[DiagnosticRecord],
    epoch: int,
    optimizer_steps: int,
    best_epoch: int,
    best_joint_accuracy: float,
    best_loss: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "architecture": "single_head_last_token_query_attention",
        "problem_config": asdict(problem),
        "model_config": asdict(model_config),
        "training_settings": asdict(training),
        "convergence_settings": asdict(convergence),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "controller_state": controller.state_dict(),
        "diagnostic_history": [record.flat_dict() for record in history],
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_joint_accuracy": best_joint_accuracy,
        "best_loss": best_loss,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train H=1 token-query attention and classify its last-token output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/last_token_model/d0_3"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--key-query-dim", type=int, default=3)
    parser.add_argument("--value-dim", type=int, default=3)
    parser.add_argument("--precision-bits", type=int, default=3)
    parser.add_argument("--max-value-norm", type=float, default=16.0)
    parser.add_argument("--initial-query-value", type=float, default=1.0)
    parser.add_argument("--initial-key-slope", type=float, default=0.25)
    parser.add_argument("--initial-value-amplitude", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=5_000)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--data-seed", type=int, default=1_001)
    parser.add_argument("--shuffle-seed", type=int, default=1_002)
    parser.add_argument("--test-data-seed", type=int, default=2_001)
    parser.add_argument("--model-seed", type=int, default=7)
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
    parser.add_argument("--success-joint-accuracy", type=float, default=0.99)
    parser.add_argument("--success-required-checks", type=int, default=2)
    return parser


def build_settings(
    args: argparse.Namespace,
) -> tuple[ProblemConfig, LastTokenAttentionConfig, TrainingSettings, ConvergenceSettings]:
    problem = ProblemConfig(sequence_length=args.sequence_length, max_value=args.max_value)
    model_config = LastTokenAttentionConfig(
        key_query_dim=args.key_query_dim,
        value_dim=args.value_dim,
        precision_bits=args.precision_bits,
        max_value_norm=args.max_value_norm,
        initial_query_value=args.initial_query_value,
        initial_key_slope=args.initial_key_slope,
        initial_value_amplitude=args.initial_value_amplitude,
    )
    training = TrainingSettings(
        sample_count=args.samples,
        batch_size=args.batch_size,
        test_sample_count=args.test_samples,
        test_batch_size=args.test_batch_size,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        data_seed=args.data_seed,
        shuffle_seed=args.shuffle_seed,
        test_data_seed=args.test_data_seed,
        model_seed=args.model_seed,
    )
    convergence = ConvergenceSettings(
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
        success_joint_accuracy=args.success_joint_accuracy,
        success_required_checks=args.success_required_checks,
    )
    return problem, model_config, training, convergence


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    problem, model_config, training, convergence = build_settings(args)
    if args.torch_num_threads <= 0:
        raise ValueError("torch_num_threads must be positive")
    torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    seed_everything(training.model_seed)

    training_batches = make_fixed_iid_batches(
        problem,
        sample_count=training.sample_count,
        batch_size=training.batch_size,
        data_seed=training.data_seed,
        shuffle_seed=training.shuffle_seed,
    )
    training_diagnostics = FixedIIDBatches(
        training_batches.inputs,
        training_batches.labels,
        batch_size=training.test_batch_size,
        shuffle_seed=None,
    )
    model = LastTokenAttentionMinMax(problem, model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=convergence.learning_rate_at(1),
        weight_decay=training.weight_decay,
    )
    controller = ConvergenceController(convergence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    diagnostics_path = args.output_dir / "training_diagnostics.csv"
    summary_path = args.output_dir / "summary.json"
    history: list[DiagnosticRecord] = []
    start_epoch = 1
    optimizer_steps = 0
    best_epoch = 0
    best_joint_accuracy = -math.inf
    best_loss = math.inf

    if latest_path.exists() and not args.no_resume:
        checkpoint = torch.load(latest_path, map_location=device)
        expected = (
            asdict(problem),
            asdict(model_config),
            asdict(training),
            asdict(convergence),
        )
        observed = (
            checkpoint["problem_config"],
            checkpoint["model_config"],
            checkpoint["training_settings"],
            checkpoint["convergence_settings"],
        )
        if observed != expected:
            raise ValueError("latest checkpoint settings do not match this requested experiment")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        controller.load_state_dict(checkpoint["controller_state"])
        history = [DiagnosticRecord.from_flat_dict(row) for row in checkpoint["diagnostic_history"]]
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_steps = int(checkpoint["optimizer_steps"])
        best_epoch = int(checkpoint["best_epoch"])
        best_joint_accuracy = float(checkpoint["best_joint_accuracy"])
        best_loss = float(checkpoint["best_loss"])
        print(f"resuming from epoch {start_epoch - 1}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "starting last-token experiment: "
        f"H=1, d0={model_config.key_query_dim}, d={model_config.value_dim}, "
        f"p={model_config.precision_bits}, L={model_config.max_value_norm:g}, "
        f"parameters={parameter_count:,}, device={device}",
        flush=True,
    )
    print(
        "input order is unsorted IID; q(x_last) attends to all tokens including x_last",
        flush=True,
    )

    stop_reason = None
    epochs_completed = start_epoch - 1
    progress = tqdm(
        range(start_epoch, convergence.max_epochs + 1),
        desc=(
            f"last-token H=1 d0={model_config.key_query_dim} "
            f"d={model_config.value_dim}"
        ),
        disable=args.no_progress,
    )
    for epoch in progress:
        learning_rate = convergence.learning_rate_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_one_epoch(
            model,
            training_batches,
            optimizer,
            epoch=epoch,
            device=device,
            max_grad_norm=training.max_grad_norm,
        )
        optimizer_steps += len(training_batches)
        epochs_completed = epoch
        if epoch % convergence.diagnostic_interval != 0 and epoch < convergence.max_epochs:
            continue

        metrics = evaluate_model(model, training_diagnostics, device=device)
        decision = controller.observe(epoch, metrics)
        record = DiagnosticRecord(
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            learning_rate=learning_rate,
            event=decision.event,
            completed_cycles=decision.completed_cycles,
            non_improving_cycles=decision.non_improving_cycles,
            smoothed_worst_loss=decision.smoothed_worst_loss,
            smoothed_worst_accuracy=decision.smoothed_worst_accuracy,
            metrics=metrics,
        )
        history.append(record)
        is_best = metrics.exact_accuracy > best_joint_accuracy or (
            metrics.exact_accuracy == best_joint_accuracy and metrics.loss < best_loss
        )
        if is_best:
            best_epoch = epoch
            best_joint_accuracy = metrics.exact_accuracy
            best_loss = metrics.loss
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "single_head_last_token_query_attention",
                    "problem_config": asdict(problem),
                    "model_config": asdict(model_config),
                    "selected_epoch": best_epoch,
                    "training_metrics": asdict(metrics),
                    "model_state": model.state_dict(),
                },
                best_path,
            )
        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                problem=problem,
                model_config=model_config,
                training=training,
                convergence=convergence,
                controller=controller,
                history=history,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                best_epoch=best_epoch,
                best_joint_accuracy=best_joint_accuracy,
                best_loss=best_loss,
            ),
            latest_path,
        )
        save_diagnostics(diagnostics_path, history)

        should_report = (
            epoch % convergence.report_interval == 0
            or decision.event.startswith("restart_cycle")
            or decision.stop_reason is not None
        )
        if should_report:
            print(
                f"epoch={epoch} lr={learning_rate:.8f} event={decision.event} "
                f"{_format_metrics(metrics)}, "
                f"smoothed_worst_loss={decision.smoothed_worst_loss:.6f}, "
                f"smoothed_worst_accuracy={decision.smoothed_worst_accuracy:.2%}, "
                f"non_improving_cycles={decision.non_improving_cycles}",
                flush=True,
            )
        if decision.stop_reason is not None:
            stop_reason = decision.stop_reason
            break

    progress.close()
    if stop_reason is None:
        stop_reason = f"safety cap of {convergence.max_epochs} epochs reached"
    if not best_path.exists():
        raise RuntimeError("training ended before a best checkpoint was written")

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    final_training = evaluate_model(model, training_diagnostics, device=device)
    test_batches = make_fixed_iid_batches(
        problem,
        sample_count=training.test_sample_count,
        batch_size=training.test_batch_size,
        data_seed=training.test_data_seed,
        shuffle_seed=None,
    )
    final_test = evaluate_model(model, test_batches, device=device)
    summary = {
        "architecture": "single_head_last_token_query_attention",
        "problem_config": asdict(problem),
        "model_config": asdict(model_config),
        "training_settings": asdict(training),
        "convergence_settings": asdict(convergence),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "training_metrics": asdict(final_training),
        "test_metrics": asdict(final_test),
        "artifacts": {
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"selected epoch: {best_epoch}; stop reason: {stop_reason}", flush=True)
    print(f"final training: {_format_metrics(final_training)}", flush=True)
    print(f"final test: {_format_metrics(final_test)}", flush=True)
    print(f"saved summary: {summary_path}", flush=True)
    return summary


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
