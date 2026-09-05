"""Reusable training, validation, metrics, device, and checkpoint utilities."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from minmax_transformer.config import ModelConfig, ProblemConfig
from minmax_transformer.data import SyntheticBatchGenerator
from minmax_transformer.model import MinMaxTransformer


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for fitting on one fixed IID sample draw."""

    sample_count: int = 5_000
    batch_size: int = 128
    max_epochs: int = 70_000
    evaluation_interval: int = 100
    target_train_accuracy: float = 0.99
    learning_rate: float = 1e-3
    balance_regularization_strength: float = 1e-2
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    data_seed: int = 1_001
    shuffle_seed: int = 1_002

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        if not 0.0 < self.target_train_accuracy <= 1.0:
            raise ValueError("target_train_accuracy must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.balance_regularization_strength < 0:
            raise ValueError("balance_regularization_strength must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.data_seed < 0 or self.shuffle_seed < 0:
            raise ValueError("data and shuffle seeds must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for one fixed IID validation draw."""

    sample_count: int = 5_000
    batch_size: int = 256
    data_seed: int = 2_001

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative")


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Sample-weighted joint and coordinate-wise extrema metrics."""

    loss: float
    exact_accuracy: float
    minimum_accuracy: float
    maximum_accuracy: float
    mean_absolute_error: float
    sample_count: int
    minimum_loss: float = float("nan")
    maximum_loss: float = float("nan")


@dataclass(frozen=True, slots=True)
class EvaluationHistoryPoint:
    """Training and validation metrics measured at one optimizer step."""

    epoch: int
    optimizer_steps: int
    training_metrics: ClassificationMetrics
    validation_metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outcome recorded when an early-stopped or capped run finishes."""

    epochs_completed: int
    stop_reason: str
    final_training_metrics: ClassificationMetrics


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

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        loss: Tensor,
        coordinate_losses: Tensor,
    ) -> None:
        batch_size = labels.shape[0]
        predictions = logits.argmax(dim=-1)
        coordinate_correct = predictions.eq(labels)
        self.loss_sum += loss.item() * batch_size
        self.minimum_loss_sum += coordinate_losses[0].item() * batch_size
        self.maximum_loss_sum += coordinate_losses[1].item() * batch_size
        self.joint_correct += coordinate_correct.all(dim=-1).sum().item()
        self.minimum_correct += coordinate_correct[:, 0].sum().item()
        self.maximum_correct += coordinate_correct[:, 1].sum().item()
        self.absolute_error_sum += predictions.sub(labels).abs().sum().item()
        self.sample_count += batch_size

    def compute(self) -> ClassificationMetrics:
        if self.sample_count == 0:
            raise ValueError("cannot compute metrics for an empty data loader")
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


def _coordinate_classification_losses(logits: Tensor, labels: Tensor) -> Tensor:
    """Return separate categorical cross-entropies for min and max."""

    return torch.stack(
        [F.cross_entropy(logits[:, coordinate], labels[:, coordinate]) for coordinate in range(2)]
    )


def _classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Average categorical cross-entropy across min and max coordinates."""

    return _coordinate_classification_losses(logits, labels).mean()


def make_iid_data_loader(
    problem: ProblemConfig,
    *,
    sample_count: int,
    batch_size: int,
    data_seed: int,
    shuffle: bool,
    shuffle_seed: int | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Draw, ascending-sort, and wrap one finite IID data set."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data_seed < 0:
        raise ValueError("data_seed must be non-negative")
    if shuffle and (shuffle_seed is None or shuffle_seed < 0):
        raise ValueError("a non-negative shuffle_seed is required when shuffle=True")

    inputs, labels = SyntheticBatchGenerator(problem, seed=data_seed).sample_iid(sample_count)
    inputs = inputs.sort(dim=-1).values
    dataset = TensorDataset(inputs, labels)
    generator = None
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps while reporting unavailable explicit devices."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(normalized)


def train_one_epoch(
    model: MinMaxTransformer,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float | None,
    balance_regularization_strength: float = 0.0,
) -> ClassificationMetrics:
    """Run one epoch with cross-entropy plus auxiliary-table balance loss."""

    if balance_regularization_strength < 0:
        raise ValueError("balance_regularization_strength must be non-negative")

    model.train()
    metrics = _MetricAccumulator()
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        coordinate_losses = _coordinate_classification_losses(logits, labels)
        classification_loss = coordinate_losses.mean()
        objective = classification_loss + (
            balance_regularization_strength * model.auxiliary_balance_loss()
        )
        objective.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        metrics.update(
            logits.detach(),
            labels,
            classification_loss.detach(),
            coordinate_losses.detach(),
        )
    return metrics.compute()


@torch.no_grad()
def evaluate(
    model: MinMaxTransformer,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> ClassificationMetrics:
    """Evaluate joint/per-coordinate accuracy and extrema-coordinate MAE."""

    model.eval()
    metrics = _MetricAccumulator()
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        coordinate_losses = _coordinate_classification_losses(logits, labels)
        loss = coordinate_losses.mean()
        metrics.update(logits, labels, loss, coordinate_losses)
    return metrics.compute()


def save_evaluation_history(
    path: str | Path,
    history: Iterable[EvaluationHistoryPoint],
) -> Path:
    """Write evaluation history in a plotting- and spreadsheet-friendly CSV."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = (
        "loss",
        "minimum_loss",
        "maximum_loss",
        "exact_accuracy",
        "minimum_accuracy",
        "maximum_accuracy",
        "mean_absolute_error",
        "sample_count",
    )
    fieldnames = ["epoch", "optimizer_steps"]
    fieldnames.extend(f"training_{name}" for name in metric_names)
    fieldnames.extend(f"validation_{name}" for name in metric_names)

    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in history:
            row: dict[str, int | float] = {
                "epoch": point.epoch,
                "optimizer_steps": point.optimizer_steps,
            }
            row.update(
                {f"training_{name}": getattr(point.training_metrics, name) for name in metric_names}
            )
            row.update(
                {
                    f"validation_{name}": getattr(point.validation_metrics, name)
                    for name in metric_names
                }
            )
            writer.writerow(row)
    return history_path


def save_checkpoint(
    path: str | Path,
    model: MinMaxTransformer,
    *,
    training_config: TrainingConfig,
    training_result: TrainingResult,
    model_seed: int,
    evaluation_history: Iterable[EvaluationHistoryPoint] = (),
) -> Path:
    """Save model parameters and all configuration needed for validation."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 4,
        "problem_config": asdict(model.problem),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "training_result": asdict(training_result),
        "evaluation_history": [asdict(point) for point in evaluation_history],
        "model_seed": model_seed,
        "model_state": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[MinMaxTransformer, dict[str, Any]]:
    """Reconstruct a model from a project checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    format_version = payload.get("format_version")
    if format_version != 4:
        raise ValueError("checkpoint predates the two-coordinate min/max task")

    problem = ProblemConfig(**payload["problem_config"])
    model_config_payload = dict(payload["model_config"])
    # Checkpoints from the temporary structured-classifier experiment contain
    # this retired initialization-only field. Model weights remain loadable.
    model_config_payload.pop("initial_classifier_scale", None)
    model_config = ModelConfig(**model_config_payload)
    model = MinMaxTransformer(problem=problem, config=model_config).to(device)
    model.load_state_dict(payload["model_state"])
    metadata = {
        "training_config": payload["training_config"],
        "training_result": payload["training_result"],
        "evaluation_history": payload.get("evaluation_history", []),
        "model_seed": payload["model_seed"],
    }
    return model, metadata
