"""Synthetic data generation for the min/max classification task."""

from __future__ import annotations

import torch
from torch import Tensor

from minmax_transformer.config import ProblemConfig


def compute_targets(inputs: Tensor) -> Tensor:
    """Return ``[min(x), max(x)]`` for each row."""

    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape [batch, sequence], got {tuple(inputs.shape)}")
    if inputs.shape[1] == 0:
        raise ValueError("the sequence dimension must not be empty")
    return torch.stack((inputs.amin(dim=1), inputs.amax(dim=1)), dim=-1)


def _validate_extrema_shape(values: Tensor, *, name: str) -> None:
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError(f"{name} must have final dimension 2, got {tuple(values.shape)}")


def target_to_class(targets: Tensor, problem: ProblemConfig) -> Tensor:
    """Map ``[minimum, maximum]`` targets to zero-based class labels."""

    _validate_extrema_shape(targets, name="targets")
    if torch.any((targets < problem.min_value) | (targets > problem.max_value)):
        raise ValueError(
            f"target coordinates must be between {problem.min_value} and {problem.max_value}"
        )
    if torch.any(targets[..., 0] > targets[..., 1]):
        raise ValueError("target minimum must not exceed target maximum")
    return targets.to(dtype=torch.long) - problem.min_value


def class_to_target(labels: Tensor, problem: ProblemConfig) -> Tensor:
    """Map zero-based minimum/maximum labels back to input values."""

    _validate_extrema_shape(labels, name="labels")
    if torch.any((labels < 0) | (labels >= problem.num_classes)):
        raise ValueError(f"labels must be between 0 and {problem.num_classes - 1}")
    return labels.to(dtype=torch.long) + problem.min_value


class SyntheticBatchGenerator:
    """Generate reproducible IID, target-balanced, or mixed synthetic batches."""

    def __init__(
        self,
        problem: ProblemConfig | None = None,
        *,
        iid_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= iid_fraction <= 1.0:
            raise ValueError("iid_fraction must be in [0, 1]")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.problem = problem or ProblemConfig()
        self.iid_fraction = iid_fraction
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def sample_iid(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Sample vectors whose elements are independently uniform."""

        self._validate_batch_size(batch_size)
        inputs = torch.randint(
            low=self.problem.min_value,
            high=self.problem.max_value + 1,
            size=(batch_size, self.problem.sequence_length),
            generator=self.generator,
            dtype=torch.long,
        )
        labels = target_to_class(compute_targets(inputs), self.problem)
        return inputs, labels

    def sample_balanced(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Sample extrema, then construct inputs with exactly those extrema."""

        self._validate_batch_size(batch_size)
        minimums = torch.randint(
            low=self.problem.min_value,
            high=self.problem.max_value + 1,
            size=(batch_size,),
            generator=self.generator,
            dtype=torch.long,
        )
        maximum_range = self.problem.max_value - minimums + 1
        maximums = minimums + torch.floor(
            torch.rand(batch_size, generator=self.generator) * maximum_range
        ).to(dtype=torch.long)

        middle_count = self.problem.sequence_length - 2
        middle_range = (maximums - minimums + 1).unsqueeze(1)
        middle_values = minimums.unsqueeze(1) + torch.floor(
            torch.rand((batch_size, middle_count), generator=self.generator) * middle_range
        ).to(dtype=torch.long)

        inputs = torch.cat(
            (minimums.unsqueeze(1), maximums.unsqueeze(1), middle_values),
            dim=1,
        )
        permutations = torch.rand(
            (batch_size, self.problem.sequence_length),
            generator=self.generator,
        ).argsort(dim=1)
        inputs = inputs.gather(dim=1, index=permutations)

        targets = torch.stack((minimums, maximums), dim=-1)
        labels = target_to_class(targets, self.problem)
        return inputs, labels

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Mix IID and target-balanced examples, then shuffle the batch."""

        self._validate_batch_size(batch_size)
        iid_count = round(batch_size * self.iid_fraction)
        balanced_count = batch_size - iid_count

        batches: list[Tensor] = []
        label_batches: list[Tensor] = []
        if iid_count:
            inputs, labels = self.sample_iid(iid_count)
            batches.append(inputs)
            label_batches.append(labels)
        if balanced_count:
            inputs, labels = self.sample_balanced(balanced_count)
            batches.append(inputs)
            label_batches.append(labels)

        mixed_inputs = torch.cat(batches, dim=0)
        mixed_labels = torch.cat(label_batches, dim=0)
        order = torch.rand(batch_size, generator=self.generator).argsort()
        return mixed_inputs[order], mixed_labels[order]
