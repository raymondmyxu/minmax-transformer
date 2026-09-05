"""Validated configuration objects and reproducibility helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ProblemConfig:
    """Define the discrete two-coordinate min/max classification problem."""

    sequence_length: int = 10
    min_value: int = 1
    max_value: int = 100

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.min_value < 0:
            raise ValueError("min_value must be non-negative")
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")

    @property
    def vocabulary_size(self) -> int:
        return self.max_value - self.min_value + 1

    @property
    def num_targets(self) -> int:
        return 2

    @property
    def num_classes(self) -> int:
        return self.vocabulary_size


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Define dimensions and constraints for the fixed-query attention heads."""

    num_heads: int = 2
    key_query_dim: int = 1
    value_dim: int = 3
    precision_bits: int = 3
    max_value_norm: float = 16.0
    initial_key_slope: float = 0.25
    initial_value_amplitude: float = 8.0
    initial_auxiliary_amplitude: float = 4.0

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.key_query_dim <= 0:
            raise ValueError("key_query_dim must be positive")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if isinstance(self.precision_bits, bool) or not isinstance(self.precision_bits, int):
            raise TypeError("precision_bits must be an integer")
        if self.precision_bits < 0:
            raise ValueError("precision_bits must be non-negative")
        if self.max_value_norm <= 0:
            raise ValueError("max_value_norm must be positive")
        if self.max_value_norm < self.quantization_step:
            raise ValueError("max_value_norm must be at least one quantization step")
        if self.initial_key_slope <= 0:
            raise ValueError("initial_key_slope must be positive")
        if self.initial_value_amplitude <= 0:
            raise ValueError("initial_value_amplitude must be positive")
        if self.initial_value_amplitude > self.max_value_norm:
            raise ValueError("initial_value_amplitude must not exceed max_value_norm")
        if self.initial_auxiliary_amplitude < 0:
            raise ValueError("initial_auxiliary_amplitude must be non-negative")
        auxiliary_dimensions = max(self.value_dim - min(self.num_heads, self.value_dim), 0)
        largest_initial_norm = math.sqrt(
            self.initial_value_amplitude**2
            + auxiliary_dimensions * self.initial_auxiliary_amplitude**2
        )
        if largest_initial_norm > self.max_value_norm:
            raise ValueError("the combined initial value coordinates exceed max_value_norm")

    @property
    def quantization_step(self) -> float:
        return 2.0**-self.precision_bits


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch without changing deterministic-kernel settings."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
