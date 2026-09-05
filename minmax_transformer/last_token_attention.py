"""Trainable-token-query attention evaluated at the final sequence position.

This module is intentionally independent of the existing fixed-query model.
It implements one attention head whose final-token query attends to the keys
and values of every token in the sequence.  A single linear classifier then
maps that last-token attention output to minimum and maximum class logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from minmax_transformer.config import ProblemConfig
from minmax_transformer.model import quantize_and_bound_values, quantize_to_grid


@dataclass(frozen=True, slots=True)
class LastTokenAttentionConfig:
    """Dimensions, constraints, and structured initialization parameters."""

    num_heads: int = 1
    key_query_dim: int = 3
    value_dim: int = 3
    precision_bits: int = 3
    max_value_norm: float = 16.0
    initial_query_value: float = 1.0
    initial_key_slope: float = 0.25
    initial_value_amplitude: float = 8.0

    def __post_init__(self) -> None:
        if self.num_heads != 1:
            raise ValueError("this experiment requires exactly one attention head")
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
        if self.initial_query_value == 0:
            raise ValueError("initial_query_value must be nonzero")
        if self.initial_key_slope <= 0:
            raise ValueError("initial_key_slope must be positive")
        if self.initial_value_amplitude <= 0:
            raise ValueError("initial_value_amplitude must be positive")
        if self.initial_value_amplitude > self.max_value_norm:
            raise ValueError("initial_value_amplitude must not exceed max_value_norm")

    @property
    def quantization_step(self) -> float:
        return 2.0**-self.precision_bits


class _GridParametrization(nn.Module):
    def __init__(self, precision_bits: int) -> None:
        super().__init__()
        self.precision_bits = precision_bits

    def forward(self, values: Tensor) -> Tensor:
        return quantize_to_grid(values, self.precision_bits)


class _BoundedGridParametrization(nn.Module):
    def __init__(self, precision_bits: int, max_norm: float) -> None:
        super().__init__()
        self.precision_bits = precision_bits
        self.max_norm = max_norm

    def forward(self, values: Tensor) -> Tensor:
        return quantize_and_bound_values(
            values,
            precision_bits=self.precision_bits,
            max_norm=self.max_norm,
        )


def chebyshev_query_features(
    vocabulary_size: int,
    key_query_dim: int,
    *,
    amplitude: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return distinct, bounded polynomial features over the vocabulary."""

    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must exceed one")
    if key_query_dim <= 0:
        raise ValueError("key_query_dim must be positive")
    if amplitude == 0:
        raise ValueError("amplitude must be nonzero")
    positions = torch.linspace(-1.0, 1.0, vocabulary_size, device=device, dtype=dtype)
    angles = torch.acos(positions.clamp(-1.0, 1.0))
    orders = torch.arange(key_query_dim, device=device, dtype=positions.dtype)
    return amplitude * torch.cos(angles[:, None] * orders[None, :])


def triangular_value_features(
    vocabulary_size: int,
    value_dim: int,
    *,
    amplitude: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return distinct piecewise-linear hat features over the vocabulary."""

    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must exceed one")
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    positions = torch.linspace(0.0, 1.0, vocabulary_size, device=device, dtype=dtype)
    if value_dim == 1:
        return amplitude * positions.unsqueeze(-1)
    centers = torch.linspace(0.0, 1.0, value_dim, device=device, dtype=dtype)
    spacing = 1.0 / (value_dim - 1)
    features = (1.0 - (positions[:, None] - centers[None, :]).abs() / spacing).clamp_min(0.0)
    return amplitude * features


class LastTokenAttentionMinMax(nn.Module):
    """One-head token-query attention followed by a multiclass linear map.

    For input ``x = (x_1, ..., x_n)``, only ``q(x_n)`` is used as a query.  It
    attends bidirectionally to ``k(x_1), ..., k(x_n)``.  The classifier is
    applied directly to the resulting weighted sum of value embeddings.
    """

    def __init__(
        self,
        problem: ProblemConfig | None = None,
        config: LastTokenAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.problem = problem or ProblemConfig()
        self.config = config or LastTokenAttentionConfig()
        vocabulary_size = self.problem.vocabulary_size

        self.query_embedding = nn.Embedding(vocabulary_size, self.config.key_query_dim)
        self.key_embedding = nn.Embedding(vocabulary_size, self.config.key_query_dim)
        self.value_embedding = nn.Embedding(vocabulary_size, self.config.value_dim)
        self.classifier = nn.Linear(
            self.config.value_dim,
            self.problem.num_targets * self.problem.num_classes,
        )

        self._initialize_embeddings()
        parametrize.register_parametrization(
            self.query_embedding,
            "weight",
            _GridParametrization(self.config.precision_bits),
        )
        parametrize.register_parametrization(
            self.key_embedding,
            "weight",
            _GridParametrization(self.config.precision_bits),
        )
        parametrize.register_parametrization(
            self.value_embedding,
            "weight",
            _BoundedGridParametrization(
                self.config.precision_bits,
                self.config.max_value_norm,
            ),
        )

    def _initialize_embeddings(self) -> None:
        token_order = torch.arange(
            self.problem.vocabulary_size,
            device=self.key_embedding.weight.device,
            dtype=self.key_embedding.weight.dtype,
        )
        initial_queries = chebyshev_query_features(
            self.problem.vocabulary_size,
            self.config.key_query_dim,
            amplitude=self.config.initial_query_value,
            device=self.query_embedding.weight.device,
            dtype=self.query_embedding.weight.dtype,
        )
        initial_keys = torch.zeros_like(self.key_embedding.weight)
        initial_keys[:, 0] = (
            -self.config.initial_key_slope
            * token_order
            * math.sqrt(self.config.key_query_dim)
            / self.config.initial_query_value
        )
        initial_values = triangular_value_features(
            self.problem.vocabulary_size,
            self.config.value_dim,
            amplitude=self.config.initial_value_amplitude,
            device=self.value_embedding.weight.device,
            dtype=self.value_embedding.weight.dtype,
        )
        with torch.no_grad():
            self.query_embedding.weight.copy_(initial_queries)
            self.key_embedding.weight.copy_(initial_keys)
            self.value_embedding.weight.copy_(initial_values)

    def _validate_inputs(self, inputs: Tensor) -> None:
        if inputs.ndim != 2 or inputs.shape[1] != self.problem.sequence_length:
            raise ValueError(
                f"inputs must have shape [batch, {self.problem.sequence_length}], "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("inputs must contain integers")
        if torch.any((inputs < self.problem.min_value) | (inputs > self.problem.max_value)):
            raise ValueError(
                f"inputs must lie between {self.problem.min_value} "
                f"and {self.problem.max_value}"
            )

    def forward(
        self,
        inputs: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(inputs)
        indices = inputs - self.problem.min_value
        last_query = self.query_embedding(indices[:, -1])
        keys = self.key_embedding(indices)
        values = self.value_embedding(indices)

        scores = torch.einsum("bd,bnd->bn", last_query, keys)
        scores = scores / math.sqrt(self.config.key_query_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        last_token_output = torch.einsum("bn,bnd->bd", attention_weights, values)
        logits = self.classifier(last_token_output).reshape(
            inputs.shape[0],
            self.problem.num_targets,
            self.problem.num_classes,
        )

        if return_attention:
            return logits, attention_weights, last_token_output
        return logits
