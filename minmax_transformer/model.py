"""Fixed-query, multi-head attention model for min/max classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from minmax_transformer.config import ModelConfig, ProblemConfig


def quantize_to_grid(values: Tensor, precision_bits: int) -> Tensor:
    """Quantize entries to multiples of 2^-p with a straight-through gradient."""

    step = 2.0**-precision_bits
    quantized = torch.round(values / step) * step
    return values + (quantized - values).detach()


def quantize_and_bound_values(
    values: Tensor,
    *,
    precision_bits: int,
    max_norm: float,
) -> Tensor:
    """Quantize row vectors to the 2^-p lattice while keeping norms at most L.

    Integer lattice coordinates are scaled toward zero whenever a quantized
    vector would exceed the norm bound. Truncating the scaled coordinates keeps
    the final vector on the lattice and cannot increase its Euclidean norm.
    """

    step = 2.0**-precision_bits
    lattice_coordinates = torch.round(values / step)
    quantized = lattice_coordinates * step
    norms = torch.linalg.vector_norm(quantized, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(step)
    scale = torch.clamp(max_norm / safe_norms, max=1.0)
    bounded_coordinates = torch.trunc(lattice_coordinates * scale)
    bounded = bounded_coordinates * step
    return values + (bounded - values).detach()


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


class FixedQueryAttention(nn.Module):
    """One attention head with a fixed all-ones query and learned keys/values."""

    def __init__(
        self,
        problem: ProblemConfig,
        config: ModelConfig,
        *,
        key_direction: int,
        output_coordinate: int,
    ) -> None:
        super().__init__()
        if key_direction not in {-1, 1}:
            raise ValueError("key_direction must be -1 or 1")
        if not 0 <= output_coordinate < config.value_dim:
            raise ValueError("output_coordinate must index the value embedding")
        self.problem = problem
        self.config = config
        self.key_embedding = nn.Embedding(problem.vocabulary_size, config.key_query_dim)
        self.value_embedding = nn.Embedding(problem.vocabulary_size, config.value_dim)
        self.register_buffer("fixed_query", torch.ones(config.key_query_dim))
        # Keep the structured extrema coordinates private while exposing every
        # additional value dimension as a shared channel from all heads.
        output_mask = torch.zeros(config.value_dim)
        output_mask[output_coordinate] = 1.0
        auxiliary_start = min(config.num_heads, config.value_dim)
        output_mask[auxiliary_start:] = 1.0
        self.register_buffer("output_mask", output_mask)

        token_order = torch.arange(
            problem.vocabulary_size,
            dtype=self.key_embedding.weight.dtype,
        )
        key_coordinates = (
            key_direction * config.initial_key_slope * token_order / math.sqrt(config.key_query_dim)
        )
        with torch.no_grad():
            self.key_embedding.weight.copy_(
                key_coordinates.unsqueeze(-1).expand_as(self.key_embedding.weight)
            )
            self.value_embedding.weight.zero_()
            self.value_embedding.weight[:, output_coordinate].copy_(
                torch.linspace(
                    -config.initial_value_amplitude,
                    config.initial_value_amplitude,
                    problem.vocabulary_size,
                )
            )
            if auxiliary_start < config.value_dim:
                shared_auxiliary_values = torch.linspace(
                    -config.initial_auxiliary_amplitude,
                    config.initial_auxiliary_amplitude,
                    problem.vocabulary_size,
                )
                self.value_embedding.weight[:, auxiliary_start:].copy_(
                    shared_auxiliary_values.unsqueeze(-1).expand(
                        -1, config.value_dim - auxiliary_start
                    )
                )
        parametrize.register_parametrization(
            self.key_embedding,
            "weight",
            _GridParametrization(config.precision_bits),
        )
        parametrize.register_parametrization(
            self.value_embedding,
            "weight",
            _BoundedGridParametrization(config.precision_bits, config.max_value_norm),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        embedding_indices = inputs - self.problem.min_value
        keys = self.key_embedding(embedding_indices)
        values = self.value_embedding(embedding_indices)

        scores = torch.einsum("bnd,d->bn", keys, self.fixed_query)
        scores = scores / math.sqrt(self.config.key_query_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.einsum("bn,bnd->bd", attention_weights, values)
        attention_output = attention_output * self.output_mask
        return attention_output, attention_weights


class MultiHeadFixedQueryAttention(nn.Module):
    """Run independent fixed-query heads and sum their value outputs."""

    def __init__(self, problem: ProblemConfig, config: ModelConfig) -> None:
        super().__init__()
        self.problem = problem
        self.config = config
        self.heads = nn.ModuleList(
            FixedQueryAttention(
                problem,
                config,
                key_direction=-1 if head_index % 2 == 0 else 1,
                output_coordinate=head_index % config.value_dim,
            )
            for head_index in range(config.num_heads)
        )

    def auxiliary_balance_loss(self) -> Tensor:
        """Penalize pairwise differences between shared auxiliary value tables."""

        auxiliary_start = min(self.config.num_heads, self.config.value_dim)
        if len(self.heads) < 2 or auxiliary_start >= self.config.value_dim:
            return self.heads[0].value_embedding.weight.sum() * 0.0

        auxiliary_tables = [head.value_embedding.weight[:, auxiliary_start:] for head in self.heads]
        pairwise_losses = [
            torch.mean((left - right).square())
            for index, left in enumerate(auxiliary_tables)
            for right in auxiliary_tables[index + 1 :]
        ]
        return torch.stack(pairwise_losses).mean()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        head_results = [head(inputs) for head in self.heads]
        attention_output = torch.stack([result[0] for result in head_results], dim=1).sum(dim=1)
        attention_weights = torch.stack([result[1] for result in head_results], dim=1)
        return attention_output, attention_weights


class MinMaxTransformer(nn.Module):
    """Classify minimum and maximum values from the summed attention output."""

    def __init__(
        self,
        problem: ProblemConfig | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.problem = problem or ProblemConfig()
        self.config = config or ModelConfig()
        self.attention = MultiHeadFixedQueryAttention(self.problem, self.config)
        self.classifier = nn.Linear(
            self.config.value_dim,
            self.problem.num_targets * self.problem.num_classes,
        )

    def auxiliary_balance_loss(self) -> Tensor:
        """Return the shared-coordinate value-table balance penalty."""

        return self.attention.auxiliary_balance_loss()

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
                f"input values must be between {self.problem.min_value} "
                f"and {self.problem.max_value}"
            )

    def forward(
        self,
        inputs: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(inputs)
        attention_output, attention_weights = self.attention(inputs)
        logits = self.classifier(attention_output).reshape(
            inputs.shape[0],
            self.problem.num_targets,
            self.problem.num_classes,
        )

        if return_attention:
            return logits, attention_weights, attention_output
        return logits
