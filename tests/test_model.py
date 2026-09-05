"""Tests for the constrained fixed-query attention model."""

import pytest
import torch
from torch import nn

from minmax_transformer.config import ModelConfig, ProblemConfig, seed_everything
from minmax_transformer.data import SyntheticBatchGenerator
from minmax_transformer.model import FixedQueryAttention, MinMaxTransformer


def _assert_on_grid(values: torch.Tensor, step: float) -> None:
    lattice_coordinates = values / step
    assert torch.allclose(lattice_coordinates, lattice_coordinates.round(), atol=1e-6)


def test_forward_shapes_and_attention_normalization() -> None:
    problem = ProblemConfig()
    config = ModelConfig()
    model = MinMaxTransformer(problem, config)
    inputs, _ = SyntheticBatchGenerator(problem, seed=3).sample(8)

    logits, attention_weights, attention_output = model(inputs, return_attention=True)

    assert logits.shape == (8, 2, 100)
    assert attention_weights.shape == (8, 2, 10)
    assert attention_output.shape == (8, 3)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones((8, 2)), atol=1e-6)


def test_architecture_is_two_fixed_query_heads_and_one_linear_classifier() -> None:
    model = MinMaxTransformer()

    attention_modules = [
        module for module in model.modules() if isinstance(module, FixedQueryAttention)
    ]
    linear_modules = [module for module in model.modules() if isinstance(module, nn.Linear)]

    assert len(attention_modules) == 2
    assert len(linear_modules) == 1
    assert model.classifier.in_features == 3
    assert model.classifier.out_features == 200
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())


def test_classifier_uses_seeded_pytorch_random_initialization() -> None:
    seed_everything(17)
    first = MinMaxTransformer()
    seed_everything(19)
    second = MinMaxTransformer()

    assert not torch.equal(first.classifier.weight, second.classifier.weight)
    assert not torch.equal(first.classifier.bias, second.classifier.bias)
    assert torch.equal(
        first.attention.heads[0].value_embedding.weight,
        second.attention.heads[0].value_embedding.weight,
    )


def test_multi_head_attention_output_is_sum_of_individual_heads() -> None:
    model = MinMaxTransformer()
    inputs, _ = SyntheticBatchGenerator(seed=5).sample(4)

    with torch.no_grad():
        summed_output, stacked_weights = model.attention(inputs)
        individual_results = [head(inputs) for head in model.attention.heads]
        expected_output = torch.stack([result[0] for result in individual_results], dim=1).sum(
            dim=1
        )
        expected_weights = torch.stack([result[1] for result in individual_results], dim=1)

    assert torch.allclose(summed_output, expected_output)
    assert torch.allclose(stacked_weights, expected_weights)


def test_structured_initialization_selects_min_and_max_and_encodes_values() -> None:
    model = MinMaxTransformer()
    inputs, _ = SyntheticBatchGenerator(seed=7).sample(32)
    minimum_head, maximum_head = model.attention.heads

    minimum_keys = minimum_head.key_embedding.weight[:, 0]
    maximum_keys = maximum_head.key_embedding.weight[:, 0]
    minimum_values = minimum_head.value_embedding.weight[:, 0]
    maximum_values = maximum_head.value_embedding.weight[:, 1]
    minimum_auxiliary_values = minimum_head.value_embedding.weight[:, 2]
    maximum_auxiliary_values = maximum_head.value_embedding.weight[:, 2]
    assert torch.all(minimum_keys[1:] < minimum_keys[:-1])
    assert torch.all(maximum_keys[1:] > maximum_keys[:-1])
    assert torch.all(minimum_values[1:] > minimum_values[:-1])
    assert torch.equal(minimum_values, maximum_values)
    assert torch.count_nonzero(minimum_head.value_embedding.weight[:, 1]) == 0
    assert torch.count_nonzero(maximum_head.value_embedding.weight[:, 0]) == 0
    assert torch.equal(minimum_auxiliary_values, maximum_auxiliary_values)
    assert minimum_auxiliary_values[0].item() == -model.config.initial_auxiliary_amplitude
    assert minimum_auxiliary_values[-1].item() == model.config.initial_auxiliary_amplitude
    assert torch.equal(minimum_head.output_mask, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(maximum_head.output_mask, torch.tensor([0.0, 1.0, 1.0]))
    assert minimum_values[0].item() == -model.config.initial_value_amplitude
    assert minimum_values[-1].item() == model.config.initial_value_amplitude

    with torch.no_grad():
        _, attention_weights = model.attention(inputs)
    selected_tokens = torch.stack(
        [
            inputs.gather(1, attention_weights[:, head].argmax(dim=1, keepdim=True)).squeeze(1)
            for head in range(2)
        ],
        dim=1,
    )

    assert torch.equal(selected_tokens[:, 0], inputs.min(dim=1).values)
    assert torch.equal(selected_tokens[:, 1], inputs.max(dim=1).values)


def test_query_is_fixed_to_one_and_embeddings_and_classifier_are_trainable() -> None:
    model = MinMaxTransformer()

    assert all(torch.equal(head.fixed_query, torch.ones(1)) for head in model.attention.heads)
    assert all(not head.fixed_query.requires_grad for head in model.attention.heads)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    for head_index in range(2):
        assert (
            f"attention.heads.{head_index}.key_embedding.parametrizations.weight.original"
            in trainable_names
        )
        assert (
            f"attention.heads.{head_index}.value_embedding.parametrizations.weight.original"
            in trainable_names
        )
    assert "classifier.weight" in trainable_names
    assert "classifier.bias" in trainable_names


def test_key_and_value_embeddings_obey_precision_grid() -> None:
    config = ModelConfig(precision_bits=3)
    model = MinMaxTransformer(config=config)

    for head in model.attention.heads:
        _assert_on_grid(head.key_embedding.weight, config.quantization_step)
        _assert_on_grid(head.value_embedding.weight, config.quantization_step)


@pytest.mark.parametrize(
    ("value_dim", "max_norm"),
    [
        (1, 1.0),
        (3, 0.75),
        (7, 1.5),
    ],
)
def test_value_embeddings_obey_euclidean_norm_bound(value_dim: int, max_norm: float) -> None:
    config = ModelConfig(
        value_dim=value_dim,
        max_value_norm=max_norm,
        initial_value_amplitude=max_norm,
        initial_auxiliary_amplitude=0.0,
    )
    model = MinMaxTransformer(config=config)

    for head in model.attention.heads:
        norms = torch.linalg.vector_norm(head.value_embedding.weight, dim=-1)
        assert norms.max().item() <= max_norm + 1e-6
        _assert_on_grid(head.value_embedding.weight, config.quantization_step)


def test_value_constraint_still_holds_for_large_underlying_parameters() -> None:
    config = ModelConfig(
        value_dim=3,
        max_value_norm=1.0,
        initial_value_amplitude=1.0,
        initial_auxiliary_amplitude=0.0,
    )
    model = MinMaxTransformer(config=config)
    original = model.attention.heads[0].value_embedding.parametrizations.weight.original
    with torch.no_grad():
        original.uniform_(-100.0, 100.0)

    effective_values = model.attention.heads[0].value_embedding.weight
    norms = torch.linalg.vector_norm(effective_values, dim=-1)

    assert norms.max().item() <= config.max_value_norm + 1e-6
    _assert_on_grid(effective_values, config.quantization_step)


def test_key_query_dimension_is_used_and_query_stays_fixed() -> None:
    config = ModelConfig(key_query_dim=4)
    model = MinMaxTransformer(config=config)
    inputs, _ = SyntheticBatchGenerator(seed=13).sample(3)

    logits, attention_weights, attention_output = model(inputs, return_attention=True)

    assert all(head.key_embedding.embedding_dim == 4 for head in model.attention.heads)
    assert all(torch.equal(head.fixed_query, torch.ones(4)) for head in model.attention.heads)
    assert logits.shape == (3, 2, 100)
    assert attention_weights.shape == (3, 2, 10)
    assert attention_output.shape == (3, 3)


def test_logits_are_permutation_invariant() -> None:
    seed_everything(29)
    model = MinMaxTransformer()
    model.eval()
    inputs, _ = SyntheticBatchGenerator(seed=31).sample(16)
    permutation = torch.tensor([9, 2, 7, 0, 5, 1, 8, 4, 6, 3])

    with torch.no_grad():
        original = model(inputs)
        permuted = model(inputs[:, permutation])

    assert torch.allclose(original, permuted, atol=1e-5, rtol=1e-5)


def test_straight_through_constraints_preserve_finite_gradients() -> None:
    seed_everything(37)
    problem = ProblemConfig()
    model = MinMaxTransformer(problem)
    inputs, labels = SyntheticBatchGenerator(problem, seed=41).sample(32)

    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert all(gradient is not None for gradient in gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    for head_index in range(2):
        key_name = f"attention.heads.{head_index}.key_embedding.parametrizations.weight.original"
        value_name = (
            f"attention.heads.{head_index}.value_embedding.parametrizations.weight.original"
        )
        assert gradients[key_name].abs().sum() > 0
        assert gradients[value_name].abs().sum() > 0
        assert gradients[value_name][:, 2].abs().sum() > 0


def test_auxiliary_balance_loss_is_zero_for_symmetric_initialization_and_detects_drift() -> None:
    model = MinMaxTransformer()

    assert model.auxiliary_balance_loss().item() == 0.0

    maximum_auxiliary = model.attention.heads[1].value_embedding.parametrizations.weight.original[
        :, 2
    ]
    with torch.no_grad():
        maximum_auxiliary.add_(1.0)

    assert model.auxiliary_balance_loss().item() > 0.0


@pytest.mark.parametrize(
    ("inputs", "error_type"),
    [
        (torch.ones((2, 9), dtype=torch.long), ValueError),
        (torch.ones((2, 10), dtype=torch.float32), TypeError),
        (torch.zeros((2, 10), dtype=torch.long), ValueError),
        (torch.full((2, 10), 101, dtype=torch.long), ValueError),
    ],
)
def test_model_rejects_invalid_inputs(
    inputs: torch.Tensor,
    error_type: type[Exception],
) -> None:
    model = MinMaxTransformer()

    with pytest.raises(error_type):
        model(inputs)
