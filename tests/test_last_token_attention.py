"""Non-training checks for the separate last-token-query experiment."""

from __future__ import annotations

import torch
from torch import nn

from minmax_transformer import ClassificationMetrics, ProblemConfig
from minmax_transformer.last_token_attention import (
    LastTokenAttentionConfig,
    LastTokenAttentionMinMax,
)
from train_last_token_attention import (
    ConvergenceController,
    ConvergenceSettings,
    build_parser,
    build_settings,
    make_fixed_iid_batches,
)


def _metrics(
    *,
    minimum_accuracy: float,
    maximum_accuracy: float,
    minimum_loss: float,
    maximum_loss: float,
    joint_accuracy: float = 0.5,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        loss=(minimum_loss + maximum_loss) / 2,
        exact_accuracy=joint_accuracy,
        minimum_accuracy=minimum_accuracy,
        maximum_accuracy=maximum_accuracy,
        mean_absolute_error=1.0,
        sample_count=100,
        minimum_loss=minimum_loss,
        maximum_loss=maximum_loss,
    )


def test_defaults_match_requested_last_token_experiment() -> None:
    args = build_parser().parse_args([])
    problem, model, training, convergence = build_settings(args)

    assert problem.sequence_length == 10
    assert problem.max_value == 100
    assert model.num_heads == 1
    assert model.key_query_dim == 3
    assert model.value_dim == 3
    assert model.precision_bits == 3
    assert model.max_value_norm == 16.0
    assert training.sample_count == 4_000
    assert training.test_sample_count == 5_000
    assert convergence.max_epochs == 500_000
    assert args.output_dir.as_posix() == "artifacts/last_token_model/d0_3"


def test_model_has_trainable_query_key_value_tables_and_one_linear_classifier() -> None:
    problem = ProblemConfig(sequence_length=4, max_value=10)
    config = LastTokenAttentionConfig()
    model = LastTokenAttentionMinMax(problem, config)

    assert model.query_embedding.weight.shape == (10, 3)
    assert model.key_embedding.weight.shape == (10, 3)
    assert model.value_embedding.weight.shape == (10, 3)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == 3
    assert model.classifier.out_features == 20
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_effective_query_key_value_embeddings_obey_requested_constraints() -> None:
    model = LastTokenAttentionMinMax()
    step = model.config.quantization_step
    for table in (
        model.query_embedding.weight,
        model.key_embedding.weight,
        model.value_embedding.weight,
    ):
        assert torch.allclose(table / step, torch.round(table / step), atol=1e-6)
    value_norms = torch.linalg.vector_norm(model.value_embedding.weight, dim=-1)
    assert value_norms.max() <= model.config.max_value_norm + 1e-6


def test_d0_three_initialization_is_distinct_but_preserves_initial_scores() -> None:
    problem = ProblemConfig(sequence_length=10, max_value=100)
    model = LastTokenAttentionMinMax(problem, LastTokenAttentionConfig())
    queries = model.query_embedding.weight
    keys = model.key_embedding.weight

    assert not torch.equal(queries[:, 0], queries[:, 1])
    assert not torch.equal(queries[:, 1], queries[:, 2])
    assert torch.count_nonzero(keys[:, 1:]) == 0

    first_last_token_scores = queries[0] @ keys.T / (3**0.5)
    final_last_token_scores = queries[-1] @ keys.T / (3**0.5)
    expected_scores = -0.25 * torch.arange(100, dtype=queries.dtype)
    assert torch.allclose(first_last_token_scores, expected_scores, atol=0.04)
    assert torch.equal(first_last_token_scores, final_last_token_scores)


def test_final_token_supplies_the_query_and_attends_to_every_position() -> None:
    problem = ProblemConfig(sequence_length=3, max_value=5)
    model = LastTokenAttentionMinMax(
        problem,
        LastTokenAttentionConfig(key_query_dim=1),
    )
    with torch.no_grad():
        model.query_embedding.parametrizations.weight.original.copy_(
            torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1)
        )
        model.key_embedding.parametrizations.weight.original.copy_(
            torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1)
        )

    inputs = torch.tensor([[1, 2, 3], [3, 2, 1]])
    _, weights, outputs = model(inputs, return_attention=True)
    expected_first = torch.softmax(torch.tensor([3.0, 6.0, 9.0]), dim=0)
    expected_second = torch.softmax(torch.tensor([3.0, 2.0, 1.0]), dim=0)

    assert weights.shape == (2, 3)
    assert outputs.shape == (2, 3)
    assert torch.allclose(weights[0], expected_first)
    assert torch.allclose(weights[1], expected_second)
    assert torch.all(weights > 0)


def test_iid_data_preserves_sequence_order_and_exact_extrema_labels() -> None:
    problem = ProblemConfig(sequence_length=10, max_value=100)
    batches = make_fixed_iid_batches(
        problem,
        sample_count=256,
        batch_size=64,
        data_seed=1_001,
        shuffle_seed=None,
    )

    assert torch.equal(batches.labels[:, 0], batches.inputs.amin(dim=-1) - 1)
    assert torch.equal(batches.labels[:, 1], batches.inputs.amax(dim=-1) - 1)
    assert torch.any(batches.inputs[:, -1] != batches.inputs.amax(dim=-1))
    assert torch.any(batches.inputs[:, 1:] < batches.inputs[:, :-1])


def test_warm_restart_and_controller_track_the_weaker_coordinate() -> None:
    settings = ConvergenceSettings(
        max_epochs=40,
        diagnostic_interval=5,
        report_interval=10,
        restart_cycle_epochs=10,
        exponential_average_decay=0.0,
    )
    controller = ConvergenceController(settings)
    first = controller.observe(
        10,
        _metrics(
            minimum_accuracy=0.9,
            maximum_accuracy=0.6,
            minimum_loss=0.2,
            maximum_loss=1.0,
        ),
    )
    second = controller.observe(
        20,
        _metrics(
            minimum_accuracy=0.9,
            maximum_accuracy=0.6,
            minimum_loss=0.2,
            maximum_loss=1.0,
        ),
    )

    assert settings.learning_rate_at(1) == 1e-3
    assert settings.learning_rate_at(10) == 1e-4
    assert settings.learning_rate_at(11) == 1e-3
    assert first.smoothed_worst_accuracy == 0.6
    assert first.smoothed_worst_loss == 1.0
    assert second.event == "restart_cycle_no_improvement"
    assert second.non_improving_cycles == 1
