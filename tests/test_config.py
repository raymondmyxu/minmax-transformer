"""Tests for validated project configuration."""

import pytest

from minmax_transformer.config import ModelConfig, ProblemConfig, seed_everything


def test_problem_defaults_match_requested_task() -> None:
    problem = ProblemConfig()

    assert problem.sequence_length == 10
    assert problem.min_value == 1
    assert problem.max_value == 100
    assert problem.vocabulary_size == 100
    assert problem.num_targets == 2
    assert problem.num_classes == 100


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sequence_length": 1}, "sequence_length"),
        ({"min_value": -1}, "min_value"),
        ({"min_value": 10, "max_value": 10}, "max_value"),
    ],
)
def test_problem_rejects_invalid_values(
    arguments: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProblemConfig(**arguments)


def test_model_defaults_match_requested_attention_constraints() -> None:
    config = ModelConfig()

    assert config.num_heads == 2
    assert config.key_query_dim == 1
    assert config.value_dim == 3
    assert config.precision_bits == 3
    assert config.quantization_step == 0.125
    assert config.max_value_norm == 16.0
    assert config.initial_key_slope == 0.25
    assert config.initial_value_amplitude == 8.0
    assert config.initial_auxiliary_amplitude == 4.0


@pytest.mark.parametrize(
    ("arguments", "error_type"),
    [
        ({"num_heads": 0}, ValueError),
        ({"key_query_dim": 0}, ValueError),
        ({"value_dim": 0}, ValueError),
        ({"precision_bits": -1}, ValueError),
        ({"precision_bits": 1.5}, TypeError),
        ({"precision_bits": True}, TypeError),
        ({"max_value_norm": 0.0}, ValueError),
        ({"precision_bits": 3, "max_value_norm": 0.1}, ValueError),
        ({"initial_key_slope": 0.0}, ValueError),
        ({"initial_value_amplitude": 0.0}, ValueError),
        ({"initial_auxiliary_amplitude": -1.0}, ValueError),
        ({"max_value_norm": 1.0, "initial_value_amplitude": 2.0}, ValueError),
        (
            {
                "value_dim": 3,
                "max_value_norm": 8.5,
                "initial_value_amplitude": 8.0,
                "initial_auxiliary_amplitude": 4.0,
            },
            ValueError,
        ),
    ],
)
def test_model_rejects_invalid_values(
    arguments: dict[str, int | float | bool],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ModelConfig(**arguments)


def test_key_query_dimension_remains_configurable() -> None:
    config = ModelConfig(key_query_dim=5)

    assert config.key_query_dim == 5


def test_seed_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)
