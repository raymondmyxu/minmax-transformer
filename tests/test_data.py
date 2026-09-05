"""Tests for synthetic input and label generation."""

import pytest
import torch

from minmax_transformer.config import ProblemConfig
from minmax_transformer.data import (
    SyntheticBatchGenerator,
    class_to_target,
    compute_targets,
    target_to_class,
)


def test_compute_targets_handles_edge_cases() -> None:
    inputs = torch.tensor(
        [
            [1, 100, 50, 50, 50, 50, 50, 50, 50, 50],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [30, 12, 30, 12, 20, 20, 20, 20, 20, 20],
        ]
    )

    assert torch.equal(
        compute_targets(inputs),
        torch.tensor([[1, 100], [7, 7], [12, 30]]),
    )


def test_class_mapping_round_trip() -> None:
    problem = ProblemConfig()
    targets = torch.tensor([[1, 1], [1, 100], [37, 82], [100, 100]])

    labels = target_to_class(targets, problem)

    assert torch.equal(labels[0], torch.tensor([0, 0]))
    assert torch.equal(labels[-1], torch.tensor([99, 99]))
    assert torch.equal(class_to_target(labels, problem), targets)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        torch.ones(10, dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    ],
)
def test_compute_targets_rejects_invalid_shapes(bad_inputs: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        compute_targets(bad_inputs)


def test_iid_batch_has_expected_shape_range_and_labels() -> None:
    problem = ProblemConfig()
    data = SyntheticBatchGenerator(problem, seed=11)

    inputs, labels = data.sample_iid(128)

    assert inputs.shape == (128, 10)
    assert inputs.dtype == torch.long
    assert labels.shape == (128, 2)
    assert inputs.min().item() >= 1
    assert inputs.max().item() <= 100
    assert torch.equal(class_to_target(labels, problem), compute_targets(inputs))


def test_balanced_batch_constructs_the_sampled_extrema() -> None:
    problem = ProblemConfig()
    data = SyntheticBatchGenerator(problem, seed=23)

    inputs, labels = data.sample_balanced(4096)
    targets = class_to_target(labels, problem)

    assert inputs.shape == (4096, 10)
    assert torch.equal(compute_targets(inputs), targets)
    assert labels.shape == (4096, 2)
    assert torch.unique(labels, dim=0).shape[0] > 1_000


def test_mixed_batch_is_reproducible_with_an_independent_generator() -> None:
    first = SyntheticBatchGenerator(seed=101, iid_fraction=0.4)
    second = SyntheticBatchGenerator(seed=101, iid_fraction=0.4)

    first_inputs, first_labels = first.sample(257)
    second_inputs, second_labels = second.sample(257)

    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)


@pytest.mark.parametrize("iid_fraction", [0.0, 1.0])
def test_mixed_batch_supports_each_pure_sampling_mode(iid_fraction: float) -> None:
    problem = ProblemConfig()
    data = SyntheticBatchGenerator(problem, seed=5, iid_fraction=iid_fraction)

    inputs, labels = data.sample(17)

    assert inputs.shape == (17, 10)
    assert torch.equal(class_to_target(labels, problem), compute_targets(inputs))


@pytest.mark.parametrize("method_name", ["sample", "sample_iid", "sample_balanced"])
def test_generators_reject_non_positive_batch_sizes(method_name: str) -> None:
    data = SyntheticBatchGenerator()

    with pytest.raises(ValueError, match="batch_size"):
        getattr(data, method_name)(0)


def test_generator_rejects_invalid_initialization() -> None:
    with pytest.raises(ValueError, match="iid_fraction"):
        SyntheticBatchGenerator(iid_fraction=1.1)
    with pytest.raises(ValueError, match="seed"):
        SyntheticBatchGenerator(seed=-1)
