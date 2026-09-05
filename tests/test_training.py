"""Non-training tests for data loading, evaluation, and checkpoint utilities."""

from dataclasses import asdict

import pytest
import torch
from torch.utils.data import TensorDataset

from minmax_transformer import MinMaxTransformer, ModelConfig, ProblemConfig
from minmax_transformer.data import class_to_target, compute_targets
from minmax_transformer.training import (
    ClassificationMetrics,
    EvaluationHistoryPoint,
    TrainingConfig,
    TrainingResult,
    ValidationConfig,
    evaluate,
    load_checkpoint,
    make_iid_data_loader,
    resolve_device,
    save_checkpoint,
    save_evaluation_history,
)


def test_training_and_validation_defaults_match_requested_sample_counts() -> None:
    training = TrainingConfig()
    validation = ValidationConfig()

    assert training.sample_count == 5_000
    assert training.max_epochs == 70_000
    assert training.evaluation_interval == 100
    assert training.target_train_accuracy == 0.99
    assert training.learning_rate == 1e-3
    assert training.balance_regularization_strength == 1e-2
    assert validation.sample_count == 5_000


@pytest.mark.parametrize(
    "arguments",
    [
        {"sample_count": 0},
        {"batch_size": 0},
        {"max_epochs": 0},
        {"evaluation_interval": 0},
        {"target_train_accuracy": 0.0},
        {"target_train_accuracy": 1.1},
        {"learning_rate": 0.0},
        {"balance_regularization_strength": -0.1},
        {"weight_decay": -1.0},
        {"max_grad_norm": 0.0},
        {"data_seed": -1},
        {"shuffle_seed": -1},
    ],
)
def test_training_config_rejects_invalid_values(arguments: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"sample_count": 0},
        {"batch_size": 0},
        {"data_seed": -1},
    ],
)
def test_validation_config_rejects_invalid_values(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ValidationConfig(**arguments)


def test_iid_loader_draws_exact_finite_reproducible_dataset() -> None:
    problem = ProblemConfig()
    first = make_iid_data_loader(
        problem,
        sample_count=37,
        batch_size=8,
        data_seed=55,
        shuffle=False,
    )
    second = make_iid_data_loader(
        problem,
        sample_count=37,
        batch_size=8,
        data_seed=55,
        shuffle=False,
    )

    assert isinstance(first.dataset, TensorDataset)
    first_inputs, first_labels = first.dataset.tensors
    second_inputs, second_labels = second.dataset.tensors
    assert first_inputs.shape == (37, 10)
    assert first_labels.shape == (37, 2)
    assert torch.all(first_inputs[:, :-1] <= first_inputs[:, 1:])
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(class_to_target(first_labels, problem), compute_targets(first_inputs))


def test_shuffled_loader_requires_seed() -> None:
    with pytest.raises(ValueError, match="shuffle_seed"):
        make_iid_data_loader(
            ProblemConfig(),
            sample_count=8,
            batch_size=4,
            data_seed=1,
            shuffle=True,
        )


def test_evaluation_reports_metrics_without_changing_parameters() -> None:
    model = MinMaxTransformer()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    data_loader = make_iid_data_loader(
        model.problem,
        sample_count=31,
        batch_size=7,
        data_seed=77,
        shuffle=False,
    )

    metrics = evaluate(model, data_loader, device=torch.device("cpu"))

    assert metrics.sample_count == 31
    assert metrics.loss > 0
    assert 0.0 <= metrics.exact_accuracy <= 1.0
    assert 0.0 <= metrics.minimum_accuracy <= 1.0
    assert 0.0 <= metrics.maximum_accuracy <= 1.0
    assert metrics.mean_absolute_error >= 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_untrained_checkpoint_round_trip_preserves_model_and_metadata(tmp_path) -> None:
    problem = ProblemConfig(sequence_length=5, max_value=20)
    model_config = ModelConfig(key_query_dim=2, value_dim=2)
    model = MinMaxTransformer(problem, model_config)
    training_config = TrainingConfig(
        sample_count=20,
        batch_size=5,
        max_epochs=3,
    )
    training_result = TrainingResult(
        epochs_completed=3,
        stop_reason="test checkpoint without training",
        final_training_metrics=evaluate(
            model,
            make_iid_data_loader(
                problem,
                sample_count=20,
                batch_size=5,
                data_seed=91,
                shuffle=False,
            ),
            device=torch.device("cpu"),
        ),
    )
    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        checkpoint,
        model,
        training_config=training_config,
        training_result=training_result,
        model_seed=19,
    )
    restored, metadata = load_checkpoint(checkpoint, device=torch.device("cpu"))

    assert asdict(restored.problem) == asdict(problem)
    assert asdict(restored.config) == asdict(model_config)
    assert metadata["training_config"] == asdict(training_config)
    assert metadata["training_result"] == asdict(training_result)
    assert metadata["evaluation_history"] == []
    assert metadata["model_seed"] == 19
    for name, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected)


def test_evaluation_history_is_saved_as_csv(tmp_path) -> None:
    metrics = ClassificationMetrics(
        loss=1.25,
        exact_accuracy=0.5,
        minimum_accuracy=0.6,
        maximum_accuracy=0.7,
        mean_absolute_error=0.8,
        sample_count=20,
        minimum_loss=1.0,
        maximum_loss=1.5,
    )
    point = EvaluationHistoryPoint(
        epoch=10,
        optimizer_steps=40,
        training_metrics=metrics,
        validation_metrics=metrics,
    )
    history_path = save_evaluation_history(tmp_path / "history.csv", [point])

    history_text = history_path.read_text(encoding="utf-8")
    assert "optimizer_steps,training_loss" in history_text
    assert "10,40,1.25,1.0,1.5,0.5,0.6,0.7,0.8,20" in history_text


def test_legacy_sum_checkpoint_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "legacy-sum.pt"
    torch.save(
        {
            "format_version": 2,
            "problem_config": asdict(ProblemConfig()),
            "model_config": asdict(ModelConfig()),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="predates the two-coordinate"):
        load_checkpoint(checkpoint, device=torch.device("cpu"))


def test_device_resolution_supports_cpu_and_rejects_unknown_name() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="device"):
        resolve_device("quantum")
