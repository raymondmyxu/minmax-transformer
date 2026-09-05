"""Non-training checks for the adaptive single-head dimension sweep."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from minmax_transformer import (
    ClassificationMetrics,
    MinMaxTransformer,
    ProblemConfig,
    make_iid_data_loader,
    seed_everything,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "train_single_head_sweep.py"
MODULE_SPEC = importlib.util.spec_from_file_location("train_single_head_sweep", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
SWEEP_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = SWEEP_MODULE
MODULE_SPEC.loader.exec_module(SWEEP_MODULE)

ConvergenceConfig = SWEEP_MODULE.ConvergenceConfig
ConvergenceController = SWEEP_MODULE.ConvergenceController
DiagnosticRecord = SWEEP_MODULE.DiagnosticRecord
SweepRunResult = SWEEP_MODULE.SweepRunResult
build_parser = SWEEP_MODULE.build_parser
build_settings = SWEEP_MODULE.build_settings
count_trainable_parameters = SWEEP_MODULE.count_trainable_parameters
initialize_dimension_worker = SWEEP_MODULE.initialize_dimension_worker
initialize_distinct_value_features = SWEEP_MODULE.initialize_distinct_value_features
make_vectorized_iid_batches = SWEEP_MODULE.make_vectorized_iid_batches
save_diagnostic_history = SWEEP_MODULE.save_diagnostic_history
save_dimension_sweep_plot = SWEEP_MODULE.save_dimension_sweep_plot
save_sweep_summary = SWEEP_MODULE.save_sweep_summary
validate_value_dims = SWEEP_MODULE.validate_value_dims


def _metrics(
    *,
    loss: float = 1.0,
    joint: float = 0.5,
    minimum: float = 0.9,
    maximum: float = 0.55,
    maximum_loss: float = 1.2,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        loss=loss,
        exact_accuracy=joint,
        minimum_accuracy=minimum,
        maximum_accuracy=maximum,
        mean_absolute_error=2.0,
        sample_count=100,
        minimum_loss=0.1,
        maximum_loss=maximum_loss,
    )


def test_default_sweep_matches_the_requested_adaptive_experiment() -> None:
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert [config.value_dim for config in settings.model_configs] == [3, 5, 7, 9, 11, 13]
    assert all(config.num_heads == 1 for config in settings.model_configs)
    assert all(config.precision_bits == 3 for config in settings.model_configs)
    assert all(config.max_value_norm == 16.0 for config in settings.model_configs)
    assert settings.training.sample_count == 4_000
    assert settings.test.sample_count == 5_000
    assert settings.convergence.max_epochs == 500_000
    assert settings.convergence.diagnostic_interval == 5_000
    assert settings.convergence.report_interval == 10_000
    assert settings.convergence.maximum_learning_rate == 1e-3
    assert settings.convergence.minimum_learning_rate == 1e-4
    assert settings.convergence.restart_cycle_epochs == 50_000
    assert settings.convergence.exponential_average_decay == 0.8
    assert settings.convergence.cycle_accuracy_improvement == 0.0025
    assert settings.convergence.cycle_loss_improvement == 0.005
    assert settings.convergence.non_improving_cycles_to_stop == 2
    assert args.parallel_workers == 1
    assert not args.no_resume
    assert args.output_dir == Path("artifacts/single_head_adaptive_sweep")


def test_worker_initializer_configures_torch_threads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(torch, "set_num_threads", lambda count: calls.append(("intra", count)))
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda count: calls.append(("interop", count)),
    )

    initialize_dimension_worker(2)

    assert calls == [("intra", 2), ("interop", 2)]


@pytest.mark.parametrize("value_dims", [[], [0, 3], [-1], [3, 3]])
def test_value_dimensions_must_be_positive_and_distinct(value_dims: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_value_dims(value_dims)


def test_each_default_model_is_single_head_constrained_and_valid() -> None:
    settings = build_settings(build_parser().parse_args([]))
    inputs = torch.arange(1, 11, dtype=torch.long).unsqueeze(0)

    for config in settings.model_configs:
        seed_everything(settings.model_seed)
        model = MinMaxTransformer(settings.problem, config)
        logits, attention_weights, attention_output = model(inputs, return_attention=True)
        values = model.attention.heads[0].value_embedding.weight

        assert len(model.attention.heads) == 1
        assert logits.shape == (1, 2, 100)
        assert attention_weights.shape == (1, 1, 10)
        assert attention_output.shape == (1, config.value_dim)
        assert count_trainable_parameters(model) == 300 * config.value_dim + 300
        assert torch.linalg.vector_norm(values, dim=-1).max() <= 16.0 + 1e-6


def test_distinct_value_features_are_quantized_low_norm_and_nonidentical() -> None:
    settings = build_settings(build_parser().parse_args(["--value-dims", "11"]))
    model = MinMaxTransformer(settings.problem, settings.model_configs[0])

    initialize_distinct_value_features(model)

    values = model.attention.heads[0].value_embedding.weight
    step = settings.model_configs[0].quantization_step
    assert torch.allclose(values / step, torch.round(values / step), atol=1e-6)
    assert torch.linalg.vector_norm(values, dim=-1).max() <= 8.0 + 1e-6
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            assert not torch.equal(values[:, left], values[:, right])


def test_vectorized_batches_exactly_match_existing_loader_across_epochs() -> None:
    problem = ProblemConfig(sequence_length=4, max_value=12)
    existing = make_iid_data_loader(
        problem,
        sample_count=23,
        batch_size=5,
        data_seed=101,
        shuffle=True,
        shuffle_seed=102,
    )
    vectorized = make_vectorized_iid_batches(
        problem,
        sample_count=23,
        batch_size=5,
        data_seed=101,
        shuffle_seed=102,
    )
    for _ in range(2):
        for left, right in zip(list(existing), list(vectorized), strict=True):
            assert torch.equal(left[0], right[0])
            assert torch.equal(left[1], right[1])


def test_controller_requires_two_successful_checks() -> None:
    controller = ConvergenceController(ConvergenceConfig())

    first = controller.observe(
        5_000,
        _metrics(loss=0.1, joint=0.995, minimum=0.995, maximum=0.995),
    )
    second = controller.observe(
        10_000,
        _metrics(loss=0.09, joint=0.996, minimum=0.996, maximum=0.996),
    )

    assert first.stop_reason is None
    assert second.event == "accuracy_target_reached"
    assert second.stop_reason is not None


def test_cosine_schedule_restarts_at_maximum_learning_rate() -> None:
    config = ConvergenceConfig(
        max_epochs=100,
        diagnostic_interval=10,
        report_interval=10,
        restart_cycle_epochs=50,
    )

    assert config.learning_rate_at(1) == pytest.approx(1e-3)
    assert 1e-4 < config.learning_rate_at(25) < 1e-3
    assert config.learning_rate_at(50) == pytest.approx(1e-4)
    assert config.learning_rate_at(51) == pytest.approx(1e-3)


def test_controller_stops_after_two_complete_non_improving_cycles() -> None:
    config = ConvergenceConfig(
        max_epochs=40,
        diagnostic_interval=5,
        report_interval=10,
        restart_cycle_epochs=10,
        exponential_average_decay=0.0,
    )
    controller = ConvergenceController(config)

    first = controller.observe(10, _metrics(maximum=0.60, maximum_loss=1.0))
    second = controller.observe(20, _metrics(maximum=0.60, maximum_loss=1.0))
    third = controller.observe(30, _metrics(maximum=0.60, maximum_loss=1.0))

    assert first.event == "restart_cycle_baseline"
    assert second.event == "restart_cycle_no_improvement"
    assert second.non_improving_cycles == 1
    assert second.stop_reason is None
    assert third.event == "restart_convergence_reached"
    assert third.non_improving_cycles == 2
    assert third.stop_reason is not None


def test_controller_resets_non_improving_cycles_when_either_metric_improves() -> None:
    config = ConvergenceConfig(
        max_epochs=50,
        diagnostic_interval=5,
        report_interval=10,
        restart_cycle_epochs=10,
        exponential_average_decay=0.0,
    )
    controller = ConvergenceController(config)

    controller.observe(10, _metrics(maximum=0.60, maximum_loss=1.0))
    stalled = controller.observe(20, _metrics(maximum=0.60, maximum_loss=1.0))
    improved = controller.observe(30, _metrics(maximum=0.61, maximum_loss=1.0))

    assert stalled.non_improving_cycles == 1
    assert improved.event == "restart_cycle_improved"
    assert improved.non_improving_cycles == 0
    assert improved.stop_reason is None


def test_diagnostic_csv_summary_and_plot_are_generated_without_training(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    settings = build_settings(build_parser().parse_args([]))
    diagnostic = DiagnosticRecord(
        epoch=5_000,
        optimizer_steps=160_000,
        learning_rate=1e-3,
        event="diagnostic",
        completed_cycles=0,
        non_improving_cycles=0,
        smoothed_maximum_loss=1.2,
        smoothed_maximum_accuracy=0.55,
        metrics=_metrics(),
    )
    runs = []
    for index, value_dim in enumerate((3, 5, 7, 9, 11, 13)):
        history_path = save_diagnostic_history(tmp_path / f"d{value_dim}.csv", [diagnostic])
        runs.append(
            SweepRunResult(
                value_dim=value_dim,
                trainable_parameters=300 * value_dim + 300,
                epochs_completed=5_000,
                optimizer_steps=160_000,
                best_epoch=5_000,
                stop_reason="test fixture",
                training_metrics=_metrics(joint=0.5 + index * 0.01),
                test_metrics=_metrics(joint=0.4 + index * 0.01),
                diagnostic_history=(diagnostic,),
                history_path=history_path,
            )
        )

    summary = save_sweep_summary(tmp_path / "summary.csv", runs, settings)
    plot = save_dimension_sweep_plot(tmp_path / "plot.png", runs, settings, show=False)

    with summary.open(encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert [int(row["value_dim"]) for row in rows] == [3, 5, 7, 9, 11, 13]
    assert "training_minimum_accuracy" in rows[0]
    assert "test_maximum_accuracy" in rows[0]
    assert plot.read_bytes().startswith(b"\x89PNG")
