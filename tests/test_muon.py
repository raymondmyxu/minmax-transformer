"""Non-experiment checks for the self-contained two-head Muon optimizer."""

from __future__ import annotations

import torch
from torch import nn

from minmax_transformer import ClassificationMetrics, EvaluationHistoryPoint
from train import (
    Muon,
    build_parser,
    save_muon_accuracy_plot,
    zeropower_via_newton_schulz,
)


def test_muon_experiment_uses_isolated_artifact_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.num_heads == 2
    assert args.sequence_length == 10
    assert args.max_value == 100
    assert args.key_query_dim == 1
    assert args.value_dim == 3
    assert args.precision_bits == 3
    assert args.max_value_norm == 16.0
    assert args.samples == 5_000
    assert args.batch_size == 128
    assert args.max_epochs == 70_000
    assert args.eval_every == 100
    assert args.report_every == 2_000
    assert args.balance_regularization_strength == 1e-2
    assert args.checkpoint.as_posix() == "artifacts/minmax_transformer_muon.pt"
    assert args.history.as_posix() == "artifacts/training_history_muon.csv"
    assert args.plot.as_posix() == "artifacts/accuracy_vs_training_steps_muon.png"
    assert args.learning_rate == 1e-3
    assert args.muon_momentum == 0.95
    assert args.muon_nesterov is True
    assert args.muon_ns_steps == 5


def test_newton_schulz_preserves_shape_and_returns_finite_matrix() -> None:
    gradient = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    update = zeropower_via_newton_schulz(gradient)

    assert update.shape == gradient.shape
    assert update.dtype == gradient.dtype
    assert torch.isfinite(update).all()
    assert torch.count_nonzero(update) > 0


def test_muon_updates_matrices_and_uses_adamw_state_for_vector_bias() -> None:
    matrix = nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    bias = nn.Parameter(torch.tensor([0.5, -0.5]))
    optimizer = Muon([matrix, bias], lr=1e-3)
    original_matrix = matrix.detach().clone()
    original_bias = bias.detach().clone()

    loss = matrix.square().sum() + bias.square().sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(matrix, original_matrix)
    assert not torch.equal(bias, original_bias)
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "exp_avg" not in optimizer.state[matrix]
    assert "exp_avg" in optimizer.state[bias]
    assert "momentum_buffer" not in optimizer.state[bias]


def test_muon_plot_is_written_to_its_requested_path(tmp_path) -> None:
    metrics = ClassificationMetrics(
        loss=1.0,
        exact_accuracy=0.4,
        minimum_accuracy=0.5,
        maximum_accuracy=0.8,
        mean_absolute_error=1.0,
        sample_count=10,
    )
    output = tmp_path / "muon.png"

    saved = save_muon_accuracy_plot(
        output,
        [
            EvaluationHistoryPoint(
                epoch=1,
                optimizer_steps=4,
                training_metrics=metrics,
                validation_metrics=metrics,
            )
        ],
    )

    assert saved == output
    assert output.read_bytes().startswith(b"\x89PNG")
