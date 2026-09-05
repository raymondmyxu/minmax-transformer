"""Tests for the PyCharm-friendly command-line entry points."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minmax_transformer import (
    ClassificationMetrics,
    EvaluationHistoryPoint,
    save_evaluation_history,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script", "expected_text"),
    [
        ("train.py", "--samples"),
        ("validate.py", "--checkpoint"),
        ("plot_performance.py", "--history"),
    ],
)
def test_script_help_does_not_start_a_job(script: str, expected_text: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_text in completed.stdout


def test_training_script_runs_single_stage_and_records_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "single-stage.pt"
    history = tmp_path / "history.csv"
    plot = tmp_path / "muon-performance.png"
    completed = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--samples",
            "200",
            "--batch-size",
            "200",
            "--max-epochs",
            "4",
            "--target-train-accuracy",
            "1.0",
            "--eval-every",
            "2",
            "--validation-samples",
            "100",
            "--validation-batch-size",
            "100",
            "--sequence-length",
            "3",
            "--max-value",
            "10",
            "--device",
            "cpu",
            "--no-progress",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--plot",
            str(plot),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = payload["training_result"]
    assert result["epochs_completed"] == 4
    assert set(result) == {
        "epochs_completed",
        "stop_reason",
        "final_training_metrics",
    }
    assert [point["epoch"] for point in payload["evaluation_history"]] == [1, 2, 4]
    assert payload["optimizer_config"]["name"] == "Muon"
    assert payload["optimizer_config"]["matrix_parameter_rule"] == "Muon"
    assert payload["optimizer_config"]["non_matrix_parameter_rule"] == "AdamW"
    assert history.is_file()
    assert plot.read_bytes().startswith(b"\x89PNG")
    assert "attention heads: 2" in completed.stdout
    assert "Muon on 5 matrix parameters" in completed.stdout
    assert "AdamW fallback on 1 non-matrix parameter" in completed.stdout
    assert "PyTorch default random linear initialization" in completed.stdout


def test_performance_plot_is_rendered_from_saved_history(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    metrics = ClassificationMetrics(
        loss=1.0,
        exact_accuracy=0.4,
        minimum_accuracy=0.5,
        maximum_accuracy=0.8,
        mean_absolute_error=1.0,
        sample_count=10,
    )
    history = tmp_path / "history.csv"
    output = tmp_path / "performance.png"
    save_evaluation_history(
        history,
        [
            EvaluationHistoryPoint(
                epoch=1,
                optimizer_steps=4,
                training_metrics=metrics,
                validation_metrics=metrics,
            )
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "plot_performance.py",
            "--history",
            str(history),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes().startswith(b"\x89PNG")
