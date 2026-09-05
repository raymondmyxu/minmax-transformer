"""Checks for the no-training final dimension-sweep evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluate_single_head_sweep import (
    COMPLETED_RUNS,
    load_evaluations,
    save_accuracy_plot,
    save_evaluation_csv,
)


def _write_diagnostics(directory: Path) -> None:
    fieldnames = [
        "epoch",
        "training_minimum_accuracy",
        "training_maximum_accuracy",
        "training_sample_count",
    ]
    for index, run in enumerate(COMPLETED_RUNS):
        with (directory / f"d{run.value_dimension}_training_diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as diagnostic_file:
            writer = csv.DictWriter(diagnostic_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": run.selected_epoch,
                    "training_minimum_accuracy": 0.99 - index * 0.001,
                    "training_maximum_accuracy": 0.10 + index * 0.20,
                    "training_sample_count": 4_000,
                }
            )


def test_loads_all_completed_dimensions_and_omits_joint_accuracy(tmp_path: Path) -> None:
    _write_diagnostics(tmp_path)

    evaluations = load_evaluations(tmp_path)
    output = save_evaluation_csv(tmp_path / "summary.csv", evaluations)

    with output.open(encoding="utf-8", newline="") as summary_file:
        reader = csv.DictReader(summary_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    assert [int(row["value_dimension"]) for row in rows] == [3, 5, 7, 9, 11]
    assert "training_minimum_accuracy" in fieldnames
    assert "validation_maximum_accuracy" in fieldnames
    assert all("joint" not in field for field in fieldnames)


def test_plot_is_generated_without_training(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _write_diagnostics(tmp_path)
    evaluations = load_evaluations(tmp_path)

    plot = save_accuracy_plot(tmp_path / "accuracy.png", evaluations)

    assert plot.read_bytes().startswith(b"\x89PNG")


def test_missing_selected_epoch_is_reported(tmp_path: Path) -> None:
    _write_diagnostics(tmp_path)
    path = tmp_path / "d3_training_diagnostics.csv"
    path.write_text(
        "epoch,training_minimum_accuracy,training_maximum_accuracy,training_sample_count\n"
        "1,1.0,0.1,4000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="epoch 30000"):
        load_evaluations(tmp_path)
