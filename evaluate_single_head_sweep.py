"""Build the final accuracy-vs-dimension evaluation without retraining.

The script combines selected best-epoch training metrics from the retained
diagnostic CSV files with the held-out metrics recorded when each completed
run was evaluated.  It writes a compact evaluation table and a single-axis
figure containing only minimum and maximum accuracies.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ARTIFACTS_DIR = Path("artifacts/single_head_adaptive_sweep")


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Metadata retained from one completed dimension experiment."""

    value_dimension: int
    selected_epoch: int
    validation_minimum_accuracy: float
    validation_maximum_accuracy: float
    training_protocol: str


# The validation values were printed and recorded when these independent
# 5,000-sample held-out evaluations completed.  The older d=3,5,7,9
# checkpoints were intentionally deleted after their diagnostics were saved,
# so these recorded values are the authoritative held-out results for them.
COMPLETED_RUNS = (
    CompletedRun(3, 30_000, 0.9860, 0.0946, "original adaptive dimension sweep"),
    CompletedRun(5, 265_000, 0.9360, 0.2320, "original adaptive dimension sweep"),
    CompletedRun(7, 110_000, 0.9342, 0.4216, "original adaptive dimension sweep"),
    CompletedRun(9, 495_000, 0.9206, 0.6502, "original adaptive dimension sweep"),
    CompletedRun(
        11,
        30_000,
        0.9234,
        0.9176,
        "distinct value features and cosine warm restarts",
    ),
)


@dataclass(frozen=True, slots=True)
class DimensionEvaluation:
    """Training and held-out coordinate accuracies for one value dimension."""

    value_dimension: int
    attention_heads: int
    precision_bits: int
    value_norm_bound: float
    sequence_length: int
    integer_range_max: int
    training_samples: int
    validation_samples: int
    selected_epoch: int
    training_minimum_accuracy: float
    training_maximum_accuracy: float
    validation_minimum_accuracy: float
    validation_maximum_accuracy: float
    training_protocol: str
    training_metric_source: str
    validation_metric_source: str


def _as_probability(value: str, *, field: str, path: Path) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} in {path} is not a probability: {probability}")
    return probability


def _read_selected_training_row(path: Path, epoch: int) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing retained diagnostics: {path}")
    with path.open(encoding="utf-8", newline="") as diagnostic_file:
        matches = [
            row
            for row in csv.DictReader(diagnostic_file)
            if int(row["epoch"]) == epoch
        ]
    if not matches:
        raise ValueError(f"epoch {epoch} is not present in {path}")
    return matches[-1]


def load_evaluations(artifacts_dir: str | Path) -> list[DimensionEvaluation]:
    """Load the five completed dimensions in increasing order."""

    directory = Path(artifacts_dir)
    evaluations: list[DimensionEvaluation] = []
    for run in COMPLETED_RUNS:
        diagnostics_path = directory / f"d{run.value_dimension}_training_diagnostics.csv"
        row = _read_selected_training_row(diagnostics_path, run.selected_epoch)
        training_samples = int(row["training_sample_count"])
        if training_samples != 4_000:
            raise ValueError(
                f"d={run.value_dimension} has {training_samples} training samples; expected 4000"
            )
        evaluations.append(
            DimensionEvaluation(
                value_dimension=run.value_dimension,
                attention_heads=1,
                precision_bits=3,
                value_norm_bound=16.0,
                sequence_length=10,
                integer_range_max=100,
                training_samples=training_samples,
                validation_samples=5_000,
                selected_epoch=run.selected_epoch,
                training_minimum_accuracy=_as_probability(
                    row["training_minimum_accuracy"],
                    field="training_minimum_accuracy",
                    path=diagnostics_path,
                ),
                training_maximum_accuracy=_as_probability(
                    row["training_maximum_accuracy"],
                    field="training_maximum_accuracy",
                    path=diagnostics_path,
                ),
                validation_minimum_accuracy=run.validation_minimum_accuracy,
                validation_maximum_accuracy=run.validation_maximum_accuracy,
                training_protocol=run.training_protocol,
                training_metric_source=f"{diagnostics_path.name}: epoch {run.selected_epoch}",
                validation_metric_source="recorded independent held-out evaluation",
            )
        )
    return evaluations


def save_evaluation_csv(
    path: str | Path,
    evaluations: list[DimensionEvaluation],
) -> Path:
    """Write the plot-ready evaluation table; joint accuracy is omitted."""

    if not evaluations:
        raise ValueError("cannot save an empty evaluation")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(evaluations[0]))
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(evaluation) for evaluation in evaluations)
    return output_path


def save_accuracy_plot(
    path: str | Path,
    evaluations: list[DimensionEvaluation],
    *,
    show: bool = False,
) -> Path:
    """Plot minimum/maximum training and validation accuracy on one axis."""

    if not evaluations:
        raise ValueError("cannot plot an empty evaluation")
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    ordered = sorted(evaluations, key=lambda evaluation: evaluation.value_dimension)
    dimensions = [evaluation.value_dimension for evaluation in ordered]
    minimum_color = "#2F6B9A"
    maximum_color = "#E07A1F"

    figure, axis = plt.subplots(figsize=(11.5, 6.8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FAFBFC")

    series = (
        ("training_minimum_accuracy", "Minimum — training", minimum_color, "-"),
        ("validation_minimum_accuracy", "Minimum — validation", minimum_color, "--"),
        ("training_maximum_accuracy", "Maximum — training", maximum_color, "-"),
        ("validation_maximum_accuracy", "Maximum — validation", maximum_color, "--"),
    )
    for attribute, label, color, line_style in series:
        axis.plot(
            dimensions,
            [getattr(evaluation, attribute) for evaluation in ordered],
            color=color,
            linestyle=line_style,
            linewidth=2.6,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=label,
            zorder=3,
        )

    # Emphasize the observed transition without implying measurements between
    # the discrete dimension settings.
    axis.axvspan(10.45, 11.55, color="#2E8B57", alpha=0.08, zorder=0)
    axis.annotate(
        "d=11: both coordinates are high\n"
        "train ≥ 99%  •  validation ≥ 91%",
        xy=(
            11,
            min(
                ordered[-1].validation_minimum_accuracy,
                ordered[-1].validation_maximum_accuracy,
            ),
        ),
        xytext=(8.2, 0.79),
        arrowprops={"arrowstyle": "->", "color": "#355A48", "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#7AA58D", "alpha": 0.96},
        color="#244737",
        fontsize=10.5,
        ha="left",
        va="center",
    )
    axis.annotate(
        "At small d, maximum accuracy is the bottleneck",
        xy=(3, ordered[0].validation_maximum_accuracy),
        xytext=(3.25, 0.29),
        arrowprops={"arrowstyle": "->", "color": "#9A5516", "lw": 1.3},
        color="#7A4312",
        fontsize=10.5,
        ha="left",
    )

    axis.set_title(
        "Single-head min/max accuracy increases with value dimension\n"
        "H=1, p=3, L=16",
        fontsize=15,
        fontweight="semibold",
        pad=15,
    )
    axis.set_xlabel("Value embedding dimension d", fontsize=11.5, labelpad=8)
    axis.set_ylabel("Accuracy", fontsize=11.5, labelpad=8)
    axis.set_xticks(dimensions)
    axis.set_xlim(min(dimensions) - 0.45, max(dimensions) + 0.6)
    axis.set_ylim(0.0, 1.035)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.9, alpha=0.85)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AEB7C2")

    legend_handles = [
        Line2D([0], [0], color=minimum_color, lw=2.6, label="Minimum"),
        Line2D([0], [0], color=maximum_color, lw=2.6, label="Maximum"),
        Line2D([0], [0], color="#4A4A4A", lw=2.6, linestyle="-", label="Training"),
        Line2D([0], [0], color="#4A4A4A", lw=2.6, linestyle="--", label="Validation"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.96,
        edgecolor="#D3D8DE",
    )
    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the final single-head dimension-sweep evaluation without training."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--show", action="store_true", help="also display the plot interactively")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluations = load_evaluations(args.artifacts_dir)
    summary_path = save_evaluation_csv(
        args.artifacts_dir / "dimension_sweep_summary.csv",
        evaluations,
    )
    plot_path = save_accuracy_plot(
        args.artifacts_dir / "accuracy_vs_value_dimension.png",
        evaluations,
        show=args.show,
    )
    print(f"saved evaluation summary: {summary_path.resolve()}")
    print(f"saved accuracy plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
