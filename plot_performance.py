"""Plot extrema and overall accuracy from a training-history CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot train/validation extrema and overall accuracy against optimizer steps."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training_history.csv"),
        help="CSV file written by train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/accuracy_vs_training_steps.png"),
        help="Destination for the rendered PNG figure.",
    )
    parser.add_argument(
        "--title",
        default="Min/max and overall classification performance",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive Matplotlib window after saving.",
    )
    return parser


def load_history(path: str | Path) -> dict[str, list[float]]:
    """Load and validate the history columns needed for the accuracy plot."""

    history_path = Path(path)
    if not history_path.is_file():
        raise FileNotFoundError(f"history file does not exist: {history_path}")

    required_columns = (
        "optimizer_steps",
        "training_exact_accuracy",
        "validation_exact_accuracy",
        "training_minimum_accuracy",
        "validation_minimum_accuracy",
        "training_maximum_accuracy",
        "validation_maximum_accuracy",
    )
    columns = {name: [] for name in required_columns}
    with history_path.open(encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        missing = set(required_columns).difference(reader.fieldnames or ())
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"history CSV is missing columns: {missing_names}")
        for row in reader:
            for name in required_columns:
                columns[name].append(float(row[name]))

    if not columns["optimizer_steps"]:
        raise ValueError("history CSV contains no evaluation rows")
    return columns


def main() -> None:
    args = build_parser().parse_args()
    try:
        history = load_history(args.history)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    optimizer_steps = history["optimizer_steps"]
    figure, axis = plt.subplots(figsize=(11, 6))
    accuracy_series = (
        ("minimum", "Minimum", "tab:blue"),
        ("maximum", "Maximum", "tab:orange"),
        ("exact", "Overall", "tab:green"),
    )
    for column_name, display_name, color in accuracy_series:
        axis.plot(
            optimizer_steps,
            history[f"training_{column_name}_accuracy"],
            label=f"{display_name} — training",
            color=color,
            linestyle="-",
            linewidth=2,
        )
        axis.plot(
            optimizer_steps,
            history[f"validation_{column_name}_accuracy"],
            label=f"{display_name} — validation",
            color=color,
            linestyle="--",
            linewidth=2,
        )
    axis.set_title(args.title)
    axis.set_xlabel("Optimizer steps")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.01)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved performance plot: {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
