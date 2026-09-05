"""Evaluate a saved model on an independent finite IID validation draw."""

from __future__ import annotations

import argparse
from pathlib import Path

from minmax_transformer import (
    ValidationConfig,
    evaluate,
    load_checkpoint,
    make_iid_data_loader,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a checkpoint on an independent IID data draw."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/minmax_transformer.pt"),
    )
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2_001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validation_config = ValidationConfig(
            sample_count=args.samples,
            batch_size=args.batch_size,
            data_seed=args.data_seed,
        )
        device = resolve_device(args.device)
        model, metadata = load_checkpoint(args.checkpoint, device=device)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    data_loader = make_iid_data_loader(
        model.problem,
        sample_count=validation_config.sample_count,
        batch_size=validation_config.batch_size,
        data_seed=validation_config.data_seed,
        shuffle=False,
    )
    metrics = evaluate(model, data_loader, device=device)

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"device: {device}")
    print("input preprocessing: ascending sort within every vector")
    print(f"training samples recorded in checkpoint: {metadata['training_config']['sample_count']}")
    print(f"epochs completed: {metadata['training_result']['epochs_completed']}")
    print(f"training stop reason: {metadata['training_result']['stop_reason']}")
    print(f"validation samples: {metrics.sample_count} IID vectors")
    print(f"cross-entropy loss: {metrics.loss:.6f}")
    print(f"joint exact accuracy: {metrics.exact_accuracy:.4%}")
    print(f"minimum accuracy: {metrics.minimum_accuracy:.4%}")
    print(f"maximum accuracy: {metrics.maximum_accuracy:.4%}")
    print(f"extrema-coordinate mean absolute error: {metrics.mean_absolute_error:.4f}")


if __name__ == "__main__":
    main()
