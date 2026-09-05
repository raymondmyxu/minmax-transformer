"""Inspect one batch and one forward pass without performing training."""

from __future__ import annotations

import argparse

import torch

from minmax_transformer import (
    MinMaxTransformer,
    ModelConfig,
    ProblemConfig,
    SyntheticBatchGenerator,
    class_to_target,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the fixed-query attention model without training it."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--key-query-dim", type=int, default=1)
    parser.add_argument("--value-dim", type=int, default=3)
    parser.add_argument("--precision-bits", type=int, default=3)
    parser.add_argument("--max-value-norm", type=float, default=16.0)
    parser.add_argument("--initial-key-slope", type=float, default=0.25)
    parser.add_argument("--initial-value-amplitude", type=float, default=8.0)
    parser.add_argument("--initial-auxiliary-amplitude", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    seed_everything(args.seed)
    problem = ProblemConfig(
        sequence_length=args.sequence_length,
        max_value=args.max_value,
    )
    model_config = ModelConfig(
        num_heads=args.num_heads,
        key_query_dim=args.key_query_dim,
        value_dim=args.value_dim,
        precision_bits=args.precision_bits,
        max_value_norm=args.max_value_norm,
        initial_key_slope=args.initial_key_slope,
        initial_value_amplitude=args.initial_value_amplitude,
        initial_auxiliary_amplitude=args.initial_auxiliary_amplitude,
    )
    data = SyntheticBatchGenerator(problem=problem, seed=args.seed)
    model = MinMaxTransformer(problem=problem, config=model_config)
    model.eval()

    inputs, labels = data.sample(args.batch_size)
    inputs = inputs.sort(dim=-1).values
    with torch.no_grad():
        logits, attention_weights, attention_output = model(inputs, return_attention=True)

    predicted_extrema = class_to_target(logits.argmax(dim=-1), problem)
    target_extrema = class_to_target(labels, problem)
    effective_values = [head.value_embedding.weight for head in model.attention.heads]

    print(f"inputs shape: {tuple(inputs.shape)}")
    print(f"fixed queries: {[head.fixed_query.tolist() for head in model.attention.heads]}")
    print(f"attention weights shape: {tuple(attention_weights.shape)}")
    print(f"summed attention output shape: {tuple(attention_output.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"quantization step: {model_config.quantization_step}")
    largest_value_norm = max(values.norm(dim=-1).max().item() for values in effective_values)
    print(f"largest value-embedding norm: {largest_value_norm:.6f}")
    print(f"target extrema [min, max]: {target_extrema.tolist()}")
    print(f"untrained extrema predictions: {predicted_extrema.tolist()}")
    print("No optimizer step or training was performed.")


if __name__ == "__main__":
    main()
