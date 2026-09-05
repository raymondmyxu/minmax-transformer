"""Train the two-head fixed-query model with the Muon optimizer."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm.auto import tqdm

from minmax_transformer import (
    EvaluationHistoryPoint,
    MinMaxTransformer,
    ModelConfig,
    ProblemConfig,
    TrainingConfig,
    TrainingResult,
    ValidationConfig,
    evaluate,
    make_iid_data_loader,
    resolve_device,
    save_checkpoint,
    save_evaluation_history,
    seed_everything,
    train_one_epoch,
)


def zeropower_via_newton_schulz(
    gradient: Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Approximately orthogonalize a matrix using Muon's quintic iteration."""

    if gradient.ndim != 2:
        raise ValueError("Muon orthogonalization requires a 2D gradient")
    if not 1 <= steps < 100:
        raise ValueError("steps must be between 1 and 99")
    if eps <= 0:
        raise ValueError("eps must be positive")

    original_dtype = gradient.dtype
    matrix = gradient.to(dtype=torch.float32)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / matrix.norm().clamp_min(eps)

    # Coefficients used by the reference Muon implementation. The iteration
    # deliberately approximates U V^T rather than computing an expensive SVD.
    coefficient_a = 3.4445
    coefficient_b = -4.7750
    coefficient_c = 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.T
        gram_update = torch.addmm(
            gram,
            gram,
            gram,
            beta=coefficient_b,
            alpha=coefficient_c,
        )
        matrix = torch.addmm(
            matrix,
            gram_update,
            matrix,
            beta=coefficient_a,
        )

    if transposed:
        matrix = matrix.T
    return matrix.to(dtype=original_dtype)


class Muon(Optimizer):
    """Muon for matrix parameters with an AdamW fallback for vectors.

    This experiment intentionally sends every trainable 2D tensor—including
    embedding tables and the classifier matrix—through Muon. The model has no
    conventional hidden-layer matrices, so restricting Muon to hidden layers
    would not test a Muon update at all. The only non-matrix trainable tensor is
    the classifier bias, which uses the standard AdamW fallback.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        fallback_betas: tuple[float, float] = (0.9, 0.999),
        fallback_eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 1 <= ns_steps < 100:
            raise ValueError("ns_steps must be between 1 and 99")
        if eps <= 0 or fallback_eps <= 0:
            raise ValueError("optimizer epsilon values must be positive")
        if any(not 0 <= beta < 1 for beta in fallback_betas):
            raise ValueError("fallback AdamW betas must be in [0, 1)")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
            "fallback_betas": fallback_betas,
            "fallback_eps": fallback_eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if torch.is_complex(parameter):
                    raise RuntimeError("Muon does not support complex parameters")
                if parameter.ndim == 2:
                    self._step_matrix(parameter, gradient, group)
                else:
                    self._step_adamw_fallback(parameter, gradient, group)
        return loss

    def _step_matrix(
        self,
        parameter: Tensor,
        gradient: Tensor,
        group: dict[str, Any],
    ) -> None:
        state = self.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                gradient,
                memory_format=torch.preserve_format,
            )
        momentum_buffer = state["momentum_buffer"]
        momentum = group["momentum"]
        momentum_buffer.lerp_(gradient, 1.0 - momentum)
        update = (
            gradient.lerp(momentum_buffer, momentum)
            if group["nesterov"]
            else momentum_buffer
        )
        update = zeropower_via_newton_schulz(
            update,
            steps=group["ns_steps"],
            eps=group["eps"],
        )

        learning_rate = group["lr"]
        parameter.mul_(1.0 - learning_rate * group["weight_decay"])
        rows, columns = parameter.shape
        adjusted_learning_rate = learning_rate * math.sqrt(max(1.0, rows / columns))
        parameter.add_(update, alpha=-adjusted_learning_rate)

    def _step_adamw_fallback(
        self,
        parameter: Tensor,
        gradient: Tensor,
        group: dict[str, Any],
    ) -> None:
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(
                gradient,
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                gradient,
                memory_format=torch.preserve_format,
            )
        state["step"] += 1
        beta1, beta2 = group["fallback_betas"]
        exponential_average = state["exp_avg"]
        squared_exponential_average = state["exp_avg_sq"]
        exponential_average.lerp_(gradient, 1.0 - beta1)
        squared_exponential_average.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )

        learning_rate = group["lr"]
        parameter.mul_(1.0 - learning_rate * group["weight_decay"])
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denominator = squared_exponential_average.sqrt().div_(
            math.sqrt(bias_correction2)
        )
        denominator.add_(group["fallback_eps"])
        parameter.addcdiv_(
            exponential_average,
            denominator,
            value=-learning_rate / bias_correction1,
        )


def save_muon_accuracy_plot(
    path: str | Path,
    history: Iterable[EvaluationHistoryPoint],
) -> Path:
    """Save the Muon run's train/validation accuracy plot under its own name."""

    points = list(history)
    if not points:
        raise ValueError("cannot plot an empty evaluation history")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    optimizer_steps = [point.optimizer_steps for point in points]
    figure, axis = plt.subplots(figsize=(11, 6))
    accuracy_series = (
        ("minimum_accuracy", "Minimum", "tab:blue"),
        ("maximum_accuracy", "Maximum", "tab:orange"),
        ("exact_accuracy", "Overall", "tab:green"),
    )
    for metric_name, display_name, color in accuracy_series:
        axis.plot(
            optimizer_steps,
            [getattr(point.training_metrics, metric_name) for point in points],
            label=f"{display_name} — training",
            color=color,
            linestyle="-",
            linewidth=2,
        )
        axis.plot(
            optimizer_steps,
            [getattr(point.validation_metrics, metric_name) for point in points],
            label=f"{display_name} — validation",
            color=color,
            linestyle="--",
            linewidth=2,
        )
    axis.set_title("Two-head min/max classification with Muon")
    axis.set_xlabel("Optimizer steps")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.01)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train two-head fixed-query attention with Muon and categorical "
            "cross-entropy on a fixed IID data draw."
        )
    )
    parser.add_argument("--samples", type=int, default=5_000, help="Training sample count S.")
    parser.add_argument("--max-epochs", type=int, default=70_000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluate and record training/validation metrics every N epochs.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=2_000,
        help="Print training/validation accuracies every N epochs.",
    )
    parser.add_argument("--target-train-accuracy", type=float, default=0.99)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-eps", type=float, default=1e-7)
    parser.add_argument(
        "--muon-nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fallback-beta1", type=float, default=0.9)
    parser.add_argument("--fallback-beta2", type=float, default=0.999)
    parser.add_argument("--fallback-eps", type=float, default=1e-8)
    parser.add_argument(
        "--balance-regularization-strength",
        type=float,
        default=1e-2,
        help="Coefficient on the MSE between heads' shared auxiliary value tables.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--data-seed", type=int, default=1_001)
    parser.add_argument("--shuffle-seed", type=int, default=1_002)
    parser.add_argument("--validation-samples", type=int, default=5_000)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--validation-data-seed", type=int, default=2_001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/minmax_transformer_muon.pt"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training_history_muon.csv"),
        help="CSV destination for the Muon train/validation metrics.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("artifacts/accuracy_vs_training_steps_muon.png"),
        help="Destination for the Muon run's evaluation plot.",
    )

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
    parser = build_parser()
    args = parser.parse_args()
    try:
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
        training_config = TrainingConfig(
            sample_count=args.samples,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            evaluation_interval=args.eval_every,
            target_train_accuracy=args.target_train_accuracy,
            learning_rate=args.learning_rate,
            balance_regularization_strength=args.balance_regularization_strength,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            data_seed=args.data_seed,
            shuffle_seed=args.shuffle_seed,
        )
        validation_config = ValidationConfig(
            sample_count=args.validation_samples,
            batch_size=args.validation_batch_size,
            data_seed=args.validation_data_seed,
        )
        if not 0 <= args.muon_momentum < 1:
            raise ValueError("muon momentum must be in [0, 1)")
        if not 1 <= args.muon_ns_steps < 100:
            raise ValueError("muon NS steps must be between 1 and 99")
        if args.muon_eps <= 0 or args.fallback_eps <= 0:
            raise ValueError("optimizer epsilon values must be positive")
        if any(not 0 <= beta < 1 for beta in (args.fallback_beta1, args.fallback_beta2)):
            raise ValueError("fallback AdamW betas must be in [0, 1)")
        if args.report_every <= 0:
            raise ValueError("report interval must be positive")
        device = resolve_device(args.device)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    seed_everything(args.model_seed)
    data_loader = make_iid_data_loader(
        problem,
        sample_count=training_config.sample_count,
        batch_size=training_config.batch_size,
        data_seed=training_config.data_seed,
        shuffle=True,
        shuffle_seed=training_config.shuffle_seed,
    )
    training_evaluation_loader = make_iid_data_loader(
        problem,
        sample_count=training_config.sample_count,
        batch_size=training_config.batch_size,
        data_seed=training_config.data_seed,
        shuffle=False,
    )
    validation_loader = make_iid_data_loader(
        problem,
        sample_count=validation_config.sample_count,
        batch_size=validation_config.batch_size,
        data_seed=validation_config.data_seed,
        shuffle=False,
    )
    model = MinMaxTransformer(problem=problem, config=model_config).to(device)
    optimizer = Muon(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        momentum=args.muon_momentum,
        nesterov=args.muon_nesterov,
        ns_steps=args.muon_ns_steps,
        eps=args.muon_eps,
        fallback_betas=(args.fallback_beta1, args.fallback_beta2),
        fallback_eps=args.fallback_eps,
    )

    print(f"device: {device}")
    print(f"training samples: {training_config.sample_count} IID vectors")
    print(f"validation samples: {validation_config.sample_count} IID vectors")
    print(f"attention heads: {model_config.num_heads}")
    print("input preprocessing: ascending sort within every vector")
    print("loss: mean minimum/maximum categorical cross-entropy")
    print("classifier initialization: PyTorch default random linear initialization")
    print(
        "balance regularizer: "
        f"{training_config.balance_regularization_strength:g} * auxiliary-table MSE"
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    matrix_parameter_count = sum(parameter.ndim == 2 for parameter in trainable_parameters)
    fallback_parameter_count = len(trainable_parameters) - matrix_parameter_count
    print(
        "optimizer: Muon on "
        f"{matrix_parameter_count} matrix parameters; AdamW fallback on "
        f"{fallback_parameter_count} non-matrix parameter"
    )
    print(
        f"Muon settings: lr={training_config.learning_rate:g}, "
        f"momentum={args.muon_momentum:g}, nesterov={args.muon_nesterov}, "
        f"Newton-Schulz steps={args.muon_ns_steps}"
    )
    print(
        f"early stop: accuracy >= {training_config.target_train_accuracy:.2%} "
        f"(error <= {1.0 - training_config.target_train_accuracy:.2%})"
    )
    print(f"evaluation interval: every {training_config.evaluation_interval} epochs")
    progress = tqdm(
        range(1, training_config.max_epochs + 1),
        desc="training",
        unit="epoch",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    epochs_completed = 0
    steps_per_epoch = len(data_loader)
    evaluation_history: list[EvaluationHistoryPoint] = []
    final_metrics = None
    final_validation_metrics = None
    for epoch in progress:
        optimization_metrics = train_one_epoch(
            model,
            data_loader,
            optimizer,
            device=device,
            max_grad_norm=training_config.max_grad_norm,
            balance_regularization_strength=(training_config.balance_regularization_strength),
        )
        epochs_completed = epoch
        should_evaluate = (
            epoch == 1
            or epoch % training_config.evaluation_interval == 0
            or epoch % args.report_every == 0
            or epoch == training_config.max_epochs
        )
        if should_evaluate:
            final_metrics = evaluate(model, training_evaluation_loader, device=device)
            final_validation_metrics = evaluate(model, validation_loader, device=device)
            evaluation_history.append(
                EvaluationHistoryPoint(
                    epoch=epoch,
                    optimizer_steps=epoch * steps_per_epoch,
                    training_metrics=final_metrics,
                    validation_metrics=final_validation_metrics,
                )
            )
            progress.set_postfix(
                update_loss=f"{optimization_metrics.loss:.4f}",
                train_joint=f"{final_metrics.exact_accuracy:.2%}",
                val_joint=f"{final_validation_metrics.exact_accuracy:.2%}",
                train_min=f"{final_metrics.minimum_accuracy:.2%}",
                val_min=f"{final_validation_metrics.minimum_accuracy:.2%}",
                train_max=f"{final_metrics.maximum_accuracy:.2%}",
                val_max=f"{final_validation_metrics.maximum_accuracy:.2%}",
                refresh=False,
            )
            if epoch % args.report_every == 0:
                print(
                    f"epoch={epoch} "
                    f"training_minimum={final_metrics.minimum_accuracy:.2%} "
                    f"training_maximum={final_metrics.maximum_accuracy:.2%} "
                    f"training_joint={final_metrics.exact_accuracy:.2%} "
                    f"validation_minimum={final_validation_metrics.minimum_accuracy:.2%} "
                    f"validation_maximum={final_validation_metrics.maximum_accuracy:.2%} "
                    f"validation_joint={final_validation_metrics.exact_accuracy:.2%}",
                    flush=True,
                )
            if final_metrics.exact_accuracy >= training_config.target_train_accuracy:
                break
    progress.close()

    if final_metrics is None or final_validation_metrics is None:
        raise RuntimeError("training completed without an evaluation point")
    if final_metrics.exact_accuracy >= training_config.target_train_accuracy:
        stop_reason = (
            f"target training accuracy {training_config.target_train_accuracy:.2%} reached"
        )
    else:
        stop_reason = f"maximum epoch cap ({training_config.max_epochs}) reached"
    training_result = TrainingResult(
        epochs_completed=epochs_completed,
        stop_reason=stop_reason,
        final_training_metrics=final_metrics,
    )
    print(f"stop reason: {stop_reason}")
    print(
        f"final training metrics: loss={final_metrics.loss:.6f}, "
        f"joint accuracy={final_metrics.exact_accuracy:.2%}, "
        f"minimum accuracy={final_metrics.minimum_accuracy:.2%}, "
        f"maximum accuracy={final_metrics.maximum_accuracy:.2%}, "
        f"extrema MAE={final_metrics.mean_absolute_error:.4f}"
    )
    print(
        f"final validation metrics: loss={final_validation_metrics.loss:.6f}, "
        f"joint accuracy={final_validation_metrics.exact_accuracy:.2%}, "
        f"minimum accuracy={final_validation_metrics.minimum_accuracy:.2%}, "
        f"maximum accuracy={final_validation_metrics.maximum_accuracy:.2%}, "
        f"extrema MAE={final_validation_metrics.mean_absolute_error:.4f}"
    )
    print(f"final auxiliary balance MSE: {model.auxiliary_balance_loss().item():.6f}")

    history_path = save_evaluation_history(args.history, evaluation_history)
    print(f"saved evaluation history: {history_path.resolve()}")

    plot_path = save_muon_accuracy_plot(args.plot, evaluation_history)
    print(f"saved Muon evaluation plot: {plot_path.resolve()}")

    checkpoint_path = save_checkpoint(
        args.checkpoint,
        model,
        training_config=training_config,
        training_result=training_result,
        model_seed=args.model_seed,
        evaluation_history=evaluation_history,
    )
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_payload["optimizer_config"] = {
        "name": "Muon",
        "matrix_parameter_rule": "Muon",
        "non_matrix_parameter_rule": "AdamW",
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "momentum": args.muon_momentum,
        "nesterov": args.muon_nesterov,
        "newton_schulz_steps": args.muon_ns_steps,
        "muon_epsilon": args.muon_eps,
        "fallback_betas": (args.fallback_beta1, args.fallback_beta2),
        "fallback_epsilon": args.fallback_eps,
        "learning_rate_adjustment": "sqrt(max(1, rows / columns))",
    }
    torch.save(checkpoint_payload, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
