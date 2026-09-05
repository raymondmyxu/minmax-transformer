"""Public API for the min/max transformer project."""

from minmax_transformer.config import ModelConfig, ProblemConfig, seed_everything
from minmax_transformer.data import (
    SyntheticBatchGenerator,
    class_to_target,
    compute_targets,
    target_to_class,
)
from minmax_transformer.model import (
    FixedQueryAttention,
    MinMaxTransformer,
    MultiHeadFixedQueryAttention,
    quantize_and_bound_values,
    quantize_to_grid,
)
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
    train_one_epoch,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationHistoryPoint",
    "FixedQueryAttention",
    "MinMaxTransformer",
    "ModelConfig",
    "MultiHeadFixedQueryAttention",
    "ProblemConfig",
    "SyntheticBatchGenerator",
    "TrainingConfig",
    "TrainingResult",
    "ValidationConfig",
    "class_to_target",
    "compute_targets",
    "evaluate",
    "load_checkpoint",
    "make_iid_data_loader",
    "quantize_and_bound_values",
    "quantize_to_grid",
    "resolve_device",
    "save_checkpoint",
    "save_evaluation_history",
    "seed_everything",
    "target_to_class",
    "train_one_epoch",
]
