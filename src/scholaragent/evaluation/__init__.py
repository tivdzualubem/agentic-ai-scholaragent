"""Evaluation tools for ScholarAgent."""

from scholaragent.evaluation.benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkError,
    load_benchmark,
)
from scholaragent.evaluation.metrics import (
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scholaragent.evaluation.runner import (
    CaseEvaluation,
    EvaluationSummary,
    evaluate_benchmark,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "BenchmarkError",
    "CaseEvaluation",
    "EvaluationSummary",
    "evaluate_benchmark",
    "load_benchmark",
    "mean",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
