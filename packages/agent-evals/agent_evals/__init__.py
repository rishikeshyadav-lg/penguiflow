"""Agent-agnostic evaluation: cases, datasets, variants, runs and the evidence they leave.

Nothing in this package imports an agent framework. An agent takes part by supplying a `RunOne`
callable; scoring, comparison and statistics work on what that callable returns.
"""

from .comparison import CaseResult, ComparisonRequest, ComparisonResult
from .evaluation import (
    EvaluationBackend,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRequest,
    EvaluationVariant,
    LocalEvaluationBackend,
    Metric,
    MetricDirection,
    MetricSpecification,
    MetricSummary,
    PairedCaseResult,
    PairedEvaluationResult,
    PairedMetricValue,
    RunOne,
    VariantCaseResult,
)
from .evidence import EvidenceContext, EvidenceEvent, EvidenceSink, redact_attributes
from .prediction import PredictionResult, PredictionStatus, ScoreResult, ScoreValue, normalize_scores, scorer_name
from .runner import ComparisonRunner, Scorers, run_case_variant, run_cases, run_comparison
from .steps import GenericStep, GenericTrajectory

__all__ = [
    "CaseResult",
    "ComparisonRequest",
    "ComparisonResult",
    "ComparisonRunner",
    "EvaluationBackend",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationRequest",
    "EvaluationVariant",
    "EvidenceContext",
    "EvidenceEvent",
    "EvidenceSink",
    "GenericStep",
    "GenericTrajectory",
    "LocalEvaluationBackend",
    "Metric",
    "MetricDirection",
    "MetricSpecification",
    "MetricSummary",
    "PairedCaseResult",
    "PairedEvaluationResult",
    "PairedMetricValue",
    "PredictionResult",
    "PredictionStatus",
    "RunOne",
    "ScoreResult",
    "ScoreValue",
    "Scorers",
    "VariantCaseResult",
    "normalize_scores",
    "redact_attributes",
    "run_case_variant",
    "run_cases",
    "run_comparison",
    "scorer_name",
]
