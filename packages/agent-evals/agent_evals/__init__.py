"""Agent-agnostic evaluation: cases, datasets, variants, runs and the evidence they leave.

Nothing in this package imports an agent framework. An agent takes part by supplying a `RunOne`
callable; scoring, comparison and statistics work on what that callable returns.
"""

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
from .steps import GenericStep, GenericTrajectory

__all__ = [
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
    "RunOne",
    "VariantCaseResult",
    "redact_attributes",
]
