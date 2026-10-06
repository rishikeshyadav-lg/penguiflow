"""Local evaluation of a baseline and one advisory-skill candidate.

The evaluation machinery -- cases, datasets, variants, results, the run backend -- lives in
`agent_evals` and is re-exported here, so every existing import keeps working and names the same
classes. What stays here is the one rule that is specific to the learning control plane: a candidate
is a baseline plus an advisory skill, so the baseline must carry no skill and the candidate must.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_evals.core.evaluation import (
    EvaluationBackend,
    EvaluationCase,
    EvaluationDataset,
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
from agent_evals.core.evaluation import EvaluationRequest as PairedEvaluationRequest


@dataclass(frozen=True)
class EvaluationRequest(PairedEvaluationRequest):
    """A paired evaluation of a baseline against one candidate that adds an advisory skill."""

    def __post_init__(self) -> None:
        PairedEvaluationRequest.__post_init__(self)
        if self.baseline.advisory_skill is not None:
            raise ValueError("baseline must not include an advisory skill")
        if self.candidate.advisory_skill is None:
            raise ValueError("candidate must include an advisory skill")


__all__ = [
    "EvaluationBackend",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationRequest",
    "EvaluationVariant",
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
]
