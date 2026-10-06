"""Evaluate one variant, two, or many against the same fixed dataset.

The paired baseline-versus-candidate types in `evaluation.py` stay as they are. These types describe
the general case: any number of variants, each run on every case. A paired view of any two of them
is derived on demand, using the same paired types, so the numbers are the ones a paired run gives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..core.evaluation import (
    EvaluationDataset,
    EvaluationRequest,
    EvaluationVariant,
    MetricSpecification,
    MetricSummary,
    PairedCaseResult,
    PairedEvaluationResult,
    VariantCaseResult,
    _non_empty,
)
from ..core.evidence import EvidenceContext


@dataclass(frozen=True, slots=True)
class ComparisonRequest:
    """Define one evaluation of one or more variants against a pinned dataset.

    One variant is a plain evaluation. Two, with a `baseline_id`, are what a paired run compares.
    More are a matrix: paired views can be taken between any two.
    """

    evaluation_id: str
    evidence_context: EvidenceContext
    dataset: EvaluationDataset
    variants: Sequence[EvaluationVariant]
    baseline_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_id", _non_empty(self.evaluation_id, "evaluation_id"))
        variants = tuple(self.variants)
        if not variants:
            raise ValueError("an evaluation needs at least one variant")
        variant_ids = [variant.variant_id for variant in variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant_id values must be unique")
        object.__setattr__(self, "variants", variants)
        if self.baseline_id is not None and self.baseline_id not in variant_ids:
            raise ValueError(f"baseline_id {self.baseline_id!r} is not one of the variants")
        if self.evidence_context.evaluation_id != self.evaluation_id:
            raise ValueError("evidence_context.evaluation_id must match evaluation_id")
        if self.evidence_context.dataset_version != self.dataset.version:
            raise ValueError("evidence_context.dataset_version must match dataset.version")

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(variant.variant_id for variant in self.variants)

    def variant(self, variant_id: str) -> EvaluationVariant:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise ValueError(f"unknown evaluation variant: {variant_id}")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Every variant's result for one case."""

    case_id: str
    by_variant: Mapping[str, VariantCaseResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_variant", dict(self.by_variant))


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The complete evidence of one comparison, with failures kept rather than skipped."""

    request: ComparisonRequest
    case_results: Sequence[CaseResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_results", tuple(self.case_results))

    def results_for(self, variant_id: str) -> tuple[VariantCaseResult, ...]:
        """Return one variant's result for every case, in dataset order."""

        self.request.variant(variant_id)
        return tuple(case.by_variant[variant_id] for case in self.case_results)

    def mean_metrics(self, variant_id: str) -> dict[str, float]:
        """Return mean metrics over the successful runs of one variant."""

        metric_values: dict[str, list[float]] = {}
        for result in self.results_for(variant_id):
            if result.error is not None:
                continue
            for name, value in result.metrics.items():
                metric_values.setdefault(name, []).append(value)
        return {name: sum(values) / len(values) for name, values in metric_values.items()}

    def failed_case_ids(self, variant_id: str) -> tuple[str, ...]:
        """Return every case whose run or scoring failed for one variant."""

        self.request.variant(variant_id)
        return tuple(case.case_id for case in self.case_results if case.by_variant[variant_id].error is not None)

    def as_paired(self, baseline_id: str, candidate_id: str) -> PairedEvaluationResult:
        """Return the paired view of two variants, exactly as a paired run would have recorded it."""

        request = EvaluationRequest(
            evaluation_id=self.request.evaluation_id,
            evidence_context=self.request.evidence_context,
            dataset=self.request.dataset,
            baseline=self.request.variant(baseline_id),
            candidate=self.request.variant(candidate_id),
        )
        pairs = tuple(
            PairedCaseResult(
                case_id=case.case_id,
                baseline=case.by_variant[baseline_id],
                candidate=case.by_variant[candidate_id],
            )
            for case in self.case_results
        )
        return PairedEvaluationResult(request=request, case_results=pairs)

    def metric_summary(
        self, specification: MetricSpecification, *, baseline_id: str, candidate_id: str
    ) -> MetricSummary:
        """Return the paired metric summary between two variants."""

        return self.as_paired(baseline_id, candidate_id).metric_summary(specification)


__all__ = ["CaseResult", "ComparisonRequest", "ComparisonResult"]
