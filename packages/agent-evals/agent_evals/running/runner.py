"""Run every case against every variant, keep every failure, and say what happened.

`run_cases` is the one place a case is run and scored. The paired `LocalEvaluationBackend` and the
general `ComparisonRunner` both use it, so a paired run and the paired view of a comparison agree by
construction. Cases run in dataset order and, within a case, variants run in request order; a
runner that keeps state between calls can rely on that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..core.evaluation import (
    EvaluationCase,
    EvaluationVariant,
    Metric,
    RunOne,
    VariantCaseResult,
    _await_value,
    _validated_metrics,
)
from ..core.evidence import EvidenceEvent, EvidenceSink
from ..core.prediction import PredictionResult, normalize_scores, scorer_name
from .comparison import CaseResult, ComparisonRequest, ComparisonResult

logger = logging.getLogger("agent_evals.runner")

Scorers = Metric | Sequence[Metric]


def _as_list(scorers: Scorers) -> list[Metric]:
    return [scorers] if callable(scorers) else list(scorers)


async def _score(
    case: EvaluationCase, output: Any, scorers: Sequence[Metric]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run every scorer on one output and merge what they say; a name used twice is an error."""

    merged: dict[str, float] = {}
    details: dict[str, Any] = {}
    for scorer in scorers:
        value = await _await_value(scorer(case, output))
        metrics, scorer_details = normalize_scores(value, scorer=scorer_name(scorer))
        for name in metrics:
            if name in merged:
                raise ValueError(f"metric {name!r} was produced by more than one scorer")
        merged.update(metrics)
        details.update(scorer_details)
    return _validated_metrics(merged), details


def _reported_usage(output: Any) -> dict[str, Any]:
    """Latency, cost and LLM usage the runner reported, if it returned a `PredictionResult`."""

    if not isinstance(output, PredictionResult):
        return {}
    return {"latency_ms": output.latency_ms, "cost_usd": output.cost_usd, "llm_usage": output.llm_usage}


async def run_case_variant(
    case: EvaluationCase, variant: EvaluationVariant, run_one: RunOne, scorers: Sequence[Metric]
) -> VariantCaseResult:
    """Run and score one case with one variant, recording a failure instead of raising it."""

    try:
        output = await _await_value(run_one(case, variant))
        reported = _reported_usage(output)
        if isinstance(output, PredictionResult) and not output.completed:
            reason = f"{output.status}: {output.error}" if output.error else output.status
            return VariantCaseResult(variant_id=variant.variant_id, output=output, error=reason, **reported)
        metrics, details = await _score(case, output, scorers)
        return VariantCaseResult(
            variant_id=variant.variant_id, output=output, metrics=metrics, score_details=details, **reported
        )
    except Exception as error:
        logger.info("Evaluation case failed", exc_info=True)
        return VariantCaseResult(variant_id=variant.variant_id, error=f"{type(error).__name__}: {error}")


async def run_cases(
    cases: Sequence[EvaluationCase], variants: Sequence[EvaluationVariant], run_one: RunOne, scorers: Scorers
) -> list[list[VariantCaseResult]]:
    """Run every case with every variant; one row per case, one entry per variant."""

    scorer_list = _as_list(scorers)
    rows: list[list[VariantCaseResult]] = []
    for case in cases:
        rows.append([await run_case_variant(case, variant, run_one, scorer_list) for variant in variants])
    return rows


async def run_comparison(request: ComparisonRequest, run_one: RunOne, scorers: Scorers) -> ComparisonResult:
    """Run a comparison and return its complete evidence."""

    rows = await run_cases(request.dataset.cases, request.variants, run_one, scorers)
    case_results = tuple(
        CaseResult(
            case_id=case.case_id,
            by_variant={variant.variant_id: result for variant, result in zip(request.variants, row, strict=True)},
        )
        for case, row in zip(request.dataset.cases, rows, strict=True)
    )
    return ComparisonResult(request=request, case_results=case_results)


class ComparisonRunner:
    """Run a comparison and emit evidence about it; failures are kept as evidence, never skipped."""

    def __init__(self, *, evidence_sink: EvidenceSink | None = None) -> None:
        self._evidence_sink = evidence_sink

    async def compare(self, request: ComparisonRequest, run_one: RunOne, scorers: Scorers) -> ComparisonResult:
        """Run every variant on every case and return the evidence."""

        self._emit_started(request)
        result = await run_comparison(request, run_one, scorers)
        self._emit_completed(result)
        return result

    def _emit_started(self, request: ComparisonRequest) -> None:
        attributes: dict[str, Any] = {
            "dataset_digest": request.dataset.manifest_digest,
            "case_count": len(request.dataset.cases),
            "variant_ids": list(request.variant_ids),
        }
        if request.baseline_id is not None:
            attributes["baseline_variant_id"] = request.baseline_id
        self._emit(
            EvidenceEvent(event_type="evaluation.started", context=request.evidence_context, attributes=attributes)
        )

    def _emit_completed(self, result: ComparisonResult) -> None:
        request = result.request
        failed = {variant_id: list(result.failed_case_ids(variant_id)) for variant_id in request.variant_ids}
        attributes: dict[str, Any] = {
            "dataset_digest": request.dataset.manifest_digest,
            "case_count": len(request.dataset.cases),
            "variant_ids": list(request.variant_ids),
            "failed_case_ids_by_variant": failed,
            "failed_case_count_by_variant": {variant_id: len(ids) for variant_id, ids in failed.items()},
        }
        if request.baseline_id is not None:
            attributes["baseline_variant_id"] = request.baseline_id
        metrics = {
            f"{variant_id}.{name}": value
            for variant_id in request.variant_ids
            for name, value in result.mean_metrics(variant_id).items()
        }
        self._emit(
            EvidenceEvent(
                event_type="evaluation.completed",
                context=request.evidence_context,
                attributes=attributes,
                metrics=metrics,
            )
        )

    def _emit(self, event: EvidenceEvent) -> None:
        if self._evidence_sink is None:
            return
        try:
            self._evidence_sink.emit(event)
        except Exception:
            logger.warning("Evaluation evidence emission failed", exc_info=True)


__all__ = [
    "ComparisonRunner",
    "Scorers",
    "run_case_variant",
    "run_cases",
    "run_comparison",
]
