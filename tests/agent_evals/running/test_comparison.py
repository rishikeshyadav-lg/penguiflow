"""Evaluating one, two or many variants, with several scorers, on any kind of runner."""

from __future__ import annotations

import math

import pytest

from agent_evals import (
    ComparisonRequest,
    ComparisonRunner,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRequest,
    EvaluationVariant,
    EvidenceContext,
    EvidenceEvent,
    LocalEvaluationBackend,
    MetricSpecification,
    PredictionResult,
    ScoreResult,
)


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        "matrix",
        "v1",
        [EvaluationCase(f"c{n}", {"query": "x" * n}, expected=n) for n in (1, 2, 3, 4)],
    )


def _context(dataset: EvaluationDataset, evaluation_id: str = "ev-1") -> EvidenceContext:
    return EvidenceContext(
        agent_id="agent", deployment_digest="sha256:dep", evaluation_id=evaluation_id, dataset_version=dataset.version
    )


def _request(*variants: EvaluationVariant, baseline_id: str | None = None) -> ComparisonRequest:
    dataset = _dataset()
    return ComparisonRequest("ev-1", _context(dataset), dataset, variants, baseline_id=baseline_id)


def run_one(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
    bonus = float(variant.config.get("bonus", 0))
    return {"length": float(len(case.inputs["query"])) + bonus}


def accuracy(case: EvaluationCase, output: dict[str, float]) -> dict[str, float]:
    return {"accuracy": 1.0 if output["length"] == case.expected else 0.0}


def closeness(case: EvaluationCase, output: dict[str, float]) -> float:
    return -abs(output["length"] - float(case.expected))


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[EvidenceEvent] = []

    def emit(self, event: EvidenceEvent) -> bool:
        self.events.append(event)
        return True


async def test_three_variants_and_two_scorers_fill_a_full_matrix() -> None:
    request = _request(
        EvaluationVariant("plain"),
        EvaluationVariant("plus1", config={"bonus": 1}),
        EvaluationVariant("plus2", config={"bonus": 2}),
    )

    result = await ComparisonRunner().compare(request, run_one, [accuracy, closeness])

    assert result.mean_metrics("plain") == {"accuracy": 1.0, "closeness": 0.0}
    assert result.mean_metrics("plus1") == {"accuracy": 0.0, "closeness": -1.0}
    assert result.mean_metrics("plus2") == {"accuracy": 0.0, "closeness": -2.0}
    for case in result.case_results:
        assert set(case.by_variant) == {"plain", "plus1", "plus2"}


async def test_a_single_variant_is_a_plain_evaluation() -> None:
    result = await ComparisonRunner().compare(_request(EvaluationVariant("only")), run_one, accuracy)

    assert result.request.baseline_id is None
    assert result.mean_metrics("only") == {"accuracy": 1.0}
    assert result.failed_case_ids("only") == ()


@pytest.mark.parametrize("async_runner", [False, True], ids=["sync-runner", "async-runner"])
@pytest.mark.parametrize("async_scorer", [False, True], ids=["sync-scorer", "async-scorer"])
async def test_runners_and_scorers_may_each_be_sync_or_async(async_runner: bool, async_scorer: bool) -> None:
    async def async_run_one(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
        return run_one(case, variant)

    async def async_accuracy(case: EvaluationCase, output: dict[str, float]) -> dict[str, float]:
        return accuracy(case, output)

    runner = async_run_one if async_runner else run_one
    scorer = async_accuracy if async_scorer else accuracy

    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), runner, scorer)

    assert result.mean_metrics("v") == {"accuracy": 1.0}


async def test_a_run_that_raises_is_kept_as_a_failed_case_and_not_dropped() -> None:
    def flaky(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
        if case.case_id == "c2" and variant.variant_id == "b":
            raise RuntimeError("tool unavailable")
        return run_one(case, variant)

    result = await ComparisonRunner().compare(_request(EvaluationVariant("a"), EvaluationVariant("b")), flaky, accuracy)

    assert result.failed_case_ids("a") == ()
    assert result.failed_case_ids("b") == ("c2",)
    failed = result.results_for("b")[1]
    assert failed.error == "RuntimeError: tool unavailable"
    assert len(result.case_results) == 4  # the failed case is still a row
    assert result.mean_metrics("b") == {"accuracy": 1.0}  # means come from successful runs only


@pytest.mark.parametrize("status", ["failed", "cancelled", "paused"])
async def test_a_prediction_that_did_not_complete_is_a_failure_and_is_not_scored(status: str) -> None:
    scored: list[str] = []

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id == "c1":
            return PredictionResult(status=status, error="no answer" if status == "failed" else None)  # type: ignore[arg-type]
        return PredictionResult(answer="ok")

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        scored.append(case.case_id)
        return {"answered": 1.0}

    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), runner, scorer)

    assert result.failed_case_ids("v") == ("c1",)
    assert "c1" not in scored
    reason = result.results_for("v")[0].error
    assert reason is not None and reason.startswith(status)


async def test_a_non_finite_score_is_recorded_as_an_error() -> None:
    def scorer(case: EvaluationCase, output: dict[str, float]) -> dict[str, float]:
        return {"accuracy": math.nan if case.case_id == "c3" else 1.0}

    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), run_one, scorer)

    assert result.failed_case_ids("v") == ("c3",)
    assert result.results_for("v")[2].error == "ValueError: metric 'accuracy' must be finite"


async def test_two_scorers_that_produce_the_same_metric_name_fail_the_case_visibly() -> None:
    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), run_one, [accuracy, accuracy])

    assert result.failed_case_ids("v") == ("c1", "c2", "c3", "c4")
    assert result.results_for("v")[0].error == "ValueError: metric 'accuracy' was produced by more than one scorer"


async def test_a_scorer_that_returns_one_number_is_named_by_its_function() -> None:
    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), run_one, closeness)

    assert result.mean_metrics("v") == {"closeness": 0.0}


async def test_a_scorer_that_returns_one_number_and_has_no_name_fails_visibly() -> None:
    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), run_one, lambda case, output: 1.0)

    assert result.failed_case_ids("v") == ("c1", "c2", "c3", "c4")
    error = result.results_for("v")[0].error
    assert error is not None and "needs a name" in error


async def test_a_score_result_keeps_its_feedback_and_checks() -> None:
    def graded(case: EvaluationCase, output: dict[str, float]) -> ScoreResult:
        return ScoreResult(score=1.0, feedback="length matches", checks={"length_ok": True})

    result = await ComparisonRunner().compare(_request(EvaluationVariant("v")), run_one, graded)

    first = result.results_for("v")[0]
    assert first.metrics == {"graded": 1.0}
    assert first.score_details == {"graded": {"feedback": "length matches", "checks": {"length_ok": True}}}


async def test_the_paired_view_of_a_comparison_matches_a_paired_run() -> None:
    """The general path and the original paired path must agree on the same data."""

    dataset = _dataset()
    baseline = EvaluationVariant("baseline")
    candidate = EvaluationVariant("candidate", advisory_skill="Be exact.", config={"bonus": 1})

    def flaky(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
        if case.case_id == "c4" and variant.variant_id == "candidate":
            raise RuntimeError("boom")
        return run_one(case, variant)

    comparison = await ComparisonRunner().compare(
        ComparisonRequest("ev-1", _context(dataset), dataset, [baseline, candidate], baseline_id="baseline"),
        flaky,
        accuracy,
    )
    paired = await LocalEvaluationBackend().evaluate(
        EvaluationRequest("ev-1", _context(dataset), dataset, baseline, candidate), flaky, accuracy
    )

    view = comparison.as_paired("baseline", "candidate")
    assert view.case_results == paired.case_results
    assert view.mean_metrics("baseline") == paired.mean_metrics("baseline")
    assert view.mean_metrics("candidate") == paired.mean_metrics("candidate")
    assert view.incomplete_case_ids == paired.incomplete_case_ids == ("c4",)
    specification = MetricSpecification("accuracy")
    assert comparison.metric_summary(specification, baseline_id="baseline", candidate_id="candidate") == (
        paired.metric_summary(specification)
    )


async def test_cases_run_in_dataset_order_and_variants_in_request_order_within_each_case() -> None:
    calls: list[tuple[str, str]] = []

    def recording(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
        calls.append((case.case_id, variant.variant_id))
        return run_one(case, variant)

    await ComparisonRunner().compare(
        _request(EvaluationVariant("first"), EvaluationVariant("second")), recording, accuracy
    )

    assert calls == [
        ("c1", "first"), ("c1", "second"),
        ("c2", "first"), ("c2", "second"),
        ("c3", "first"), ("c3", "second"),
        ("c4", "first"), ("c4", "second"),
    ]  # fmt: skip


async def test_the_paired_backend_keeps_the_same_order_after_delegating_to_the_shared_engine() -> None:
    calls: list[tuple[str, str]] = []
    dataset = _dataset()

    def recording(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, float]:
        calls.append((case.case_id, variant.variant_id))
        return run_one(case, variant)

    request = EvaluationRequest(
        "ev-1", _context(dataset), dataset, EvaluationVariant("base"), EvaluationVariant("cand", advisory_skill="s")
    )
    await LocalEvaluationBackend().evaluate(request, recording, accuracy)

    assert calls[:4] == [("c1", "base"), ("c1", "cand"), ("c2", "base"), ("c2", "cand")]


async def test_a_comparison_emits_started_and_completed_evidence_naming_every_variant() -> None:
    sink = _RecordingSink()
    request = _request(EvaluationVariant("a"), EvaluationVariant("b", config={"bonus": 1}), baseline_id="a")

    await ComparisonRunner(evidence_sink=sink).compare(request, run_one, accuracy)

    started, completed = (event.record() for event in sink.events)
    assert started["event_type"] == "evaluation.started"
    assert started["attributes"]["variant_ids"] == ["a", "b"]
    assert started["attributes"]["baseline_variant_id"] == "a"
    assert completed["event_type"] == "evaluation.completed"
    assert completed["attributes"]["failed_case_count_by_variant"] == {"a": 0, "b": 0}
    assert completed["metrics"] == {"a.accuracy": 1.0, "b.accuracy": 0.0}


async def test_a_sink_that_raises_does_not_stop_the_evaluation() -> None:
    class ExplodingSink:
        def emit(self, event: EvidenceEvent) -> bool:
            raise RuntimeError("sink down")

    result = await ComparisonRunner(evidence_sink=ExplodingSink()).compare(
        _request(EvaluationVariant("v")), run_one, accuracy
    )

    assert result.mean_metrics("v") == {"accuracy": 1.0}


def test_a_comparison_needs_at_least_one_variant() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        _request()


def test_a_comparison_rejects_duplicate_variant_ids() -> None:
    with pytest.raises(ValueError, match="variant_id values must be unique"):
        _request(EvaluationVariant("same"), EvaluationVariant("same"))


def test_a_comparison_rejects_a_baseline_that_is_not_one_of_its_variants() -> None:
    with pytest.raises(ValueError, match="is not one of the variants"):
        _request(EvaluationVariant("a"), baseline_id="missing")


def test_a_comparison_rejects_evidence_for_another_evaluation() -> None:
    dataset = _dataset()

    with pytest.raises(ValueError, match="evidence_context.evaluation_id must match"):
        ComparisonRequest("ev-1", _context(dataset, evaluation_id="ev-2"), dataset, [EvaluationVariant("a")])


async def test_asking_for_an_unknown_variant_is_an_error_not_an_empty_answer() -> None:
    result = await ComparisonRunner().compare(_request(EvaluationVariant("a")), run_one, accuracy)

    with pytest.raises(ValueError, match="unknown evaluation variant: nope"):
        result.mean_metrics("nope")
    with pytest.raises(ValueError, match="unknown evaluation variant: nope"):
        result.as_paired("a", "nope")


def test_a_variant_carries_free_form_config_and_copies_it() -> None:
    settings = {"model": "small", "temperature": 0.0}

    variant = EvaluationVariant("v", config=settings)
    settings["model"] = "changed-later"

    assert variant.config == {"model": "small", "temperature": 0.0}
    assert EvaluationVariant("v").config == {}
    with pytest.raises(ValueError, match="config must be a mapping"):
        EvaluationVariant("v", config=["not", "a", "mapping"])  # type: ignore[arg-type]


def test_a_variant_can_still_be_built_the_old_positional_way() -> None:
    variant = EvaluationVariant("candidate", "Check the evidence.")

    assert variant.advisory_skill == "Check the evidence."
    assert variant.config == {}


def test_a_run_that_is_ok_cannot_carry_an_error_and_a_failed_run_must_say_why() -> None:
    with pytest.raises(ValueError, match="cannot carry an error"):
        PredictionResult(status="ok", error="oops")
    with pytest.raises(ValueError, match="must say why"):
        PredictionResult(status="failed")
    with pytest.raises(ValueError, match="status must be one of"):
        PredictionResult(status="finished")  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["latency_ms", "cost_usd"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_a_prediction_rejects_a_negative_or_non_finite_cost_or_latency(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        PredictionResult(**{field: value})  # type: ignore[arg-type]


def test_a_score_must_be_finite_and_its_checks_must_be_booleans() -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        ScoreResult(score=math.inf)
    with pytest.raises(ValueError, match="True or False"):
        ScoreResult(score=1.0, checks={"ok": "yes"})  # type: ignore[dict-item]
