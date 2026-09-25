from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from learning_control_plane.contracts.evidence import EvidenceContext
from learning_control_plane.control_plane import AdvisorySkillCandidate, LearningControlPlane
from learning_control_plane.control_plane.persistence import _metric_summary_from_payload, _metric_summary_payload
from learning_control_plane.control_plane.worker import OfflineEvaluationWorker
from learning_control_plane.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    LocalEvaluationBackend,
    MetricSpecification,
)
from learning_control_plane.evaluation.verification import (
    FinalAnswerRubricV1,
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.judging import (
    AgentRun,
    AgentStep,
    CombinedMetric,
    VerificationMetric,
    all_criteria,
    judge_rubric,
    outcome_metrics,
    verification_policy,
)
from learning_control_plane.judging.golden import GoldenCaseResult, GoldenReport

RUBRIC = judge_rubric(FinalAnswerRubricV1())
PASSED = {criterion.criterion_id: VerificationCheck(criterion.criterion_id, "passed") for criterion in RUBRIC.criteria}


def _judge(run: AgentRun, reference: Any) -> InvestigationVerification:
    """Synthetic judge: "right" verifies, "wrong" is a hard failure, "partial" fails softly, "ask" is handled."""

    answer = run.final_answer or ""
    if answer.startswith("right"):
        final = score_final_answer(PASSED, rubric=RUBRIC)
    elif answer.startswith("wrong"):
        final = score_final_answer(
            all_criteria("failed", "stated_values_incorrect", RUBRIC),
            hard_failure_codes=("primary_result_incorrect",),
            rubric=RUBRIC,
        )
    elif answer.startswith("partial"):
        checks = dict(PASSED)
        checks["completeness"] = VerificationCheck("completeness", "failed", ("required_values_missing",))
        final = score_final_answer(checks, rubric=RUBRIC)
    else:
        final = score_final_answer(
            all_criteria("not_applicable", "clarification_requested", RUBRIC),
            hard_failure_codes=("clarification_requested",),
            rubric=RUBRIC,
        )
    evidence = [SafeStepEvidence(0, "query_stock", result_checks=(VerificationCheck("tool_execution", "passed"),))]
    return InvestigationVerification(step_evidence=evidence, final_answer=final)


def _to_run(case: EvaluationCase, output: Any) -> AgentRun:
    return AgentRun(question=str(case.inputs["question"]), steps=(AgentStep("query_stock"),), final_answer=output)


METRIC = VerificationMetric(_judge, _to_run)
CONTEXT = EvidenceContext(agent_id="inventory-agent", deployment_digest="sha256:inventory-v1")
PASSING_GOLDEN = GoldenReport(results=(GoldenCaseResult("g1", "verified", "verified"),))
FAILING_GOLDEN = GoldenReport(results=(GoldenCaseResult("g1", "verified", "failed"),))


def _plane(answers: Mapping[str, tuple[str, str]], **policy_options: Any) -> tuple[LearningControlPlane, str, Any]:
    """A control plane with one job whose baseline and candidate give the listed answers per case."""

    plane = LearningControlPlane(
        policy=verification_policy("verification-v1", **policy_options),
        evaluation_backend=LocalEvaluationBackend(),
    )
    plane.register_candidate(
        AdvisorySkillCandidate(
            candidate_id="restock-skill", advisory_skill="Check every store.", source_trace_ids=("t",)
        ),
        CONTEXT,
    )
    dataset = EvaluationDataset(
        dataset_id="stock-heldout",
        version="v1",
        cases=tuple(EvaluationCase(case_id=case_id, inputs={"question": case_id}) for case_id in answers),
    )
    job = plane.create_job(candidate_id="restock-skill", evaluation_id="evaluation-1", context=CONTEXT, dataset=dataset)

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> str:
        baseline, candidate = answers[case.case_id]
        return candidate if variant.advisory_skill else baseline

    return plane, job.job_id, run_one


def _run_job(answers: Mapping[str, tuple[str, str]], **policy_options: Any) -> Any:
    plane, job_id, run_one = _plane(answers, **policy_options)
    return asyncio.run(plane.run_job(job_id, run_one, METRIC, golden_report=PASSING_GOLDEN))


def test_a_run_handled_correctly_on_one_arm_is_left_out_of_the_comparison() -> None:
    job = _run_job({"c1": ("ask", "right"), "c2": ("partial", "right"), "c3": ("right", "right")})

    summary = next(s for s in job.decision.metric_summaries if s.specification.name == "verified_success")
    assert summary.excluded_case_ids == ("c1",)
    assert [value.case_id for value in summary.paired_values] == ["c2", "c3"]
    assert job.state == "ready_for_review"


def test_a_run_handled_correctly_on_both_arms_is_left_out_of_the_comparison() -> None:
    job = _run_job({"c1": ("ask", "ask"), "c2": ("partial", "right")})

    summary = next(s for s in job.decision.metric_summaries if s.specification.name == "verified_success")
    assert summary.excluded_case_ids == ("c1",)
    assert job.decision.approved


def test_a_gate_where_every_run_was_handled_correctly_rejects_for_lack_of_judged_pairs() -> None:
    job = _run_job({"c1": ("ask", "ask"), "c2": ("ask", "right")})

    assert job.state == "rejected"
    assert "no judged pairs for metric verified_success" in job.decision.reasons


def test_more_verified_runs_bought_with_more_hard_failures_are_rejected() -> None:
    job = _run_job({"c1": ("partial", "right"), "c2": ("partial", "wrong")})

    assert job.state == "rejected"
    assert any(reason.startswith("protected metric regressed: hard_failure") for reason in job.decision.reasons)


def test_a_job_is_refused_before_any_run_without_a_golden_set() -> None:
    plane, job_id, _ = _plane({"c1": ("partial", "right")})
    calls: list[str] = []

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> str:
        calls.append(case.case_id)
        return "right"

    with pytest.raises(ValueError, match="requires a passing golden set"):
        asyncio.run(plane.run_job(job_id, run_one, METRIC))
    assert calls == []
    assert plane.get_job(job_id).state == "draft"


def test_a_job_is_refused_when_the_judge_failed_its_golden_set() -> None:
    plane, job_id, run_one = _plane({"c1": ("partial", "right")})

    with pytest.raises(ValueError, match="failed its golden set"):
        asyncio.run(plane.run_job(job_id, run_one, METRIC, golden_report=FAILING_GOLDEN))
    assert plane.get_job(job_id).attempt_count == 0


def test_a_policy_without_the_golden_requirement_runs_without_a_report() -> None:
    plane, job_id, run_one = _plane({"c1": ("partial", "right")}, require_golden_set=False)

    job = asyncio.run(plane.run_job(job_id, run_one, METRIC))

    assert job.state == "ready_for_review"


def test_the_worker_passes_its_golden_report_to_every_job() -> None:
    plane, _, run_one = _plane({"c1": ("partial", "right")})

    run = asyncio.run(
        OfflineEvaluationWorker(plane, run_one=run_one, metric=METRIC, golden_report=PASSING_GOLDEN).run_pending()
    )

    assert len(run.ready_for_review_job_ids) == 1


def test_an_agent_error_is_judged_and_a_hard_failure() -> None:
    final = score_final_answer(
        all_criteria("failed", "answer_truncated", RUBRIC), hard_failure_codes=("agent_error",), rubric=RUBRIC
    )

    metrics = outcome_metrics(InvestigationVerification(step_evidence=(), final_answer=final))

    assert metrics == {"verified_success": 0.0, "hard_failure": 1.0, "handled_correctly": 0.0, "judged": 1.0}


def test_a_verification_metric_uses_the_reference_when_given() -> None:
    seen: list[Any] = []

    async def reference(run: AgentRun) -> str:
        return "units=5"

    def judge(run: AgentRun, ref: Any) -> InvestigationVerification:
        seen.append(ref)
        return _judge(run, ref)

    metrics = asyncio.run(
        VerificationMetric(judge, _to_run, reference)(EvaluationCase("c1", inputs={"question": "q"}), "right")
    )

    assert seen == ["units=5"]
    assert metrics["verified_success"] == 1.0


def test_combined_metrics_merge_their_numbers() -> None:
    def latency(case: EvaluationCase, output: Any) -> dict[str, float]:
        return {"latency_ms": 120.0}

    metrics = asyncio.run(CombinedMetric(METRIC, latency)(EvaluationCase("c1", inputs={"question": "q"}), "right"))

    assert metrics["latency_ms"] == 120.0
    assert metrics["verified_success"] == 1.0


def test_combined_metrics_may_not_report_the_same_name_twice() -> None:
    with pytest.raises(ValueError, match="same names"):
        asyncio.run(CombinedMetric(METRIC, METRIC)(EvaluationCase("c1", inputs={"question": "q"}), "right"))


def test_a_combined_metric_needs_a_metric() -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        CombinedMetric()


def test_a_metric_denominator_and_its_exclusions_survive_persistence() -> None:
    plane, job_id, run_one = _plane({"c1": ("ask", "right"), "c2": ("partial", "right")})
    job = asyncio.run(plane.run_job(job_id, run_one, METRIC, golden_report=PASSING_GOLDEN))
    summary = next(s for s in job.decision.metric_summaries if s.specification.name == "verified_success")

    restored = _metric_summary_from_payload(_metric_summary_payload(summary))

    assert restored == summary
    assert restored.specification == MetricSpecification("verified_success", denominator="judged")


def test_a_pair_missing_its_denominator_is_a_missing_metric() -> None:
    def no_denominator(case: EvaluationCase, output: Any) -> dict[str, float]:
        return {"verified_success": 1.0, "hard_failure": 0.0}

    plane, job_id, run_one = _plane({"c1": ("right", "right")})

    job = asyncio.run(plane.run_job(job_id, run_one, no_denominator, golden_report=PASSING_GOLDEN))

    assert "missing metric verified_success for case c1" in job.decision.reasons
