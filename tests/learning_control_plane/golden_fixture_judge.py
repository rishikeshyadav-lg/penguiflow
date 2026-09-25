"""A tiny synthetic judge for the golden-set command-line tests."""

from __future__ import annotations

from typing import Any

from learning_control_plane.evaluation.verification import (
    FinalAnswerRubricV1,
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.judging import AgentRun, all_criteria, judge_rubric


def judge_by_answer(run: AgentRun, reference: Any) -> InvestigationVerification:
    """Verify an answer that states the reference's units, and hand a clarification back as handled."""

    rubric = judge_rubric(FinalAnswerRubricV1())
    answer = run.final_answer or ""
    if "which one" in answer.casefold():
        final = score_final_answer(
            all_criteria("not_applicable", "clarification_requested", rubric),
            hard_failure_codes=("clarification_requested",),
            rubric=rubric,
        )
    elif reference is not None and str(reference) in answer:
        final = score_final_answer(all_criteria_passed(), rubric=rubric)
    else:
        final = score_final_answer(all_criteria("failed", "stated_values_incorrect", rubric), rubric=rubric)
    evidence = [SafeStepEvidence(0, "query_stock", result_checks=(VerificationCheck("tool_execution", "passed"),))]
    return InvestigationVerification(step_evidence=evidence, final_answer=final)


def all_criteria_passed() -> dict[str, VerificationCheck]:
    return {
        criterion.criterion_id: VerificationCheck(criterion.criterion_id, "passed")
        for criterion in FinalAnswerRubricV1().criteria
    }


def build_judge() -> Any:
    return judge_by_answer


def build_reference() -> Any:
    async def units(run: AgentRun) -> str:
        return "1,200"

    return units
