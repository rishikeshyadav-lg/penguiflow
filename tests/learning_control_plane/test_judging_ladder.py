from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from learning_control_plane.evaluation.verification import (
    FinalAnswerRubricV1,
    RubricCriterion,
    SafeStepEvidence,
    VerificationCheck,
)
from learning_control_plane.judging import (
    AgentRun,
    AgentStep,
    MeaningCheck,
    MeaningQuestion,
    OutcomeLadder,
    RubricJudgment,
    outcome_of,
)

PASSED = RubricJudgment(
    numerical=VerificationCheck("factual_numerical_correctness", "passed", ("stated_values_match_reference",)),
    completeness=VerificationCheck("completeness", "passed", ("all_required_values_stated",)),
    scope=VerificationCheck("scope_correctness", "passed", ("requested_scope_present",)),
    grounding=VerificationCheck("evidence_grounding", "passed", ("values_match_independent_reference",)),
)
SCOPE_ONLY = RubricJudgment(
    numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("no_required_values_stated",)),
    completeness=VerificationCheck("completeness", "not_applicable", ("nothing_to_complete",)),
    scope=VerificationCheck("scope_correctness", "passed", ("requested_scope_present",)),
    grounding=VerificationCheck("evidence_grounding", "not_applicable", ("no_matched_values_to_ground",)),
)


@dataclass
class _InventoryJudge:
    judgment: RubricJudgment | None = PASSED
    candidates: Sequence[str] = ()
    service_off: bool = False
    references: list[Any] = field(default_factory=list)

    def step_evidence(self, run: AgentRun) -> Sequence[SafeStepEvidence]:
        return [
            SafeStepEvidence(index, step.tool, result_checks=(VerificationCheck("tool_execution", "passed"),))
            for index, step in enumerate(run.steps)
        ]

    def clarification_candidates(self, run: AgentRun) -> Sequence[str]:
        return self.candidates

    def service_unavailable(self, run: AgentRun) -> bool:
        return self.service_off

    def judge(self, run: AgentRun, reference: Any) -> RubricJudgment | None:
        self.references.append(reference)
        return self.judgment


class _FixedMeaningJudge:
    def __init__(self, probabilities: Mapping[str, float] | Exception) -> None:
        self._probabilities = probabilities

    def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> Mapping[str, float]:
        if isinstance(self._probabilities, Exception):
            raise self._probabilities
        return self._probabilities


def _run(answer: str | None = "North Store sold 1,200 units.") -> AgentRun:
    return AgentRun(
        question="How many units did North Store sell?",
        steps=(AgentStep("query_stock", {"store": "North Store"}, {"rows": [{"units": 1200}]}),),
        final_answer=answer,
    )


def test_a_matched_answer_is_verified_and_records_its_outcome() -> None:
    domain = _InventoryJudge()

    verification = OutcomeLadder(domain).judge(_run(), reference={"units": 1200})

    assert verification.verified_success
    assert verification.outcome == "verified"
    assert verification.record()["outcome"] == "verified"
    assert domain.references == [{"units": 1200}]


def test_a_missing_answer_fails() -> None:
    verification = OutcomeLadder(_InventoryJudge()).judge(_run(answer=" "))

    assert verification.outcome == "failed"
    assert verification.final_answer is not None
    assert verification.final_answer.hard_failure_codes == ("primary_result_incorrect",)


def test_a_truncated_answer_is_an_agent_error_before_any_value_is_read() -> None:
    domain = _InventoryJudge()

    verification = OutcomeLadder(domain).judge(_run("North Store sold 1,200 units and the"))

    assert verification.outcome == "agent_error"
    assert domain.references == []


def test_asking_which_store_was_meant_is_handled_correctly() -> None:
    domain = _InventoryJudge(candidates=["North Store", "North Outlet"])
    answer = "I found North Store and North Outlet. Which one do you mean?"

    verification = OutcomeLadder(domain).judge(_run(answer))

    assert verification.outcome == "handled_correctly"
    assert not verification.verified_success


def test_a_switched_off_service_is_handled_correctly() -> None:
    verification = OutcomeLadder(_InventoryJudge(service_off=True)).judge(_run())

    assert verification.outcome == "handled_correctly"
    assert verification.final_answer is not None
    assert verification.final_answer.hard_failure_codes == ("service_unavailable",)


def test_a_question_the_domain_cannot_check_is_not_judgeable() -> None:
    verification = OutcomeLadder(_InventoryJudge(judgment=None)).judge(_run())

    assert verification.outcome == "not_judgeable"


def test_a_meaning_finding_fails_an_otherwise_verified_run() -> None:
    meaning = MeaningCheck(_FixedMeaningJudge({"contradiction": 0.95}), reads=1)

    verification = OutcomeLadder(_InventoryJudge(), meaning).judge(_run())

    assert verification.outcome == "failed"
    assert verification.final_answer is not None
    assert "contradicts_tool_results" in verification.final_answer.hard_failure_codes


def test_supplied_meaning_findings_replace_running_the_check() -> None:
    meaning = MeaningCheck(_FixedMeaningJudge({"contradiction": 0.95}), reads=1)

    verification = OutcomeLadder(_InventoryJudge(), meaning).judge(_run(), meaning_findings=())

    assert verification.outcome == "verified"


def test_a_meaning_check_that_breaks_never_fails_the_run() -> None:
    class _BrokenCheck(MeaningCheck):
        def check(self, question: str, shown_answer: str, steps: Sequence[AgentStep]) -> Any:
            raise RuntimeError("meaning service misconfigured")

    broken = _BrokenCheck(_FixedMeaningJudge({}), reads=1)

    verification = OutcomeLadder(_InventoryJudge(), broken).judge(_run())

    assert verification.outcome == "verified"


def test_a_confirmed_no_data_answer_cannot_be_failed_for_not_answering() -> None:
    no_data = RubricJudgment(
        numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("scope_has_no_data",)),
        completeness=VerificationCheck("completeness", "not_applicable", ("scope_has_no_data",)),
        scope=VerificationCheck("scope_correctness", "not_applicable", ("scope_has_no_data",)),
        grounding=VerificationCheck("evidence_grounding", "not_applicable", ("scope_has_no_data",)),
        hard_failures=("no_data_confirmed",),
    )

    verification = OutcomeLadder(_InventoryJudge(judgment=no_data)).judge(
        _run("Store 4411 was not found."), meaning_findings=("question_not_answered", "invented_figure")
    )

    assert verification.outcome == "handled_correctly"
    assert verification.final_answer is not None
    assert verification.final_answer.hard_failure_codes == ("no_data_confirmed", "invented_figure")


def test_a_scope_pass_alone_never_verifies_an_answer() -> None:
    verification = OutcomeLadder(_InventoryJudge(judgment=SCOPE_ONLY)).judge(_run())

    assert not verification.verified_success
    assert outcome_of(verification) == "failed"


def test_a_required_criterion_that_does_not_apply_is_dropped_from_the_gate() -> None:
    judgment = RubricJudgment(
        numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("nothing_numeric",)),
        completeness=VerificationCheck("completeness", "passed", ("all_required_values_stated",)),
        scope=VerificationCheck("scope_correctness", "passed", ("requested_scope_present",)),
        grounding=VerificationCheck("evidence_grounding", "passed", ("values_match_agent_tool_results",)),
    )

    verification = OutcomeLadder(_InventoryJudge(judgment=judgment)).judge(_run())

    assert verification.verified_success


def test_the_ladder_rejects_a_rubric_with_other_criteria() -> None:
    rubric = FinalAnswerRubricV1(criteria=(RubricCriterion("helpfulness", 1.0),), required_full_score=())

    with pytest.raises(ValueError, match="exactly these criteria"):
        OutcomeLadder(_InventoryJudge(), rubric=rubric)


def test_extra_hard_failure_codes_are_accepted_by_the_rubric() -> None:
    judgment = RubricJudgment(
        numerical=PASSED.numerical,
        completeness=PASSED.completeness,
        scope=PASSED.scope,
        grounding=VerificationCheck("evidence_grounding", "failed", ("unsupported_link",)),
        hard_failures=("unsupported_link",),
    )

    verification = OutcomeLadder(
        _InventoryJudge(judgment=judgment), extra_hard_failure_codes=("unsupported_link",)
    ).judge(_run())

    assert verification.outcome == "failed"
