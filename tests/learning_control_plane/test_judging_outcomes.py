from __future__ import annotations

import pytest
from judging_fixtures import INVENTORY_VOCABULARY

from learning_control_plane.evaluation.verification import (
    FinalAnswerRubricV1,
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.judging import (
    all_criteria,
    judge_rubric,
    no_data_judgment,
    nothing_fetched_judgment,
    outcome_of,
    outcome_reason_codes,
)


def _verification(checks: dict[str, VerificationCheck], hard: tuple[str, ...] = ()) -> InvestigationVerification:
    return InvestigationVerification(
        step_evidence=(
            SafeStepEvidence(0, "query_stock", result_checks=(VerificationCheck("tool_execution", "passed"),)),
        ),
        final_answer=score_final_answer(checks, hard_failure_codes=hard, rubric=judge_rubric(FinalAnswerRubricV1())),
    )


def test_every_criterion_passing_is_verified() -> None:
    assert outcome_of(_verification(all_criteria_passed())) == "verified"


def test_a_clarification_is_handled_correctly_not_failed() -> None:
    verification = _verification(
        all_criteria("not_applicable", "clarification_requested"), ("clarification_requested",)
    )

    assert outcome_of(verification) == "handled_correctly"
    assert not verification.verified_success


def test_a_truncated_answer_is_an_agent_error_even_with_other_codes() -> None:
    verification = _verification(all_criteria("failed", "answer_truncated"), ("no_data_confirmed", "agent_error"))

    assert outcome_of(verification) == "agent_error"


def test_a_question_the_judge_cannot_check_is_not_judgeable() -> None:
    verification = _verification(all_criteria_passed(), ("category_not_verifiable",))

    assert outcome_of(verification) == "not_judgeable"


def test_a_wrong_answer_is_failed_with_its_failing_reason_codes() -> None:
    checks = all_criteria_passed()
    checks["factual_numerical_correctness"] = VerificationCheck(
        "factual_numerical_correctness", "failed", ("stated_values_incorrect",)
    )
    verification = _verification(checks, ("primary_result_incorrect",))

    assert outcome_of(verification) == "failed"
    assert outcome_reason_codes(verification) == ["primary_result_incorrect", "stated_values_incorrect"]


def test_a_recorded_outcome_wins_and_is_written_to_the_record() -> None:
    verification = InvestigationVerification(step_evidence=(), outcome="handled_correctly")

    assert outcome_of(verification) == "handled_correctly"
    assert verification.record()["outcome"] == "handled_correctly"


def test_a_verification_without_an_outcome_keeps_its_old_record_shape() -> None:
    assert "outcome" not in InvestigationVerification(step_evidence=()).record()


def test_an_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported judge outcome"):
        InvestigationVerification(step_evidence=(), outcome="maybe")  # type: ignore[arg-type]


def test_a_run_with_no_final_answer_assessment_has_no_reason_codes() -> None:
    verification = InvestigationVerification(step_evidence=())

    assert outcome_reason_codes(verification) == []
    assert outcome_of(verification) == "failed"


def test_the_judge_rubric_accepts_the_judges_own_codes_once() -> None:
    rubric = judge_rubric(FinalAnswerRubricV1(), ("unsupported_link", "wrong_scope"))

    assert "clarification_requested" in rubric.hard_failure_codes
    assert "unsupported_link" in rubric.hard_failure_codes
    assert rubric.hard_failure_codes.count("wrong_scope") == 1


def test_saying_the_scope_has_no_data_is_a_confirmed_no_data_answer() -> None:
    judgment = no_data_judgment("Store 4411 was not found in the stock table.", ["units"], INVENTORY_VOCABULARY)

    assert judgment.hard_failures == ("no_data_confirmed",)


def test_stating_zero_totals_for_a_scope_with_no_data_fails() -> None:
    judgment = no_data_judgment("Store 4411 was not found; units sold: 0.", ["units"], INVENTORY_VOCABULARY)

    assert judgment.hard_failures == ("primary_result_incorrect",)


def test_stating_values_without_saying_there_is_no_data_fails() -> None:
    judgment = no_data_judgment("Store 4411 sold 1,200 units.", ["units"], INVENTORY_VOCABULARY)

    assert judgment.numerical.status == "failed"


def test_an_answer_that_neither_states_values_nor_says_no_data_has_nothing_to_check() -> None:
    judgment = no_data_judgment("Here is some general advice.", ["units"], INVENTORY_VOCABULARY)

    assert judgment.hard_failures == ("no_data_to_check",)


def test_values_nobody_fetched_leave_the_answer_incomplete() -> None:
    judgment = nothing_fetched_judgment()

    assert judgment.completeness.reason_codes == ("required_data_not_fetched",)
    assert judgment.hard_failures == ()


def all_criteria_passed() -> dict[str, VerificationCheck]:
    return {
        criterion.criterion_id: VerificationCheck(criterion.criterion_id, "passed")
        for criterion in FinalAnswerRubricV1().criteria
    }
