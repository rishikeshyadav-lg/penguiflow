"""What a judge concludes about a run, beyond "verified" or not.

A run the judge cannot verify is not always a failure. Asking which of several matching entities
was meant, saying the requested scope has no data, or saying a service the question needs is off
are correct answers with nothing to check: they are `handled_correctly`. They are never verified
and so never mined, and pass rates leave them out instead of counting them as failures. An answer
cut off mid-sentence is an `agent_error`, whatever numbers it reached. A run whose kind of question
the judge cannot check at all is `not_judgeable`.

Each outcome is carried as a hard-failure code on the final-answer assessment, so the rubric
arithmetic and every existing reader keep working; `outcome_of` reads it back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from ..evaluation.verification import (
    JUDGE_OUTCOMES,
    FinalAnswerRubricV1,
    InvestigationVerification,
    JudgeOutcome,
    VerificationCheck,
)

HANDLED_CORRECTLY_CODES = ("clarification_requested", "no_data_confirmed", "service_unavailable")
AGENT_ERROR_CODE = "agent_error"
NOT_JUDGEABLE_CODES = ("no_data_to_check", "category_not_verifiable")
# Codes the judge itself may add, beyond a rubric's own hard failures. Meaning-check findings are
# here too, so a finding can fail an otherwise verified run.
JUDGE_HARD_FAILURE_CODES = (
    AGENT_ERROR_CODE,
    *HANDLED_CORRECTLY_CODES,
    *NOT_JUDGEABLE_CODES,
    "scope_unconfirmed",
    "question_not_answered",
    "unsupported_claim",
    "invented_figure",
    "contradicts_tool_results",
)
# Checked in this order: the first code found decides the outcome.
_OUTCOME_BY_CODE: tuple[tuple[str, JudgeOutcome], ...] = (
    (AGENT_ERROR_CODE, "agent_error"),
    *((code, "handled_correctly") for code in HANDLED_CORRECTLY_CODES),
    *((code, "not_judgeable") for code in NOT_JUDGEABLE_CODES),
)


def judge_rubric(base: FinalAnswerRubricV1, extra_hard_failures: Sequence[str] = ()) -> FinalAnswerRubricV1:
    """Return `base` extended to accept the judge's own hard-failure codes."""

    codes = (*base.hard_failure_codes, *extra_hard_failures, *JUDGE_HARD_FAILURE_CODES)
    return replace(base, hard_failure_codes=tuple(dict.fromkeys(codes)))


def all_criteria(
    status: Literal["failed", "not_applicable"],
    code: str,
    rubric: FinalAnswerRubricV1 | None = None,
) -> dict[str, VerificationCheck]:
    """Return the same check for every criterion, for outcomes decided before any value is read."""

    selected = rubric or FinalAnswerRubricV1()
    return {
        criterion.criterion_id: VerificationCheck(criterion.criterion_id, status, (code,))
        for criterion in selected.criteria
    }


def outcome_of(verification: InvestigationVerification) -> JudgeOutcome:
    """Return the judge's outcome for a run: its recorded outcome, or the one its codes imply."""

    if verification.outcome is not None:
        return verification.outcome
    hard_failures = verification.final_answer.hard_failure_codes if verification.final_answer else ()
    for code, outcome in _OUTCOME_BY_CODE:
        if code in hard_failures:
            return outcome
    if verification.verified_success:
        return "verified"
    return "failed"


def outcome_reason_codes(verification: InvestigationVerification) -> list[str]:
    """Return the hard failures, then the reason codes of every criterion that did not pass."""

    final_answer = verification.final_answer
    if final_answer is None:
        return []
    failing = [
        code
        for criterion in final_answer.criteria
        if criterion.status in ("failed", "partial")
        for code in criterion.reason_codes
    ]
    return [*final_answer.hard_failure_codes, *failing]


__all__ = [
    "AGENT_ERROR_CODE",
    "HANDLED_CORRECTLY_CODES",
    "JUDGE_HARD_FAILURE_CODES",
    "JUDGE_OUTCOMES",
    "JudgeOutcome",
    "NOT_JUDGEABLE_CODES",
    "all_criteria",
    "judge_rubric",
    "outcome_of",
    "outcome_reason_codes",
]
