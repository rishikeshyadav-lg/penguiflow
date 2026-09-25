"""The order in which a run's outcome is decided, with the domain's judgment as one step of it.

`OutcomeLadder.judge` settles the outcomes that need no value checking first, then asks the
integration's `DomainJudge` to check the answer against what its question required, then applies
the fail-only meaning check:

1. no final answer: failed;
2. an answer cut off mid-sentence: agent error, whatever numbers it reached;
3. an answer asking which of several matching entities was meant: handled correctly;
4. a service the question needs is off: handled correctly;
5. the domain judgment (values, scope, completeness, grounding);
6. meaning findings, which can fail a run but never verify one.

The rubric then requires at least one content criterion to have passed, and drops a required
criterion that does not apply instead of letting it fail every answer.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, Protocol

from ..evaluation.verification import (
    FinalAnswerAssessment,
    FinalAnswerRubricV1,
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from .answer_text import asks_user_to_choose, looks_truncated, shown_to_user
from .expectations import RubricJudgment
from .meaning import MeaningCheck
from .outcomes import all_criteria, judge_rubric, outcome_of
from .runs import AgentRun
from .scope import entity_named

logger = logging.getLogger("learning_control_plane.judging")

_LADDER_CRITERIA = (
    "factual_numerical_correctness",
    "scope_correctness",
    "evidence_grounding",
    "completeness",
    "interpretation_correctness",
)
# Criteria that confirm something about an answer's content; a scope pass alone only says it names the scope.
_SUBSTANTIVE_CRITERIA = (
    "factual_numerical_correctness",
    "evidence_grounding",
    "completeness",
    "interpretation_correctness",
)
# Meaning findings that a correct "no data" answer cannot have earned: saying the scope has no data
# answers the question, and the judge's own query confirmed it.
_FINDINGS_ANSWERED_BY_NO_DATA = frozenset({"question_not_answered", "contradicts_tool_results"})


class DomainJudge(Protocol):
    """The integration's domain knowledge: which steps are safe evidence and whether the answer is right."""

    def step_evidence(self, run: AgentRun) -> Sequence[SafeStepEvidence]:
        """Return allowlisted, content-free evidence for each tool step."""
        ...

    def clarification_candidates(self, run: AgentRun) -> Sequence[str]:
        """Return the entity names an answer might ask the user to choose between, or none."""
        ...

    def service_unavailable(self, run: AgentRun) -> bool:
        """Return whether a service the question needs reported itself switched off."""
        ...

    def judge(self, run: AgentRun, reference: Any) -> RubricJudgment | None:
        """Return the content criteria for this answer, or None when its kind of question cannot be checked."""
        ...


class ReferenceBuilder(Protocol):
    """Compute the judge's own answer to a run's question, independently of the agent's calls."""

    def __call__(self, run: AgentRun) -> Awaitable[Any]:
        """Return the reference, or None when there is none for this question."""
        ...


class VerificationProjector(Protocol):
    """Judge one run in-process and return only safe verification evidence."""

    def __call__(self, run: AgentRun) -> InvestigationVerification:
        """Return redacted checks without retaining the run's content."""
        ...


class OutcomeLadder:
    """Decide a run's verification: the outcomes that need no values first, then the domain, then meaning."""

    def __init__(
        self,
        domain: DomainJudge,
        meaning: MeaningCheck | None = None,
        *,
        rubric: FinalAnswerRubricV1 | None = None,
        extra_hard_failure_codes: Sequence[str] = (),
        is_named: Callable[[str, str], bool] = entity_named,
    ) -> None:
        selected_rubric = rubric or FinalAnswerRubricV1()
        if {criterion.criterion_id for criterion in selected_rubric.criteria} != set(_LADDER_CRITERIA):
            raise ValueError(f"the outcome ladder scores exactly these criteria: {list(_LADDER_CRITERIA)}")
        self._domain = domain
        self._meaning = meaning
        self._rubric = selected_rubric
        self._extra_hard_failure_codes = tuple(extra_hard_failure_codes)
        self._is_named = is_named

    def judge(
        self,
        run: AgentRun,
        reference: Any = None,
        *,
        meaning_findings: Sequence[str] | None = None,
    ) -> InvestigationVerification:
        """Return the verification for one run; `meaning_findings` replaces running the meaning check."""

        verification = InvestigationVerification(
            step_evidence=self._domain.step_evidence(run),
            final_answer=self.final_answer(run, reference, meaning_findings=meaning_findings),
        )
        return replace(verification, outcome=outcome_of(verification))

    def final_answer(
        self,
        run: AgentRun,
        reference: Any = None,
        *,
        meaning_findings: Sequence[str] | None = None,
    ) -> FinalAnswerAssessment:
        """Return the final-answer assessment, deciding each rung of the ladder in order."""

        answer = run.final_answer
        if not isinstance(answer, str) or not answer.strip():
            return score_final_answer(
                all_criteria("failed", "final_answer_missing", self._rubric),
                hard_failure_codes=("primary_result_incorrect",),
                rubric=self._rubric,
            )
        outcome_rubric = judge_rubric(self._rubric)
        if looks_truncated(answer):
            return self._settled(outcome_rubric, "failed", "answer_truncated", "agent_error")
        candidates = self._domain.clarification_candidates(run)
        if candidates and asks_user_to_choose(answer, candidates, is_named=self._is_named):
            return self._settled(outcome_rubric, "not_applicable", "clarification_requested", "clarification_requested")
        if self._domain.service_unavailable(run):
            return self._settled(outcome_rubric, "not_applicable", "service_unavailable", "service_unavailable")

        judgment = self._domain.judge(run, reference)
        if judgment is None:
            return self._settled(outcome_rubric, "not_applicable", "no_domain_judgment", "category_not_verifiable")
        findings = list(meaning_findings) if meaning_findings is not None else self._meaning_codes(run)
        if "no_data_confirmed" in judgment.hard_failures:
            findings = [code for code in findings if code not in _FINDINGS_ANSWERED_BY_NO_DATA]

        criteria = {
            "factual_numerical_correctness": judgment.numerical,
            "scope_correctness": judgment.scope,
            "evidence_grounding": judgment.grounding,
            "completeness": judgment.completeness,
            "interpretation_correctness": VerificationCheck(
                "interpretation_correctness", "not_applicable", ("no_structured_interpretation_expected",)
            ),
        }
        hard_failures = list(judgment.hard_failures)
        if findings:
            criteria["interpretation_correctness"] = VerificationCheck(
                "interpretation_correctness", "failed", tuple(findings)
            )
            hard_failures.extend(findings)
        rubric = requiring_a_substantive_pass(
            excluding_inapplicable_required_criteria(self._rubric, criteria), criteria
        )
        return score_final_answer(
            criteria,
            hard_failure_codes=hard_failures,
            rubric=judge_rubric(rubric, self._extra_hard_failure_codes),
        )

    def _settled(
        self,
        rubric: FinalAnswerRubricV1,
        status: Literal["failed", "not_applicable"],
        reason_code: str,
        hard_failure_code: str,
    ) -> FinalAnswerAssessment:
        """Return an assessment decided before any value was read."""

        return score_final_answer(
            all_criteria(status, reason_code, rubric),
            hard_failure_codes=(hard_failure_code,),
            rubric=rubric,
        )

    def _meaning_codes(self, run: AgentRun) -> list[str]:
        if self._meaning is None:
            return []
        try:
            return list(self._meaning.check(run.question, shown_to_user(run), run.steps).codes)
        except Exception:  # noqa: BLE001 -- the meaning check can only fail runs, so skipping it is safe
            logger.warning("meaning check failed; judging without it", exc_info=True)
            return []


def requiring_a_substantive_pass(
    rubric: FinalAnswerRubricV1, criteria: dict[str, VerificationCheck]
) -> FinalAnswerRubricV1:
    """Keep an answer from verifying unless something about its content was actually confirmed.

    A scope pass only says the answer names what was asked about. When every other criterion does
    not apply, that alone used to verify the answer, so "no data was found for store 4411" counted as
    a verified success. If no content criterion passed, factual correctness is put back into the
    full-score gate; as it does not apply it has no score, so the gate fails.
    """

    if any(criteria[criterion_id].status == "passed" for criterion_id in _SUBSTANTIVE_CRITERIA):
        return rubric
    if "factual_numerical_correctness" in rubric.required_full_score:
        return rubric
    return replace(rubric, required_full_score=(*rubric.required_full_score, "factual_numerical_correctness"))


def excluding_inapplicable_required_criteria(
    rubric: FinalAnswerRubricV1, criteria: dict[str, VerificationCheck]
) -> FinalAnswerRubricV1:
    """Drop a required-full-score criterion from the gate when it does not apply to this answer.

    A criterion that does not apply has no score, so requiring its full score would fail every answer
    with no value to check, however correct. When every criterion is inapplicable the weighted score
    is still 0, so the answer still cannot verify.
    """

    applicable_required = tuple(
        criterion_id for criterion_id in rubric.required_full_score if criteria[criterion_id].status != "not_applicable"
    )
    if applicable_required == tuple(rubric.required_full_score):
        return rubric
    return replace(rubric, required_full_score=applicable_required)


__all__ = [
    "DomainJudge",
    "OutcomeLadder",
    "ReferenceBuilder",
    "VerificationProjector",
    "excluding_inapplicable_required_criteria",
    "requiring_a_substantive_pass",
]
