"""Judge an answer when the judge's own query found no data for the question's scope.

Two mistakes this settles. Saying "not found" for a scope that truly has no rows was failed, although
it is the right answer. And an answer that summed the missing rows into zero totals was accepted,
although zeros claim the scope delivered nothing, which a scope with no rows cannot show.

A confirmed "no data" answer is `handled_correctly` (code `no_data_confirmed`): right, but with
nothing to check or learn from. Stating the scope's values, or zero totals, fails. An answer that
does neither has `no_data_to_check`.

A question with no reference query (a trend, a period-over-period change) can still have a
confirmed "no data" answer: `EmptyLookupRule` says which of the agent's calls looked the question's
scope up and what an empty result looks like, and `no_data_confirmed_by_empty_lookups` accepts the
answer when every such lookup came back empty.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..evaluation.verification import VerificationCheck
from .answer_facts import MetricVocabulary, stated_values
from .expectations import RubricJudgment
from .runs import AgentRun, AgentStep
from .steps import step_failed

# An answer saying the scope has no data; figures it adds (other entities as suggestions) are not
# claims about the scope.
NO_DATA_PHRASES = re.compile(
    r"\b(not found|no data|no records|no rows|no match(?:es|ing)?|not available|does not appear|doesn't appear|"
    r"(unable|could not|couldn't|was unable) to (find|locate)|returns? (zero|no) (results|rows))\b",
    re.IGNORECASE,
)


def no_data_judgment(
    answer: str,
    required_metrics: Sequence[str],
    vocabulary: MetricVocabulary,
    *,
    phrases: re.Pattern[str] = NO_DATA_PHRASES,
) -> RubricJudgment:
    """Judge an answer to a question whose scope the judge confirmed has no data."""

    says_no_data = bool(phrases.search(answer))
    required_values = [
        value for value in stated_values(answer, vocabulary) if value.metric in required_metrics and value.explicit
    ]
    states_zero_totals = any(value.value == 0 for value in required_values)
    if (required_values and not says_no_data) or states_zero_totals:
        return RubricJudgment(
            numerical=VerificationCheck(
                "factual_numerical_correctness", "failed", ("values_stated_for_scope_with_no_data",)
            ),
            completeness=VerificationCheck("completeness", "not_applicable", ("scope_has_no_data",)),
            scope=VerificationCheck("scope_correctness", "not_applicable", ("scope_has_no_data",)),
            grounding=VerificationCheck("evidence_grounding", "failed", ("values_stated_for_scope_with_no_data",)),
            hard_failures=("primary_result_incorrect",),
        )
    return RubricJudgment(
        numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("scope_has_no_data",)),
        completeness=VerificationCheck("completeness", "not_applicable", ("scope_has_no_data",)),
        scope=VerificationCheck("scope_correctness", "not_applicable", ("scope_has_no_data",)),
        grounding=VerificationCheck("evidence_grounding", "not_applicable", ("scope_has_no_data",)),
        hard_failures=("no_data_confirmed",) if says_no_data else ("no_data_to_check",),
    )


# Meaning findings a correct "no data" answer cannot have earned: saying the scope has no data answers
# the question, and nothing found contradicts it.
FINDINGS_ANSWERED_BY_NO_DATA = frozenset({"question_not_answered", "contradicts_tool_results"})


@dataclass(frozen=True, slots=True)
class EmptyLookupRule:
    """Which calls look up the question's scope, and when one of them found nothing.

    `is_scoped_lookup` marks a data call restricted to what the question named (for example one
    with filters); `returned_nothing` marks its result as empty. `reference_found_data` says whether
    the judge's own reference, when there is one, found rows.
    """

    is_scoped_lookup: Callable[[AgentStep], bool]
    returned_nothing: Callable[[AgentStep], bool]
    vocabulary: MetricVocabulary
    reference_found_data: Callable[[Any], bool] = bool
    phrases: re.Pattern[str] = NO_DATA_PHRASES


def no_data_confirmed_by_empty_lookups(
    run: AgentRun, rule: EmptyLookupRule, *, reference: Any, findings: Sequence[str]
) -> bool:
    """Return whether a "no data" answer is confirmed by the agent's own empty lookups.

    The answer must say there is no data and state no zero totals; every scoped lookup must have
    come back empty; the judge's reference must not have found rows; and no meaning finding other
    than "not answered" or "contradiction" may apply. Such a run is handled correctly, never
    verified or mined, so a wrong call here only changes how a run is reported.
    """

    answer = run.final_answer or ""
    if reference is not None and rule.reference_found_data(reference):
        return False
    if not rule.phrases.search(answer):
        return False
    if any(code not in FINDINGS_ANSWERED_BY_NO_DATA for code in findings):
        return False
    if any(value.explicit and value.value == 0 for value in stated_values(answer, rule.vocabulary)):
        return False
    scoped_lookups = [step for step in run.steps if rule.is_scoped_lookup(step) and not step_failed(step)]
    return bool(scoped_lookups) and all(rule.returned_nothing(step) for step in scoped_lookups)


def nothing_fetched_judgment() -> RubricJudgment:
    """Judge an answer whose question needs values the agent never fetched and the judge could not compute."""

    return RubricJudgment(
        numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("no_required_values_stated",)),
        completeness=VerificationCheck("completeness", "failed", ("required_data_not_fetched",)),
        scope=VerificationCheck("scope_correctness", "not_applicable", ("no_expectation_to_scope",)),
        grounding=VerificationCheck("evidence_grounding", "not_applicable", ("no_matched_values_to_ground",)),
    )


__all__ = [
    "EmptyLookupRule",
    "FINDINGS_ANSWERED_BY_NO_DATA",
    "NO_DATA_PHRASES",
    "no_data_confirmed_by_empty_lookups",
    "no_data_judgment",
    "nothing_fetched_judgment",
]
