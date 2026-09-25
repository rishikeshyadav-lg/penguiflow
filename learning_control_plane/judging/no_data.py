"""Judge an answer when the judge's own query found no data for the question's scope.

Two mistakes this settles. Saying "not found" for a scope that truly has no rows was failed, although
it is the right answer. And an answer that summed the missing rows into zero totals was accepted,
although zeros claim the scope delivered nothing, which a scope with no rows cannot show.

A confirmed "no data" answer is `handled_correctly` (code `no_data_confirmed`): right, but with
nothing to check or learn from. Stating the scope's values, or zero totals, fails. An answer that
does neither has `no_data_to_check`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..evaluation.verification import VerificationCheck
from .answer_facts import MetricVocabulary, stated_values
from .expectations import RubricJudgment

# An answer saying the scope has no data; figures it adds (other entities as suggestions) are not
# claims about the scope.
NO_DATA_PHRASES = re.compile(
    r"\b(not found|no data|no records|no rows|no match|not available|does not appear|doesn't appear|"
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


def nothing_fetched_judgment() -> RubricJudgment:
    """Judge an answer whose question needs values the agent never fetched and the judge could not compute."""

    return RubricJudgment(
        numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("no_required_values_stated",)),
        completeness=VerificationCheck("completeness", "failed", ("required_data_not_fetched",)),
        scope=VerificationCheck("scope_correctness", "not_applicable", ("no_expectation_to_scope",)),
        grounding=VerificationCheck("evidence_grounding", "not_applicable", ("no_matched_values_to_ground",)),
    )


__all__ = ["NO_DATA_PHRASES", "no_data_judgment", "nothing_fetched_judgment"]
