"""Match the values an answer states to the values its question required.

An `Expectation` says which rows the answer must cover (the overall total, each named entity, or
each group of a breakdown) and each required metric's value on each row, and whether those values
came from the judge's own reference query or from the agent's own tool results. `judge_expectation`
then matches the stated values (see `answer_facts`) to it by metric and by row:
- a value stated beside the wrong row or metric no longer counts;
- a rounded value counts when the integration's tolerance accepts it;
- a value the answer gives no row name to can confirm a row but never contradict one;
- a combined total is never compared with one group's row.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..evaluation.verification import VerificationCheck
from .answer_facts import MetricVocabulary, StatedValue, values_in_text
from .scope import squash

# The row key for a question answered by one overall total.
OVERALL = "overall"
_NAME_PIECE_SEPARATOR = re.compile(r"\s+[-–—]\s+")


class Tolerance(Protocol):
    """Decide whether a stated value is close enough to the expected one that a reader would not act differently."""

    def __call__(self, metric: str, stated: float, expected: float, /, *, abbreviated: bool = False) -> bool:
        """Return whether `stated` matches `expected` for this metric."""
        ...


@dataclass(frozen=True, slots=True)
class RelativeTolerance:
    """Match within a share of the expected value, per metric; shorthand like "1.2k" gets a wider share."""

    default: float = 0.005
    per_metric: Mapping[str, float] = field(default_factory=dict)
    abbreviated_share: float = 0.05

    def __call__(self, metric: str, stated: float, expected: float, *, abbreviated: bool = False) -> bool:
        """Return whether `stated` is within this metric's share of `expected`."""

        share = self.per_metric.get(metric, self.default)
        if abbreviated:
            share = max(share, self.abbreviated_share)
        return math.isclose(stated, expected, rel_tol=share, abs_tol=1e-9)


@dataclass(frozen=True, slots=True)
class Expectation:
    """The values an answer must state: row key -> metric -> value, and where they came from.

    `source` is "reference" when the judge computed the values itself and "agent" when they came
    from the agent's own tool results. `label_free` is set for a roll-up (one call per part, each
    filtered differently): the parts' names in the answer need not match the filter text, so any
    stated value may match any part.
    """

    rows: Mapping[str, Mapping[str, float]]
    source: str
    label_free: bool = False


@dataclass(frozen=True, slots=True)
class RubricJudgment:
    """The four content criteria a domain judge decided, and the hard failures it found."""

    numerical: VerificationCheck
    completeness: VerificationCheck
    scope: VerificationCheck
    grounding: VerificationCheck
    hard_failures: tuple[str, ...] = ()


def row_name_forms(key: str, *, suffix_patterns: Sequence[re.Pattern[str]] = ()) -> tuple[str, ...]:
    """Return the ways an answer may name a row: whole, without its suffix, its last part, its trailing pieces.

    "Acme_Store North_2026" may be written "Store North"; "Region - Store North - Late" may be
    written "Store North – Late". The overall row has no name.
    """

    if key == OVERALL:
        return ()
    stripped = key
    for pattern in suffix_patterns:
        stripped = pattern.sub("", stripped)
    stripped = stripped.strip("_ ")
    last_part = stripped.rsplit("_", 1)[-1]
    forms = [squash(key), squash(stripped)]
    if len(squash(last_part)) >= 4:
        forms.append(squash(last_part))
    pieces = _NAME_PIECE_SEPARATOR.split(last_part)
    for start in range(1, len(pieces)):
        tail = squash(" ".join(pieces[start:]))
        if len(tail) >= 6:
            forms.append(tail)
    return tuple(dict.fromkeys(form for form in forms if form))


def row_is_named(text: str, key: str, *, suffix_patterns: Sequence[re.Pattern[str]] = ()) -> bool:
    """Return whether the text names a row in any of its forms."""

    squashed = squash(text)
    return any(form in squashed for form in row_name_forms(key, suffix_patterns=suffix_patterns))


def anchored_values(
    answer: str,
    keys: Sequence[str],
    vocabulary: MetricVocabulary,
    *,
    implicit_metric: str | None = None,
    suffix_patterns: Sequence[re.Pattern[str]] = (),
) -> list[StatedValue]:
    """Return the values in prose that follow a row's name, labelled with that row.

    "North had 1,200 units and South had 800" is cut at every mention of a row's name; each piece up
    to the next mention (or paragraph or heading) is read as being about the row just named.
    """

    mentions: list[tuple[int, int, str]] = []
    for key in keys:
        for form in row_name_forms(key, suffix_patterns=suffix_patterns):
            pattern = re.compile(r"[^a-z0-9]*".join(re.escape(char) for char in form), re.IGNORECASE)
            mentions.extend((match.start(), match.end(), key) for match in pattern.finditer(answer))
    mentions.sort()
    values: list[StatedValue] = []
    for index, (_, end, key) in enumerate(mentions):
        stop = mentions[index + 1][0] if index + 1 < len(mentions) else len(answer)
        piece = re.split(r"\n\s*\n|\n\s*#", answer[end:stop], maxsplit=1)[0]
        values.extend(values_in_text(piece, vocabulary, label=key, implicit_metric=implicit_metric))
    return values


def judge_expectation(
    expectation: Expectation,
    stated: Sequence[StatedValue],
    *,
    shown_text: str,
    tolerance: Tolerance,
    required_metrics: Sequence[str] = (),
    grouped_rows: bool = False,
    needs_own_lookup: bool = False,
    suffix_patterns: Sequence[re.Pattern[str]] = (),
) -> RubricJudgment:
    """Match each required (row, metric) value to what the answer states for that row.

    `grouped_rows` marks a breakdown or ranking, whose rows are groups rather than the question's
    scope. `needs_own_lookup` marks a question about several entities: without the judge's own
    reference it cannot tell a complete list from a partial one, so an agent-sourced expectation
    fails with `scope_unconfirmed`.
    """

    single_row = len(expectation.rows) == 1
    correct = wrong = 0
    missing: list[tuple[str, str]] = []
    # Only the agent can fail to fetch a metric; a metric the reference could not compute is not checked.
    not_fetched = (
        sorted(
            {metric for metric in required_metrics for metrics in expectation.rows.values() if metric not in metrics}
        )
        if expectation.source == "agent"
        else []
    )
    for key, metrics in expectation.rows.items():
        for metric, expected in metrics.items():
            candidates = _candidates(
                stated,
                metric,
                key,
                single_row=single_row,
                label_free=expectation.label_free,
                suffix_patterns=suffix_patterns,
            )
            if any(tolerance(metric, value.value, expected, abbreviated=value.abbreviated) for value in candidates):
                correct += 1
            elif any(value.explicit for value in candidates) and not expectation.label_free:
                wrong += 1
            else:
                missing.append((key, metric))
    if expectation.label_free and correct >= 2:
        missing = []

    entity_keys = [] if expectation.label_free else [key for key in expectation.rows if key != OVERALL]
    unnamed = (
        [key for key in entity_keys if not row_is_named(shown_text, key, suffix_patterns=suffix_patterns)]
        if len(entity_keys) > 1
        else []
    )
    source = expectation.source
    if wrong:
        numerical = VerificationCheck("factual_numerical_correctness", "failed", ("stated_values_incorrect",))
    elif correct:
        numerical = VerificationCheck("factual_numerical_correctness", "passed", (f"stated_values_match_{source}",))
    else:
        numerical = VerificationCheck("factual_numerical_correctness", "not_applicable", ("no_required_values_stated",))

    completeness_codes: list[str] = []
    if not_fetched:
        # The question needs a metric the numbers the judge holds do not have: nobody fetched it.
        completeness_codes.append("required_data_not_fetched")
    if unnamed:
        completeness_codes.append("group_missing_from_answer" if grouped_rows else "campaign_missing_from_answer")
    if missing:
        completeness_codes.append("required_values_missing")
    if completeness_codes:
        completeness = VerificationCheck("completeness", "failed", tuple(completeness_codes))
    else:
        completeness = VerificationCheck("completeness", "passed", ("all_required_values_stated",))

    if grouped_rows or expectation.label_free:
        # Groups are not the question's scope; the caller checks the scope itself.
        scope = VerificationCheck("scope_correctness", "not_applicable", ("scope_checked_separately",))
    elif entity_keys and len(unnamed) == len(entity_keys):
        scope = VerificationCheck("scope_correctness", "failed", ("requested_scope_missing",))
    elif entity_keys:
        scope = VerificationCheck("scope_correctness", "passed", ("requested_scope_present",))
    else:
        scope = VerificationCheck("scope_correctness", "not_applicable", ("single_scope_checked_by_values",))

    if numerical.status == "passed" and source == "reference":
        grounding = VerificationCheck("evidence_grounding", "passed", ("values_match_independent_reference",))
    elif numerical.status == "passed":
        grounding = VerificationCheck("evidence_grounding", "passed", ("values_match_agent_tool_results",))
    else:
        grounding = VerificationCheck("evidence_grounding", "not_applicable", ("no_matched_values_to_ground",))

    hard_failures: list[str] = []
    if numerical.status == "failed":
        hard_failures.append("primary_result_incorrect")
    if scope.status == "failed":
        hard_failures.append("wrong_scope")
    if source == "agent" and needs_own_lookup:
        hard_failures.append("scope_unconfirmed")
    return RubricJudgment(numerical, completeness, scope, grounding, tuple(hard_failures))


def judgment_passes(judgment: RubricJudgment) -> bool:
    """Return whether a judgment confirms the answer: a matched value and nothing missing or wrong."""

    return (
        judgment.numerical.status == "passed"
        and judgment.completeness.status == "passed"
        and not judgment.hard_failures
    )


def _candidates(
    values: Sequence[StatedValue],
    metric: str,
    key: str,
    *,
    single_row: bool,
    label_free: bool,
    suffix_patterns: Sequence[re.Pattern[str]],
) -> list[StatedValue]:
    """Return the stated values that could be the answer's value for this row and metric.

    A total is compared only with total-level statements (a labelled line, a metric table, a "Total"
    row), never with one group's row. A group or entity is compared with values labelled by its
    name; when the answer covers a single row, its total-level statements count too.
    """

    same_metric = [value for value in values if value.metric == metric]
    if label_free:
        return same_metric
    if key == OVERALL:
        return [value for value in same_metric if not value.row]
    named = [value for value in same_metric if _names_row(value.label, key, suffix_patterns)]
    if single_row:
        named.extend(value for value in same_metric if not value.row and value not in named)
    else:
        # A value the answer gives no row name to can still confirm a row, but never contradict it.
        named.extend(
            StatedValue(
                value.metric, value.value, value.abbreviated, value.label, value.source, value.row, explicit=False
            )
            for value in same_metric
            if not value.label and value not in named
        )
    return named


def _names_row(label: str, key: str, suffix_patterns: Sequence[re.Pattern[str]]) -> bool:
    squashed = squash(label)
    if not squashed:
        return False
    return any(
        form in squashed or (len(squashed) >= 4 and squashed in form)
        for form in row_name_forms(key, suffix_patterns=suffix_patterns)
    )


__all__ = [
    "OVERALL",
    "Expectation",
    "RelativeTolerance",
    "RubricJudgment",
    "Tolerance",
    "anchored_values",
    "judge_expectation",
    "judgment_passes",
    "row_is_named",
    "row_name_forms",
]
